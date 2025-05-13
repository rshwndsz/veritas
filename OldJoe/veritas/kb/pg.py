import asyncio
import csv
import io
import logging
from dataclasses import asdict
from typing import Optional, Tuple

from aiofile import async_open
from attr import dataclass
from openai import AsyncOpenAI
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from veritas.kb.chunk import Chunk

logger = logging.getLogger("veritas")


@dataclass
class RetrievalConfig:
    k_semantic_chunks: int
    k_keyword_chunks: int
    keyword_penalty: int
    semantic_penalty: int
    k: int
    prefix: Optional[str]


class DB:
    def __init__(
        self,
        db_name: str,
        embedding_dim: int,
        db_user: str = "dsouzars",
        db_password: str = "POSTGRES",
        db_host: str = "localhost",
        db_port: int = 5432,
        db_connections: int = 4,
        pool_timeout: int = 120,
    ):
        # DB params
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_connections = db_connections
        self.pool_timeout = pool_timeout

        # Embedding params
        self.emb_dim = embedding_dim

        # Create an async connection pool
        conninfo = (
            f"dbname={self.db_name} user={self.db_user} "
            f"password={self.db_password} host={self.db_host} port={self.db_port}"
        )
        logger.info(
            f"Creating async DB pool for '{self.db_name}' as user '{self.db_user}' "
            f"with pool size {self.db_connections} and timeout {self.pool_timeout}s"
        )
        # Note: Pool creation itself is synchronous, but connections are async
        self.pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=self.db_connections,
            timeout=self.pool_timeout,
        )

        self._pool_initialized = asyncio.Event()  # Event to signal pool readiness

    async def _async_pool_init(self):
        """Asynchronously initialize pool resources like extensions."""
        # This method should be called after pool creation or via the 'open' hook if supported
        # For now, call it explicitly in __aenter__ or a dedicated init method.
        async with self.pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector_async(conn)
        self._pool_initialized.set()

    async def __aenter__(self):
        # Ensure pool is initialized before use
        if not self._pool_initialized.is_set():
            await self._async_pool_init()
        await self._pool_initialized.wait()  # Wait if initialization is in progress
        logger.info("Async DB pool initialized with pgvector.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
            logger.info("Async DB pool closed.")

    async def table_exists(self, table_name: str) -> bool:
        await self._pool_initialized.wait()  # Ensure pool ready
        async with self.pool.connection() as conn:
            query = sql.SQL("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = {tn})").format(
                tn=sql.Literal(table_name)
            )
            result = await conn.execute(query)
            row = await result.fetchone()
            return row is not None and row[0]

    async def create(self, table_name: str):
        async def _table_exists_internal(conn, table_name: str) -> bool:
            # Internal check using an existing async connection.
            query = sql.SQL("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = {tn})").format(
                tn=sql.Literal(table_name)
            )
            result = await conn.execute(query)
            row = await result.fetchone()
            return row is not None and row[0]

        await self._pool_initialized.wait()  # Ensure pool ready
        # Check existence first
        if await self.table_exists(table_name):
            logger.debug(f"Table '{table_name}' already exists.")
            return

        # If it doesn't exist, attempt creation
        async with self.pool.connection() as conn:
            # Double-check inside the connection context
            if not await _table_exists_internal(conn, table_name):
                # Create the table
                logger.info(f" Creating new table '{table_name}'")
                query = sql.SQL(
                    """
                    CREATE TABLE {tn} (
                       id bigserial PRIMARY KEY,
                       doc_id text,
                       source_url text,
                       chunk_index text,
                       content text,
                       embedding vector({es}),
                       UNIQUE (doc_id, chunk_index)
                    );
                    CREATE INDEX ON {tn} USING GIN (to_tsvector('english', content));
                    CREATE INDEX ON {tn} USING hnsw (embedding vector_cosine_ops); -- Example HNSW index
                    """
                ).format(tn=sql.Identifier(table_name), es=sql.Literal(self.emb_dim))
                await conn.execute(query)
                await conn.commit()
                logger.info(f" Successfully created table '{table_name}' with indexes.")
            else:
                logger.debug(f"Table '{table_name}' was created concurrently.")

    async def drop(self, table_name: str):
        await self._pool_initialized.wait()  # Ensure pool ready
        logger.warning(f"Dropping table '{table_name}'!")
        query = sql.SQL("DROP TABLE IF EXISTS {tn}").format(tn=sql.Identifier(table_name))
        async with self.pool.connection() as conn:
            await conn.execute(query)
            await conn.commit()

    async def nuke(self):
        await self._pool_initialized.wait()  # Ensure pool ready
        logger.warning("Nuking database. You have 5 seconds to cancel!")
        await asyncio.sleep(5)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Fetch all table names in the given schema
                await cur.execute("""SELECT tablename FROM pg_tables WHERE schemaname = %s; """, ("public",))
                tables = await cur.fetchall()
                for (table_name,) in tables:
                    logger.warning(f" Dropping table: {table_name}")
                    query = sql.SQL("DROP TABLE IF EXISTS {tn} CASCADE").format(tn=sql.Identifier(table_name))
                    await cur.execute(query)
            await conn.commit()

    async def insert(self, chunks: list[Chunk], table: str) -> None:
        await self._pool_initialized.wait()  # Ensure pool ready

        for chunk in chunks:
            if not chunk.embedding:
                # Remove chunk from batch and log an error
                logger.error(f"Chunk with ID {chunk.doc_id} has no embedding. Skipping this chunk.")
                chunks.remove(chunk)
                continue
            if len(chunk.embedding) != self.emb_dim:
                # Remove chunk from batch and log an error
                logger.error(
                    f"Chunk with embedding of length {len(chunk.embedding)} does not match expected dimension {self.emb_dim}. "
                    f"Skipping this chunk."
                )
                chunks.remove(chunk)

        # Build values
        values = [{**asdict(chunk)} for chunk in chunks]
        if not values:
            logger.warning(f"Received empty chunk list for insertion into '{table}'.")
            return

        inserted_count = 0
        # Insert into DB asynchronously
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL(
                    """
                    INSERT INTO {tn} (doc_id, source_url, chunk_index, content, embedding)
                    VALUES ({doc_id}, {source_url}, {chunk_index}, {content}, {embedding})
                    ON CONFLICT (doc_id, chunk_index) DO NOTHING
                    """
                ).format(
                    tn=sql.Identifier(table),
                    doc_id=sql.Placeholder("doc_id"),
                    source_url=sql.Placeholder("source_url"),
                    chunk_index=sql.Placeholder("chunk_index"),
                    content=sql.Placeholder("content"),
                    embedding=sql.Placeholder("embedding"),
                )
                # Use executemany for batch insertion
                await cur.executemany(query, values)
                inserted_count = cur.rowcount  # Get number of rows actually inserted
            await conn.commit()  # Commit transaction

        # Log actual number inserted
        if inserted_count < len(chunks):
            logger.info(f"Inserted {inserted_count}/{len(chunks)} chunks into table '{table}' (duplicates skipped).")
        else:
            logger.info(f"Inserted {inserted_count} chunks into table '{table}'.")

    async def insert_fast(self, chunks: list[Chunk], table: str) -> None:
        await self._pool_initialized.wait()  # Ensure pool ready

        # Assert all chunks have embeddings
        if not all(chunk.embedding for chunk in chunks):
            logger.error(
                f"Attempted to insert chunks without embeddings into '{table}'. Skipping insert for this batch."
            )
            raise ValueError(f"All chunks must have embeddings for table '{table}'")

        # Build values for CSV (Ensure string fields are properly quoted/escaped)
        values = [{**asdict(chunk)} for chunk in chunks]
        if not values:
            logger.warning(f"Received empty chunk list for insertion into '{table}'.")
            return

        # Prepare CSV data in-memory
        output = io.StringIO()
        # Use QUOTE_ALL to handle potential special characters in content better
        csv_writer = csv.writer(output, quoting=csv.QUOTE_ALL)

        # Write header
        csv_writer.writerow(["doc_id", "source_url", "chunk_index", "content", "embedding"])

        # Write rows
        for row in values:
            # Ensure embedding is formatted correctly for pgvector COPY (string representation)
            embedding_str = str(row["embedding"]) if row["embedding"] else None
            csv_writer.writerow([row["doc_id"], row["source_url"], row["chunk_index"], row["content"], embedding_str])

        # Get the CSV data as a string
        csv_data = output.getvalue()
        output.close()

        # Use COPY command to insert data
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = sql.SQL("""
                    COPY {tn} (doc_id, source_url, chunk_index, content, embedding)
                    FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                """).format(tn=sql.Identifier(table))

                # Use `async with cur.copy()` for async data transfer
                async with cur.copy(query) as copy:
                    # Encode the CSV data to bytes and write it asynchronously
                    await copy.write(csv_data.encode("utf-8"))
            await conn.commit()

        logger.info(f"Inserted {len(values)} chunks into table '{table}' using COPY.")

    async def insert_csv(self, csv_fpath: str, table: str, chunk_size: int = 65536) -> None:
        """Performs a fast bulk insert into the specified table from a CSV file using COPY FROM."""
        await self._pool_initialized.wait()  # Ensure pool ready
        logger.info(f"Starting bulk insert into '{table}' from CSV file: {csv_fpath} in chunks of {chunk_size} bytes.")

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Construct the COPY query
                    query = sql.SQL("""
                        COPY {tn} (doc_id, source_url, chunk_index, content, embedding)
                        FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                    """).format(tn=sql.Identifier(table))

                    # Use async cursor copy to stream data from the file
                    async with cur.copy(query) as copy:
                        # Open the file asynchronously in binary mode
                        async with async_open(csv_fpath, mode="rb") as f:
                            # Read the file in chunks and write to the COPY stream
                            while True:
                                chunk = await f.read(chunk_size)
                                if not chunk:
                                    break
                                await copy.write(chunk)

                await conn.commit()
                # Note: COPY FROM STDIN doesn't easily return row count in psycopg async.
                # We log success based on completion without errors.
                logger.info(f"Successfully completed bulk insert into '{table}' from {csv_fpath} using COPY.")

        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_fpath}")
            raise
        except Exception as e:
            logger.error(f"Error during CSV bulk insert into '{table}' from {csv_fpath}: {e}")
            # TODO Possible rollback/retry
            raise

    async def retrieve(
        self, embedding_client: AsyncOpenAI, embedding_model: str, queries: list[str], table: str, cfg: RetrievalConfig
    ) -> Tuple[list[Chunk], list[float]]:
        await self._pool_initialized.wait()  # Ensure pool ready
        # Embed
        query_texts = [cfg.prefix + q if cfg.prefix else q for q in queries]
        res = await embedding_client.embeddings.create(model=embedding_model, input=query_texts)
        texts = query_texts
        embeddings = [d.embedding for d in res.data]

        # Build query
        # This SQL performs multi-query retrieval using both semantic and keyword search,
        # all in a single query. Input is a list of (query_text, query_embedding) pairs.
        #
        # Step 1: We unpack this list using `unnest()` into a temporary table named `input_queries`
        # which has the following structure:
        #
        #     input_queries (qid, qtext, qembedding)
        #
        # where `qid` is the index (0-based) of the original query. For example:
        #
        #     queries = ["how to bake bread", "yeast temperature"]
        #     embeddings = [[...], [...]]
        #
        # becomes:
        #
        #     qid |         qtext         |      qembedding
        #     ----+------------------------+-----------------------
        #      0  | "how to bake bread"    | [0.01, 0.42, ...]
        #      1  | "yeast temperature"    | [0.12, 0.07, ...]
        #
        # Step 2: The `semantic_search` CTE does vector search using pgvector’s `<=>` operator.
        # For each (qid, embedding) in `input_queries`, we rank the chunks in the target table
        # by their distance to the query embedding. The result has:
        #
        #     semantic_search (qid, id, rank, score)
        #
        # where rank is per-qid using `RANK() OVER (PARTITION BY qid ORDER BY distance ASC)`,
        # and score is computed as `1 / (k + rank)` to smoothly decay with rank. For example:
        #
        #     qid | id  | rank | score
        #     ----+-----+------+-------
        #      0  | 42  | 1    | 0.090
        #      0  | 77  | 2    | 0.083
        #      1  | 77  | 1    | 0.090
        #
        # Note that the same chunk ID (77) may appear for multiple queries.
        #
        # Step 3: The `keyword_search` CTE does a parallel full-text search using
        # `plainto_tsquery(qtext)` against the `content` column. Results are again ranked per-qid
        # and scored with the same formula.
        #
        # Step 4: We union the two CTEs (`semantic_search` and `keyword_search`) together.
        # At this point, each (qid, id) pair may have multiple scores — one from semantic,
        # one from keyword, or both.
        #
        # Step 5: In the `combined` CTE, we group by chunk ID (`id`) and sum scores across
        # all matching queries. That means chunks that are relevant to more than one query
        # get boosted naturally. For example:
        #
        #     id  | total_score
        #     ----+-------------
        #     77  | 0.173  -- matched qid 0 and qid 1
        #     42  | 0.090
        #
        # Step 6: We join back to the chunks table using the `id` to fetch full metadata
        # (doc_id, source_url, chunk_index, content) and sort by the aggregated score.
        #
        # The final result is a flat list of top-k chunks that are most relevant *across*
        # all queries, with soft reranking from both vector distance and keyword match.
        s = sql.SQL("""
        WITH
        input_queries AS (
            SELECT
                qid,
                qtext,
                qembedding
            FROM
                UNNEST(%(qtexts)s::text[], %(qembeds)s::vector[]) WITH ORDINALITY
                AS t(qtext, qembedding, qid)
        ),
        semantic_search AS (
            SELECT
                t.id,
                iq.qid,
                1.0 / (%(srpenalty)s + RANK() OVER (PARTITION BY iq.qid ORDER BY t.embedding <=> iq.qembedding)) AS score
            FROM {tname} t, input_queries iq
            ORDER BY t.embedding <=> iq.qembedding
            LIMIT %(slimit)s
        ),
        keyword_search AS (
            SELECT
                t.id,
                iq.qid,
                1.0 / (%(krpenalty)s + RANK() OVER (PARTITION BY iq.qid ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', iq.qtext)) DESC)) AS score
            FROM {tname} t, input_queries iq
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', iq.qtext)
            ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', iq.qtext)) DESC
            LIMIT %(klimit)s
        ),
        combined AS (
            SELECT id, SUM(score) AS total_score
            FROM (
                SELECT * FROM semantic_search
                UNION ALL
                SELECT * FROM keyword_search
            ) s
            GROUP BY id
        )
        SELECT
            t.doc_id, t.source_url, t.chunk_index, t.content, c.total_score
        FROM combined c
        JOIN {tname} t ON t.id = c.id
        ORDER BY c.total_score DESC
        LIMIT %(topk)s
        """).format(tname=sql.Identifier(table))

        # Run query asynchronously
        async with self.pool.connection() as conn:
            result = await conn.execute(
                s,
                {
                    "qtexts": texts,
                    "qembeds": [str(e) for e in embeddings],  # pgvector needs string representation
                    "slimit": cfg.k_semantic_chunks,
                    "srpenalty": cfg.semantic_penalty,
                    "klimit": cfg.k_keyword_chunks,
                    "krpenalty": cfg.keyword_penalty,
                    "topk": cfg.k,
                },
            )
            rows = await result.fetchall()

        # Build chunks
        chunks = [
            Chunk(doc_id=row[0], source_url=row[1], chunk_index=row[2], content=row[3]) for row in rows
        ]  # No need to return embeddings
        scores = [row[4] for row in rows]
        return chunks, scores

    async def close(self):
        """Explicitly close the pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Async DB pool closed explicitly.")
