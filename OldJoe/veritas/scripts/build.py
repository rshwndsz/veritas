import asyncio
import csv
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
from asyncio import Queue as AsyncQueue
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import hydra
import requests
from aiofile import async_open
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from openai import APIError, AsyncOpenAI, RateLimitError
from rich.console import Console
from semantic_text_splitter import TextSplitter

from veritas.kb.chunk import Chunk, Document, chunk_document

logger = logging.getLogger("veritas")
console = Console()


@dataclass
class ModelConfig:
    model_id: str
    base_url: str
    api_key: str = "EMPTY"
    max_retries: int = 3
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 512


@dataclass
class BuildConfig:
    evidence_dir: str
    embedding_model: ModelConfig
    embedding_prefix: Optional[str] = None
    chunk_size: int = 2048
    batch_size: int = 1024
    rate_limit: int = 8192
    max_queue_size: int = 2048
    results_file_dir: str = "./results"
    limit: Optional[int] = None
    n_gpus: int = 1
    n_processes_per_gpu: int = 1
    base_port: int = 1775
    gpu_ids: Optional[List[int]] = None
    server_startup_delay: int = 5
    csv_batch_size: int = 8192


cs = ConfigStore.instance()
cs.store(name="config", node=BuildConfig)


async def csv_writer_task(
    writer_id: str,
    queue: AsyncQueue[Optional[Tuple[List[Chunk], str]]],
    stop_event: asyncio.Event,
    results_dir: str,
    csv_batch_size: int,
) -> bool:
    """Asynchronous task to write chunks from its dedicated queue to a specific CSV file."""
    logger.info(f"CSV Writer for '{writer_id}.csv' task started (batch size: {csv_batch_size}).")

    # Buffer to accumulate chunks until they reach batch size
    accumulated_chunks: List[Chunk] = []

    try:
        while not stop_event.is_set():
            # Get next item from queue
            item = await queue.get()

            # Handle sentinel
            if item is None:
                logger.info(f"CSV Writer for '{writer_id}.csv' received sentinel, finishing.")
                # Write any remaining accumulated chunks before exiting
                if accumulated_chunks:
                    logger.info(
                        f"CSV Writer for '{writer_id}.csv' writing final {len(accumulated_chunks)} accumulated chunks."
                    )
                    try:
                        await write(accumulated_chunks, writer_id, results_dir, stop_event)
                    except Exception as e_csv:
                        logger.critical(
                            f"CSV Writer for '{writer_id}.csv' failed to write final chunks: {e_csv}", exc_info=True
                        )
                        stop_event.set()
                        queue.task_done()
                        return False
                queue.task_done()
                break

            # Validate item structure
            if not isinstance(item, tuple) or len(item) != 2:
                logger.error(f"CSV Writer for '{writer_id}.csv' received invalid item type: {type(item)}. Skipping.")
                queue.task_done()
                continue

            chunks, table_name = item

            # Sanity check: table_name should match writer_id
            if table_name != writer_id:
                logger.error(
                    f"CSV Writer for '{writer_id}.csv' received item for wrong table '{table_name}'. Skipping."
                )
                queue.task_done()
                continue

            # Skip empty chunks
            if not chunks:
                queue.task_done()
                continue

            # Add received chunks to our accumulation buffer
            accumulated_chunks.extend(chunks)
            logger.debug(
                f"CSV Writer for '{writer_id}.csv' accumulated {len(chunks)} chunks "
                f"(total: {len(accumulated_chunks)}/{csv_batch_size})"
            )

            # Only write when we've accumulated enough chunks
            if len(accumulated_chunks) >= csv_batch_size:
                logger.debug(
                    f"CSV Writer for '{writer_id}.csv' batch threshold reached ({len(accumulated_chunks)} chunks)."
                )
                try:
                    await write(accumulated_chunks, writer_id, results_dir, stop_event)
                    # Clear buffer after successful write
                    accumulated_chunks = []
                except Exception as e_csv:
                    logger.critical(
                        f"CSV Writer for '{writer_id}.csv' failed to write batched chunks: {e_csv}", exc_info=True
                    )
                    stop_event.set()
                    queue.task_done()
                    return False

            queue.task_done()

    except asyncio.CancelledError:
        logger.warning(f"CSV Writer for '{writer_id}.csv' task cancelled.")
        # Try to write any remaining chunks before exiting
        if accumulated_chunks and not stop_event.is_set():
            try:
                await write(accumulated_chunks, writer_id, results_dir, stop_event)
                logger.info(
                    f"CSV Writer for '{writer_id}.csv' wrote {len(accumulated_chunks)} chunks before cancellation."
                )
            except Exception:
                logger.warning(f"CSV Writer for '{writer_id}.csv' failed to write final chunks during cancellation.")
        return True  # Cancellation is not a failure

    except Exception as e:
        logger.critical(
            f"CSV Writer for '{writer_id}.csv' failed during initialization or critical operation: {e}", exc_info=True
        )
        stop_event.set()  # Signal critical failure
        return False  # Indicate failure
    finally:
        logger.info(f"CSV Writer for '{writer_id}.csv' task finished.")

    return True


async def write(chunks: List[Chunk], table_name: str, results_dir: str, stop_event: asyncio.Event):
    """Writes a list of chunks to a CSV file named after the table.
    Assumes only one writer calls this per table_name.
    """

    def _write(csv_data: str, csv_fpath: Path):
        # Synchronous write of a string to a file
        try:
            with open(csv_fpath, "a", encoding="utf-8") as f:
                f.write(csv_data)
        except Exception as e:
            logger.error(f"Error during sync CSV write to {csv_fpath}: {e}")
            raise

    if stop_event.is_set():
        logger.warning(f"CSV write for {table_name} skipped due to stop event.")
        return

    csv_fpath = Path(results_dir) / f"{table_name}.csv"
    try:
        # Check if file exists and is empty to write header
        file_exists = csv_fpath.exists()
        is_empty = not file_exists or csv_fpath.stat().st_size == 0

        # Prepare buffer in memory first
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)

        # Write header if file is empty
        if is_empty:
            writer.writerow(["doc_id", "chunk_index", "content", "source_url", "embedding"])

        # Write all chunks to memory at once
        for chunk in chunks:
            embeddings = getattr(chunk, "embedding", None)
            embedding_str = json.dumps(embeddings) if embeddings is not None else json.dumps([])
            row = [
                getattr(chunk, "doc_id", ""),
                str(getattr(chunk, "chunk_index", "")),
                getattr(chunk, "content", ""),
                getattr(chunk, "source_url", ""),
                embedding_str,
            ]
            writer.writerow(row)

        # Get full CSV data as string
        csv_data = output.getvalue()
        output.close()

        # Use await asyncio.to_thread for file I/O to avoid blocking the event loop
        await asyncio.to_thread(_write, csv_data, csv_fpath)
        logger.debug(f"Successfully wrote {len(chunks)} chunks to {csv_fpath}")

    except Exception as e:
        logger.error(f"Failed to write chunks to CSV '{csv_fpath}': {e}", exc_info=True)
        raise


def start_servers(config: BuildConfig) -> List[Dict]:
    servers = []
    gpu_ids = config.gpu_ids if config.gpu_ids else list(range(config.n_gpus))
    parsed_url = urlparse(config.embedding_model.base_url)
    hostname = parsed_url.hostname or "localhost"
    server_startup_delay = config.server_startup_delay

    server_id_counter = 0
    for gpu_id in gpu_ids:
        for _ in range(config.n_processes_per_gpu):
            port = config.base_port + server_id_counter
            api_url = f"{parsed_url.scheme or 'http'}://{hostname}:{port}/v1"
            health_url = f"{parsed_url.scheme or 'http'}://{hostname}:{port}/health"

            # Base command to start vLLM server
            # Keep formatter off to preserve this layout
            # fmt: off
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", config.embedding_model.model_id,
                "--host", hostname, "--port", str(port), "--api-key", config.embedding_model.api_key,
                "--trust-remote-code", "--max-num-seqs", str(config.embedding_model.max_num_seqs),
                "--gpu-memory-utilization", str(config.embedding_model.gpu_memory_utilization),
                "--max-model-len", str(config.embedding_model.max_model_len),
                "--disable-log-requests",
            ]
            # fmt: on

            logger.info(
                f"Starting vLLM server #{server_id_counter + 1}"
                f" on GPU {gpu_id}"
                f" at {api_url}: {' '.join(map(str, cmd))}"
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Use Popen with process group leader capability
            process = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
            servers.append(
                {
                    "process": process,
                    "port": port,
                    "gpu_id": gpu_id,
                    "server_id": server_id_counter,
                    "api_url": api_url,
                    "health_url": health_url,
                }
            )
            server_id_counter += 1
            if server_startup_delay > 0:
                logger.info(f"Waiting {server_startup_delay}s before starting next server...")
                time.sleep(server_startup_delay)

    return servers


def wait_for_servers(servers: List[Dict], max_wait_time: int = 300, check_interval: int = 5) -> List[Dict]:
    logger.info(f"Waiting for {len(servers)} vLLM server(s) to become healthy...")
    start_time = timer()
    ready_servers = []
    pending_servers = list(servers)  # Copy the list

    with console.status(f"Waiting for servers... 0/{max_wait_time}s", spinner="dots") as status:
        while pending_servers and (timer() - start_time < max_wait_time):
            elapsed = timer() - start_time
            status.update(
                f"Waiting for servers... {len(ready_servers)}/{len(servers)} ready. {elapsed:.1f}/{max_wait_time}s"
            )

            for server in pending_servers[:]:  # Iterate over a copy for safe removal
                # First, check if the process terminated unexpectedly
                if server["process"].poll() is not None:
                    logger.error(
                        f"Server #{server['server_id']} (Port {server['port']}) terminated prematurely"
                        f" with code {server['process'].returncode}."
                    )
                    pending_servers.remove(server)
                    continue  # Don't try to check health

                # Check health endpoint
                try:
                    response = requests.get(server["health_url"], timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✓ Server #{server['server_id']} (Port {server['port']}) is healthy.")
                        ready_servers.append(server)
                        pending_servers.remove(server)
                except (requests.RequestException, ConnectionError) as e:
                    # Server not yet responding or connection error
                    logger.debug(f"Server #{server['server_id']} (Port {server['port']}) not ready yet: {e}")
                    pass  # Continue waiting

            if pending_servers:
                time.sleep(check_interval)

    # Final check for any remaining pending servers that might have failed
    for server in pending_servers:
        if server["process"].poll() is not None:
            logger.error(f"Server #{server['server_id']} (Port {server['port']}) terminated prematurely during wait.")
        else:
            logger.warning(
                f"❌ Server #{server['server_id']} (Port {server['port']}) "
                f" did not become healthy within {max_wait_time} seconds."
            )

    if not ready_servers:
        logger.error("No servers became healthy.")
    else:
        logger.info(f"All {len(ready_servers)} required servers are healthy.")

    return ready_servers


def cleanup_servers(servers: List[Dict]):
    logger.info(f"Attempting to shut down {len(servers)} vLLM server processes...")
    for server in servers:
        if server.get("process") and server["process"].poll() is None:  # Check if process exists and is running
            pid = server["process"].pid
            pgid = os.getpgid(pid)
            logger.info(f"Sending SIGTERM to process group {pgid} for server #{server['server_id']} (PID {pid})")
            try:
                os.killpg(pgid, signal.SIGTERM)
                server["process"].terminate()  # Also send terminate just in case
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server #{server['server_id']} not found.")
            except Exception as e:
                logger.error(f"Error terminating process group {pgid}: {e}")

    # Wait a bit for processes to terminate gracefully
    time.sleep(5)

    # Force kill any remaining processes
    for server in servers:
        if server.get("process") and server["process"].poll() is None:
            pid = server["process"].pid
            pgid = os.getpgid(pid)
            logger.warning(
                f"Process group {pgid} for server #{server['server_id']} (PID {pid}) "
                " did not terminate gracefully. Sending SIGKILL."
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
                server["process"].kill()  # Force kill
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server #{server['server_id']} already gone.")
            except Exception as e:
                logger.error(f"Error killing process group {pgid}: {e}")
        server["process"] = None  # Mark as cleaned up


# Global list to hold server info for signal handler
_server_list_for_signal_handler: List[Dict] = []


def install_signal_handlers(servers: List[Dict]):
    global _server_list_for_signal_handler
    _server_list_for_signal_handler = servers  # Store servers globally for the handler

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received signal {signal_name}. Initiating shutdown...")
        # Access the global list here
        cleanup_servers(_server_list_for_signal_handler)
        logger.warning("Shutdown complete due to signal.")
        sys.exit(1)  # Exit with error code

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination request


async def embed(
    client: AsyncOpenAI,
    texts: List[str],
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> List[List[float]]:
    # Ensure semaphore is acquired before entering the try block
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.embeddings.create(model=model, input=texts)
                return [d.embedding for d in response.data]
            except RateLimitError:
                wait_time = 2**attempt
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{max_retries}. Retrying after {wait_time}s...")
                await asyncio.sleep(wait_time)
            except APIError as e:
                wait_time = 2**attempt
                logger.error(
                    f"API error during embedding (attempt {attempt + 1}/{max_retries}): {e}."
                    f"Retrying after {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Catch unexpected errors
                wait_time = 2**attempt
                logger.error(
                    f"Unexpected error during embedding (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True
                )
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for embedding batch. Skipping batch.")
                    return []
                await asyncio.sleep(wait_time)

    # If all retries fail, due to consistent API errors), return empty list.
    logger.error("Embedding failed after all retries.")
    return []


def _chunk(data: dict, chunk_size: int) -> List[dict]:
    # A wrapper for chunking
    try:
        chunker = TextSplitter(chunk_size)
        doc = Document(**data)
        chunks = chunk_document(doc, chunker)
        return [asdict(chunk) for chunk in chunks]
    except Exception as e:
        logger.error(f"Error chunking document: {e}", exc_info=True)
        return []


async def process_files_worker(
    worker_id: int,
    server: Dict,
    files: List[str],
    config: BuildConfig,
    chunker: TextSplitter,
    claim2queue: Dict[str, AsyncQueue[Optional[Tuple[List[Chunk], str]]]],
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event,
) -> int:
    """Worker coroutine processes files, embeds chunks, sends to the correct CSV queue via map."""
    client = AsyncOpenAI(
        base_url=server["api_url"],
        api_key=config.embedding_model.api_key,
        max_retries=config.embedding_model.max_retries,
        timeout=60.0,
    )
    logger.info(
        f"Worker {worker_id} started, assigned to server #{server['server_id']}"
        f"({server['api_url']}) for {len(files)} files."
    )

    total_processed = 0
    prefix = config.embedding_prefix or ""

    for i, fpath_str in enumerate(files):
        if stop_event.is_set():
            logger.warning(f"Worker {worker_id} received stop signal, terminating early.")
            break

        fpath = Path(fpath_str)
        if not fpath.is_file():
            logger.warning(f"Worker {worker_id}: File not found {fpath_str}, skipping.")
            continue

        try:
            claim_id = fpath.stem
            csv_filename_stem = f"claim_{claim_id}"
            # Get the target queue from the map
            target_queue = claim2queue.get(csv_filename_stem)
            if target_queue is None:
                # This should not happen if the map is built correctly
                logger.error(
                    f"Worker {worker_id}: No queue found for stem '{csv_filename_stem}'. Skipping file {fpath.name}."
                )
                continue  # Skip this file
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to derive CSV filename stem for {fpath.name}: {e}", exc_info=True)
            stop_event.set()  # Signal failure if filename cannot be derived
            break  # Stop processing files for this worker

        batch_chunks: List[Chunk] = []
        batch_texts: List[str] = []
        csv_batch: List[Chunk] = []
        docs_in_file = 0

        try:
            async with async_open(fpath, "r", encoding="utf-8") as f:
                logger.info(f"Worker {worker_id}: Processing file {fpath.name} ({i + 1}/{len(files)}) ")
                async for line in f:
                    if stop_event.is_set():
                        break
                    if not line.strip():
                        # Skip if no content
                        continue

                    try:
                        # Load document
                        doc_data = json.loads(line)
                        doc = Document(**doc_data)
                        docs_in_file += 1

                        # Chunk
                        doc_chunks: List[Chunk] = chunk_document(doc, chunker)
                        logger.debug(f"Worker {worker_id}: Chunking document {claim_id}")
                        if not doc_chunks:
                            # Skip if no chunks
                            continue

                        for chunk in doc_chunks:
                            if stop_event.is_set():
                                break

                            # Batch chunks
                            batch_chunks.append(chunk)
                            batch_texts.append(f"{prefix}{chunk.content}")

                            if len(batch_texts) >= config.batch_size:
                                if stop_event.is_set():
                                    break
                                # Embed
                                embeddings = await embed(
                                    client,
                                    batch_texts,
                                    config.embedding_model.model_id,
                                    semaphore,
                                    config.embedding_model.max_retries,
                                )
                                # Validate embeddings before adding to csv_batch
                                if embeddings and len(embeddings) == len(batch_chunks):
                                    valid_chunks_in_batch = 0
                                    for chunk_to_update, emb in zip(batch_chunks, embeddings):
                                        # Ensure embedding is not empty/None (basic check)
                                        if emb:
                                            chunk_to_update.embedding = emb
                                            csv_batch.append(chunk_to_update)
                                            valid_chunks_in_batch += 1
                                        else:
                                            logger.warning(
                                                f"Worker {worker_id}: Received empty embedding "
                                                f"for a chunk in {fpath.name}. Skipping chunk."
                                            )
                                    if valid_chunks_in_batch > 0:
                                        total_processed += valid_chunks_in_batch
                                    if valid_chunks_in_batch != len(batch_chunks):
                                        logger.warning(
                                            f"Worker {worker_id}: Done ONLY {valid_chunks_in_batch}/{len(batch_chunks)}"
                                            f" embeddings successfully for a batch in {fpath.name}."
                                        )
                                else:
                                    # Skip
                                    logger.error(
                                        f"Worker {worker_id}: Embedding failed or returned incorrect count for "
                                        f"a batch from {fpath.name} expected {len(batch_chunks)} embeddings, "
                                        f"got {len(embeddings) if embeddings else 'None'}. Skipping batch."
                                    )

                                # Queue csv_batch if it reaches size
                                if len(csv_batch) >= config.batch_size and not stop_event.is_set():
                                    # Put into the specific queue for this file stem
                                    await target_queue.put((list(csv_batch), csv_filename_stem))
                                    logger.debug(
                                        f"Worker {worker_id}: Queued {len(csv_batch)} chunks for {csv_filename_stem}.csv"
                                        f" (Qsize: {target_queue.qsize()})"
                                    )
                                    csv_batch.clear()

                                # Always clear text/chunk batches after processing
                                batch_chunks.clear()
                                batch_texts.clear()

                        if stop_event.is_set():
                            break  # Check after inner loop

                    except json.JSONDecodeError:
                        logger.warning(f"Worker {worker_id}: Invalid JSON in a doc in {fpath.name}, skipping line.")
                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id}: Error processing a doc in {fpath.name}: {e}",
                            exc_info=True,
                        )
                if stop_event.is_set():
                    break  # Check after file loop

            # Process remaining embedding batch
            if batch_texts and not stop_event.is_set():
                embeddings = await embed(
                    client, batch_texts, config.embedding_model.model_id, semaphore, config.embedding_model.max_retries
                )
                # Validate final batch embeddings
                if embeddings and len(embeddings) == len(batch_chunks):
                    valid_chunks_in_batch = 0
                    for chunk_to_update, emb in zip(batch_chunks, embeddings):
                        if emb:
                            chunk_to_update.embedding = emb
                            csv_batch.append(chunk_to_update)
                            valid_chunks_in_batch += 1
                        else:
                            logger.warning(
                                f"Worker {worker_id}: Received empty embedding for a chunk in "
                                f"final batch of {fpath.name}. Skipping chunk."
                            )
                    if valid_chunks_in_batch > 0:
                        total_processed += valid_chunks_in_batch
                    if valid_chunks_in_batch != len(batch_chunks):
                        logger.warning(
                            f"Worker {worker_id}: Processed {valid_chunks_in_batch}/{len(batch_chunks)} "
                            f"embeddings successfully for final batch in {fpath.name}."
                        )
                else:
                    logger.error(
                        f"Worker {worker_id}: Embedding failed or returned incorrect count for "
                        f"the final batch of {fpath.name}"
                        f" expected {len(batch_chunks)} embeddings, got {len(embeddings) if embeddings else 'None'}. "
                        "Skipping batch."
                    )

                # Queue any remaining chunks in the csv_batch
                if csv_batch and not stop_event.is_set():
                    # Put into the specific queue for this file stem
                    await target_queue.put((list(csv_batch), csv_filename_stem))
                    logger.debug(
                        f"Worker {worker_id}: Queued final {len(csv_batch)} chunks for {csv_filename_stem}.csv "
                        f" (Qsize: {target_queue.qsize()})"
                    )
                    csv_batch.clear()

                # Clear text/chunk batches after processing
                batch_chunks.clear()
                batch_texts.clear()

                if not stop_event.is_set():  # Only update progress if not stopped
                    logger.debug(
                        f"Worker {worker_id}: Finished processing {fpath.name} ({i + 1}/{len(files)}), "
                        f"{docs_in_file} documents."
                    )

        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed during processing file {fpath_str}: {e}", exc_info=True)
            stop_event.set()  # Signal failure on file processing error
        finally:
            # Clear batches if error occurs or stopped
            batch_chunks.clear()
            batch_texts.clear()
            csv_batch.clear()  # Clear csv_batch

        if stop_event.is_set():
            logger.warning(f"Worker {worker_id} stopping after processing {fpath.name} due to stop signal.")
            break

    logger.info(f"Worker {worker_id} finished. Processed {total_processed} chunks (embeddings generated).")
    await client.close()
    return total_processed


async def build(config: BuildConfig):
    logger.info("Starting multi-GPU build process (CSV output only)...")
    logger.info(OmegaConf.to_yaml(config))

    servers = []
    csv_writer_tasks = []
    claim2queue: Dict[str, AsyncQueue[Optional[Tuple[List[Chunk], str]]]] = {}
    total_chunks_generated = 0
    stop_processing_event = asyncio.Event()
    start_time = timer()
    results_dir = Path(config.results_file_dir)

    try:
        # 0. Ensure results directory exists before starting anything else
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured results directory exists: {results_dir}")
        except OSError as e:
            logger.critical(f"Failed to create results directory '{results_dir}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to create results directory: {e}") from e

        # 1. Start vLLM Servers
        servers = start_servers(config)
        if servers:
            install_signal_handlers(servers)  # Install signal handlers early
        else:
            logger.error("No vLLM servers could be initialized. Aborting.")
            return

        # 2. Wait for Servers to be Ready
        ready_servers = wait_for_servers(servers)
        if not ready_servers:
            logger.error("Aborting build: Not all required servers became healthy.")
            raise RuntimeError("Server startup failed")

        # 3. Prepare File List, Chunker, and Determine Unique Stems
        chunker = TextSplitter(config.chunk_size)
        evidence_dir = Path(config.evidence_dir)
        if not evidence_dir.is_dir():
            raise FileNotFoundError(f"Evidence directory not found: {config.evidence_dir}")

        evidence_files = [str(p) for p in evidence_dir.glob("**/*.json")]
        if not evidence_files:
            raise FileNotFoundError(f"No .json files found in {config.evidence_dir}")

        if config.limit:
            evidence_files = evidence_files[: config.limit]
        logger.info(f"Found {len(evidence_files)} evidence files to process.")

        # Determine unique output filename stems
        unique_stems: Set[str] = set()
        for fpath_str in evidence_files:
            try:
                stem = Path(fpath_str).stem
                unique_stems.add(f"claim_{stem}")
            except Exception as e:
                logger.error(f"Failed to get stem for file {fpath_str}: {e}. Skipping.")
        logger.info(f"Identified {len(unique_stems)} unique output CSV files.")
        if not unique_stems:
            logger.warning("No unique output file stems identified. No writers will be started.")

        # 4. Initialize Queues and CSV Writer Tasks per Stem
        logger.info(f"Initializing {len(unique_stems)} CSV writer queues and tasks (one per unique file)...")
        q_max_size = config.max_queue_size  # Apply max_queue_size to each queue if > 0
        for stem in unique_stems:
            queue = AsyncQueue(q_max_size)
            claim2queue[stem] = queue
            writer_task = asyncio.create_task(
                csv_writer_task(
                    writer_id=stem,  # Use stem as the writer ID
                    queue=queue,
                    stop_event=stop_processing_event,
                    results_dir=config.results_file_dir,
                    csv_batch_size=config.csv_batch_size,
                ),
                name=f"CSVWriter-{stem}",
            )
            csv_writer_tasks.append(writer_task)

        if not csv_writer_tasks:
            logger.warning("No CSV writer tasks were created (no unique stems found).")
        else:
            logger.info(f"{len(csv_writer_tasks)} CSV writer tasks started.")

        # 5. Distribute Files and Create Worker Tasks
        worker_tasks = []
        num_workers = len(ready_servers)
        logger.info(f"Distributing {len(evidence_files)} files across {num_workers} workers...")
        for i, server in enumerate(ready_servers):
            worker_files = evidence_files[i::num_workers]
            if worker_files:
                server_semaphore = asyncio.Semaphore(config.rate_limit)
                worker_tasks.append(
                    asyncio.create_task(
                        process_files_worker(
                            worker_id=i,
                            server=server,
                            files=worker_files,
                            config=config,
                            chunker=chunker,
                            claim2queue=claim2queue,
                            semaphore=server_semaphore,
                            stop_event=stop_processing_event,
                        ),
                        name=f"Worker-{i}",
                    )
                )

        # 6. Run Worker Tasks and Monitor
        if worker_tasks:
            # Wait for all file processing workers to complete
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Process results from workers
            valid_results = [r for r in results if isinstance(r, int)]
            total_chunks_generated = sum(valid_results)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Worker {i} failed: {result}", exc_info=result)
                    stop_processing_event.set()  # Signal stop if any worker fails critically
                elif isinstance(result, asyncio.CancelledError):
                    logger.warning(f"Worker {i} was cancelled.")

            logger.info(f"File processing workers finished. Generated {total_chunks_generated} chunks (embeddings).")

            # Check if we need to stop early due to worker failure
            if stop_processing_event.is_set():
                logger.error("Stopping build early due to worker failure during embedding/processing.")
                raise RuntimeError("Worker task failed during execution.")

        else:  # No worker tasks created
            logger.info("No worker tasks were created (likely no files found).")
            total_chunks_generated = 0  # Ensure it's 0

        # 7. Signal CSV Writers to Finish and Wait
        logger.info(f"Signaling {len(csv_writer_tasks)} CSV writers to finish processing queued items...")
        all_queues = list(claim2queue.values())
        for i, q in enumerate(all_queues):
            await q.put(None)  # Send sentinel to each queue

        logger.info("Waiting for CSV queues to be fully processed...")
        # Wait for all queues to be joined
        await asyncio.gather(*(q.join() for q in all_queues))
        logger.info("All CSV queues processed.")

        # Wait for the CSV writer tasks themselves to finish
        logger.info("Waiting for CSV writer tasks to complete...")
        writer_results = await asyncio.gather(*csv_writer_tasks, return_exceptions=True)
        csv_writer_tasks.clear()  # Clear task references

        # Check results from writers
        all_writers_succeeded = True
        for i, result in enumerate(writer_results):
            if isinstance(result, Exception):
                logger.error(f"CSV Writer {i} failed: {result}", exc_info=result)
                all_writers_succeeded = False
                stop_processing_event.set()  # Signal stop if any writer fails critically
            elif result is False:  # Writer returned False indicating failure
                logger.error(f"CSV Writer {i} reported failure.")
                all_writers_succeeded = False
                stop_processing_event.set()
            elif isinstance(result, asyncio.CancelledError):
                logger.warning(f"CSV Writer {i} was cancelled.")
                # Don't necessarily mark as failure unless explicitly needed

        if not all_writers_succeeded or stop_processing_event.is_set():
            logger.error("One or more CSV Writers encountered a critical failure or stop signal received.")
            raise RuntimeError("CSV writing failed. Build incomplete.")

        logger.info(f"CSV writing completed successfully. Total chunks written to CSV: {total_chunks_generated}")

        logger.info(f"Build completed successfully in {timer() - start_time:.2f} seconds.")
        logger.info(f"Total chunks processed and written to CSV files: {total_chunks_generated}")

    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Build failed: {e}")
        stop_processing_event.set()
    except asyncio.CancelledError:
        logger.warning("Build task was cancelled.")
        stop_processing_event.set()
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the build: {e}", exc_info=True)
        stop_processing_event.set()
    finally:
        # 8. Cleanup Resources
        logger.info("Starting final cleanup...")
        stop_processing_event.set()  # Ensure all tasks know to stop

        # Cancel any remaining CSV writers if they are still running
        if csv_writer_tasks:  # Check if list wasn't cleared
            logger.warning(f"Attempting to cancel {len(csv_writer_tasks)} potentially running CSV writer tasks...")
            for task in csv_writer_tasks:
                if not task.done():
                    task.cancel()
            try:
                # Wait briefly for cancellations
                await asyncio.gather(*csv_writer_tasks, return_exceptions=True)
            except Exception as e_csvw_cancel:
                logger.error(f"Error during CSV writer cancellation cleanup: {e_csvw_cancel}")

        # Cleanup Servers
        if servers:
            logger.info("Cleaning up embedding servers...")
            cleanup_servers(servers)

        logger.info("Build process finished.")


@hydra.main(version_base=None, config_path="../conf", config_name="build")
def main(config: BuildConfig):
    try:
        asyncio.run(build(config))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received in main. Initiating shutdown...")
    except RuntimeError as e:
        logger.error(f"Build runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
