import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from semantic_text_splitter import TextSplitter

from veritas.kb.chunk import Document, chunk_document

logger = logging.getLogger("veritas")
console = Console()


@dataclass
class ChunkConfig:
    evidence_dir: str
    results_dir: str
    chunk_size: int = 2048
    batch_size: int = 1000
    limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="config", node=ChunkConfig)


def get_total_documents(files):
    """Count the total number of documents in all provided files."""
    total = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn()
    ) as progress:
        count_task = progress.add_task("[yellow]Counting documents...", total=len(files))
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    # Count non-empty lines
                    file_lines = sum(1 for line in f if line.strip())
                    total += file_lines
                progress.update(count_task, advance=1)
            except Exception as e:
                logger.error(f"Error counting documents in {file}: {e}")
                progress.update(count_task, advance=1)
    return total


def chunk(args: Tuple[str, int, int, str]) -> Tuple[int, int]:
    """Process a single evidence file: read, chunk documents, save chunks to JSON.
    Returns a tuple of (processed_lines, total_chunks)"""
    fpath, chunk_size, batch_size, results_dir = args

    fpath = Path(fpath)
    if not fpath.is_file():
        logger.warning(f"File not found {fpath}, skipping.")
        return 0, 0

    try:
        # Create output file path - maintain same filename
        output_path = Path(results_dir) / fpath.name

        # Prepare chunker
        chunker = TextSplitter(chunk_size)
        total_chunks = 0
        chunk_buffer = []
        processed_lines = 0

        # Process line by line
        with open(output_path, "w", encoding="utf-8") as out_f, open(fpath, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue

                processed_lines += 1

                try:
                    # Parse JSON document
                    doc_data = json.loads(line)
                    doc = Document(**doc_data)

                    # Chunk the document
                    doc_chunks = chunk_document(doc, chunker)

                    # Add chunks to buffer
                    for chunk in doc_chunks:
                        chunk_dict = asdict(chunk)
                        chunk_buffer.append(json.dumps(chunk_dict, ensure_ascii=False))
                        total_chunks += 1

                        # Write batch when buffer reaches batch size
                        if len(chunk_buffer) >= batch_size:
                            # Perform efficient batch write
                            out_f.write("\n".join(chunk_buffer) + "\n")
                            out_f.flush()  # Ensure data is written to disk
                            chunk_buffer = []  # Clear buffer after writing

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in a document in {fpath}, skipping line.")
                except Exception as e:
                    logger.error(f"Error processing a document in {fpath}: {e}", exc_info=True)

            # Write any remaining chunks in the buffer
            if chunk_buffer:
                out_f.write("\n".join(chunk_buffer) + "\n")
                out_f.flush()  # Ensure final batch is written to disk

        logger.debug(f"Processed {fpath.name}: generated {total_chunks} chunks from {processed_lines} documents")
        return processed_lines, total_chunks

    except Exception as e:
        logger.error(f"Error processing document {fpath}: {e}", exc_info=True)
        return 0, 0


@hydra.main(version_base=None, config_path="../conf", config_name="chunk")
def main(config: ChunkConfig):
    logger.info("Starting parallel chunking process...")
    logger.info(OmegaConf.to_yaml(config))

    start_time = timer()

    # Validate and prepare directories
    evidence_dir = Path(config.evidence_dir)
    results_dir = Path(config.results_dir)
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"Evidence directory not found: {config.evidence_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get list of evidence files
    evidence_files = [str(p) for p in evidence_dir.glob("**/*.json")]
    if not evidence_files:
        raise FileNotFoundError(f"No .json files found in {config.evidence_dir}")

    if config.limit:
        evidence_files = evidence_files[: config.limit]

    logger.info(f"Found {len(evidence_files)} evidence files to process")

    # First pass: count total lines across all files
    logger.info("Performing initial pass to count documents...")
    total_docs = get_total_documents(evidence_files)
    logger.info(f"Found {total_docs} documents to process across all files")

    # Process files in parallel with progress bar
    total_chunks = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[chunks]} chunks"),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as progress:
        # Create the task with total_docs as the total
        task = progress.add_task("[green]Processing documents...", total=total_docs, chunks=0)

        # Prepare arguments for parallel processing
        args_list = [(fpath, config.chunk_size, config.batch_size, config.results_dir) for fpath in evidence_files]

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            # Process files and update progress as results come in
            for processed_lines, chunks_in_file in executor.map(chunk, args_list):
                total_chunks += chunks_in_file
                progress.update(task, advance=processed_lines, chunks=total_chunks)

    elapsed = timer() - start_time
    logger.info(f"Chunking completed in {elapsed:.2f} seconds")
    logger.info(f"Total chunks generated: {total_chunks}")


if __name__ == "__main__":
    main()
