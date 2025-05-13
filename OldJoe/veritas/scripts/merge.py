import csv
import json
import logging
import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger("veritas")


@dataclass
class MergeConfig:
    chunks_dir: str
    embeddings_dir: str
    output_dir: str
    embedding_dim: int = 1024
    limit: Optional[int] = None
    num_workers: int = 4


cs = ConfigStore.instance()
cs.store(name="config", node=MergeConfig)


def pair(chunks_dir_path: Path, embeddings_dir_path: Path, limit: Optional[int]) -> List[Tuple[Path, Path]]:
    """Finds chunk JSONL files and their corresponding embedding PKL files."""
    if not chunks_dir_path.is_dir():
        logger.error(f"Chunks directory not found: {chunks_dir_path}")
        sys.exit(1)
    if not embeddings_dir_path.is_dir():
        logger.error(f"Embeddings directory not found: {embeddings_dir_path}")
        sys.exit(1)

    chunk_files = sorted(list(chunks_dir_path.glob("*.json")))
    if not chunk_files:
        logger.warning(f"No .json chunk files found in {chunks_dir_path}")
        return []

    if limit:
        chunk_files = chunk_files[:limit]
        logger.info(f"Processing limit applied: {limit} files.")

    logger.info(f"Found {len(chunk_files)} potential chunk files.")

    valid_pairs = []
    for chunk_file_path in chunk_files:
        embedding_file_path = embeddings_dir_path / f"{chunk_file_path.stem}.pkl"
        if not embedding_file_path.exists():
            logger.warning(f"Embedding file not found for {chunk_file_path.name}, skipping pair: {embedding_file_path}")
        else:
            valid_pairs.append((chunk_file_path, embedding_file_path))

    logger.info(f"Found {len(valid_pairs)} valid chunk/embedding file pairs.")
    return valid_pairs


def parse_and_merge(chunk_fpath: Path, embedding_fpath: Path, embedding_dim: int) -> Generator[Dict, None, None]:
    """Reads chunk and embedding files, merges, validates, and yields dictionaries."""
    rows = 0
    try:
        # Load embeddings from PKL file
        with open(embedding_fpath, "rb") as f_emb:
            embeddings = pickle.load(f_emb)

        if not isinstance(embeddings, list):
            logger.warning(f"Embedding file {embedding_fpath.name} does not contain a list. Skipping.")
            return
        logger.debug(f"Loaded {len(embeddings)} embeddings from {embedding_fpath.name}")

        # Read chunks from JSONL file
        chunks_data = []
        with open(chunk_fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunks_data.append(data)
                    rows += 1
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {chunk_fpath.name}. Skipping line: {line.strip()}")
                except Exception as e:
                    logger.error(f"Unexpected error processing a line in {chunk_fpath.name}: {e}", exc_info=True)

        # Validate counts
        if len(chunks_data) != len(embeddings):
            logger.warning(
                f"Mismatch between chunks ({len(chunks_data)}) and embeddings ({len(embeddings)}) "
                f"for {chunk_fpath.name}. Skipping file."
            )
            return

        # Combine chunks and embeddings, yield valid ones as dicts
        for chunk_dict, embedding in zip(chunks_data, embeddings):
            if not isinstance(embedding, (list, tuple)) or len(embedding) != embedding_dim:
                logger.warning(
                    f"Invalid or dimension mismatch embedding for chunk {chunk_dict.get('doc_id', 'N/A')}/{chunk_dict.get('chunk_index', 'N/A')} "
                    f"in {chunk_fpath.name}. Expected {embedding_dim}, "
                    f"got {len(embedding) if isinstance(embedding, (list, tuple)) else type(embedding)}. Skipping chunk."
                )
                continue

            # Add embedding (serialized as JSON string for CSV compatibility)
            chunk_dict["embedding"] = json.dumps(embedding)
            yield chunk_dict

        logger.debug(f"Finished parsing {chunk_fpath.name} & {embedding_fpath.name} ({rows} chunks yielded).")
    except pickle.UnpicklingError:
        logger.error(f"Failed to unpickle embedding file {embedding_fpath.name}. Skipping.")
    except FileNotFoundError:
        logger.error(f"File disappeared during processing: {chunk_fpath} or {embedding_fpath}")
    except Exception as e:
        logger.error(f"Failed to process {chunk_fpath.name}/{embedding_fpath.name}: {e}", exc_info=True)


def merge_worker(
    worker_id: int,
    assigned_files: List[Tuple[Path, Path]],
    output_dir: Path,
    config: MergeConfig,
) -> int:
    """Worker process to process a subset of file pairs and write to CSV."""
    logger.info(f"[Worker {worker_id}] Starting with {len(assigned_files)} files.")
    fieldnames = ["doc_id", "source_url", "chunk_index", "content", "embedding"]
    files_processed_count = 0

    for i, (chunk_file_path, embedding_file_path) in enumerate(assigned_files):
        output_csv_path = output_dir / f"{chunk_file_path.stem}.csv"
        logger.info(f"[Worker {worker_id}] Processing file {i + 1}/{len(assigned_files)} -> {output_csv_path.name}")

        try:
            rows_written = 0
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()

                for data_dict in parse_and_merge(chunk_file_path, embedding_file_path, config.embedding_dim):
                    writer.writerow(data_dict)
                    rows_written += 1

            logger.info(
                f"[Worker {worker_id}] Finished merging to '{output_csv_path.name}'. Wrote {rows_written} rows."
            )
            files_processed_count += 1

        except Exception as e:
            logger.error(
                f"[Worker {worker_id}] Error processing {chunk_file_path.name}/{embedding_file_path.name} to {output_csv_path.name}: {e}",
                exc_info=True,
            )

    logger.info(f"[Worker {worker_id}] Done processing {files_processed_count}/{len(assigned_files)} assigned files.")
    # Return count of successfully processed files in this batch
    return files_processed_count


def merge_runner(config: MergeConfig):
    """Orchestrates the merging process using multiprocessing."""
    logger.info("Starting chunk and embedding merge process...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    chunks_dir = Path(config.chunks_dir)
    embeddings_dir = Path(config.embeddings_dir)
    output_dir = Path(config.output_dir)

    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        sys.exit(1)

    # Get valid file pairs
    file_pairs = pair(chunks_dir, embeddings_dir, config.limit)
    if not file_pairs:
        logger.warning("No valid file pairs found to process.")
        return
    total_files_to_process = len(file_pairs)

    # Decide on number of worker processes
    num_workers = config.num_workers
    if num_workers <= 0:
        logger.warning("num_workers must be > 0. Setting workers to 1.")
        num_workers = 1
    # Don't need more workers than files
    if num_workers > total_files_to_process:
        num_workers = total_files_to_process
        logger.info(f"Adjusted number of workers to {num_workers} (number of files).")
    logger.info(f"Using {num_workers} worker processes for parallel processing.")

    # Split file pairs among workers - Each worker gets a list of files
    # We will submit tasks per file pair for better progress tracking with imap_unordered
    # Instead of splitting into large chunks, prepare arguments for each file pair
    worker_args = []
    for i, file_pair in enumerate(file_pairs):
        # Assign a dummy worker_id for logging, real parallelism is managed by pool
        worker_args.append((i % num_workers, [file_pair], output_dir, config))

    # Define progress bar columns
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} files)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    processed_files_count = 0
    try:
        with Progress(*progress_columns) as progress:
            merge_task = progress.add_task("[cyan]Merging files...", total=total_files_to_process)

            # Use starmap with single-file assignments for progress update per file
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use imap_unordered to get results as they complete
                results_iterator = pool.starmap(
                    merge_worker,
                    [(i, [pair], output_dir, config) for i, pair in enumerate(file_pairs)],  # Pass single pair list
                )

                # Process results as they arrive and update progress
                for result in results_iterator:
                    # result should be 1 if the single file was processed, 0 otherwise
                    processed_files_count += result
                    progress.update(merge_task, advance=1)  # Advance by 1 file

            # Final update for progress bar description
            progress.update(merge_task, description="[green]File merging complete.")

    except Exception as e:
        logger.critical(f"An error occurred during the merge process: {e}", exc_info=True)
    finally:
        logger.info(
            f"Merge process finished. Processed {processed_files_count}/{total_files_to_process} files. Output CSVs are in {output_dir}"
        )


@hydra.main(version_base=None, config_path="../conf", config_name="merge")
def main(cfg: MergeConfig):
    try:
        merge_runner(cfg)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Ensure spawn method is used if necessary, especially on Windows/macOS
    # multiprocessing.set_start_method('spawn', force=True)
    main()
