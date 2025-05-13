import asyncio
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional
from urllib.parse import urlparse

import hydra
import requests
from hydra.core.config_store import ConfigStore
from openai import APIError, AsyncOpenAI, RateLimitError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

# Setup logging
logger = logging.getLogger("veritas")
console = Console()
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global list to hold server info for signal handler
_server_list_for_signal_handler: List[Dict] = []


@dataclass
class ModelConfig:
    model_id: str
    base_url: str
    api_key: str = "EMPTY"
    max_retries: int = 3
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 8192
    emb_dim: int = 1024
    emb_prefix: Optional[str] = None


@dataclass
class EmbedConfig:
    chunks_dir: str
    results_dir: str
    embedding_model: ModelConfig
    batch_size: int = 1024
    rate_limit: int = 8192
    n_gpus: int = 1
    n_processes_per_gpu: int = 1
    base_port: int = 1775
    gpu_ids: Optional[List[int]] = None
    server_startup_delay: int = 5
    show_vllm_logs: bool = False


# Register config with hydra
cs = ConfigStore.instance()
cs.store(name="config", node=EmbedConfig)


def start_servers(config: EmbedConfig) -> List[Dict]:
    """Start vLLM servers"""
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

            # Command to start vLLM server
            # fmt: off
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", config.embedding_model.model_id, "--host", hostname, "--port", str(port), 
                "--api-key", config.embedding_model.api_key,
                "--trust-remote-code", "--max-num-seqs", str(config.embedding_model.max_num_seqs),
                "--gpu-memory-utilization", str(config.embedding_model.gpu_memory_utilization),
                "--max-model-len", str(config.embedding_model.max_model_len),
                "--disable-log-requests",
            ]
            # fmt: on

            logger.info(f"Starting vLLM server #{server_id_counter + 1} on GPU {gpu_id} at {api_url}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if not config.show_vllm_logs:
                env["VLLM_CONFIGURE_LOGGING"] = str("0")

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
    """Wait for servers to become healthy"""
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
                # Check if the process terminated unexpectedly
                if server["process"].poll() is not None:
                    logger.error(
                        f"Server #{server['server_id']} (Port {server['port']}) terminated prematurely"
                        f" with code {server['process'].returncode}."
                    )
                    pending_servers.remove(server)
                    continue

                # Check health endpoint
                try:
                    response = requests.get(server["health_url"], timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✓ Server #{server['server_id']} (Port {server['port']}) is healthy.")
                        ready_servers.append(server)
                        pending_servers.remove(server)
                except (requests.RequestException, ConnectionError) as e:
                    logger.debug(f"Server #{server['server_id']} (Port {server['port']}) not ready yet: {e}")
                    pass

            if pending_servers:
                time.sleep(check_interval)

    # Final check for any remaining pending servers
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
    """Clean up all server processes"""
    logger.info(f"Shutting down {len(servers)} vLLM server processes...")
    for server in servers:
        if server.get("process") and server["process"].poll() is None:
            pid = server["process"].pid
            pgid = os.getpgid(pid)
            logger.info(f"Sending SIGTERM to process group {pgid} for server #{server['server_id']} (PID {pid})")
            try:
                os.killpg(pgid, signal.SIGTERM)
                server["process"].terminate()
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server #{server['server_id']} not found.")
            except Exception as e:
                logger.error(f"Error terminating process group {pgid}: {e}")

    # Wait for processes to terminate gracefully
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
                server["process"].kill()
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server #{server['server_id']} already gone.")
            except Exception as e:
                logger.error(f"Error killing process group {pgid}: {e}")
        server["process"] = None  # Mark as cleaned up


def install_signal_handlers(servers: List[Dict]):
    """Install signal handlers for graceful shutdown"""
    global _server_list_for_signal_handler
    _server_list_for_signal_handler = servers  # Store servers globally for the handler

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received signal {signal_name}. Initiating shutdown...")
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
    """Embed a batch of texts with retries"""
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
                wait_time = 2**attempt
                logger.error(
                    f"Unexpected error during embedding (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True
                )
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for embedding batch. Skipping batch.")
                    return []
                await asyncio.sleep(wait_time)

    # If all retries fail, return empty list.
    logger.error("Embedding failed after all retries.")
    return []


async def embedding_worker(
    worker_id: int,
    server: Dict,
    file_paths: List[str],
    config: EmbedConfig,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event,
    progress: Progress,
    file_task_id: TaskID,
) -> Dict[str, int]:
    """Process and embed chunks from JSONL files"""
    client = AsyncOpenAI(
        base_url=server["api_url"],
        api_key=config.embedding_model.api_key,
        max_retries=config.embedding_model.max_retries,
        timeout=60.0,
    )
    logger.info(
        f"Worker {worker_id} started, assigned to server #{server['server_id']} "
        f"({server['api_url']}) for {len(file_paths)} files."
    )

    results = {}  # Store success count per file
    prefix = config.embedding_model.emb_prefix or ""
    results_dir = Path(config.results_dir)

    # Get this worker's task ID for updating progress
    total_chunks_processed = 0

    for i, file_path_str in enumerate(file_paths):
        if stop_event.is_set():
            logger.warning(f"Worker {worker_id} received stop signal, terminating early.")
            break

        file_path = Path(file_path_str)
        if not file_path.exists():
            logger.warning(f"Worker {worker_id}: File not found {file_path_str}, skipping.")
            continue

        # Extract claim ID from filename
        claim_id = file_path.stem
        output_path = results_dir / f"{claim_id}.pkl"

        # Skip if already processed
        if output_path.exists():
            logger.info(f"Worker {worker_id}: File {claim_id}.pkl already exists, skipping.")
            results[claim_id] = 0  # Track as skipped
            # Update overall file progress
            progress.update(file_task_id, advance=1)
            continue

        try:
            chunks_data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        try:
                            chunk = json.loads(line)
                            chunks_data.append(chunk)
                        except json.JSONDecodeError:
                            logger.warning(f"Worker {worker_id}: Invalid JSON in file {file_path_str}, skipping line.")

            if not chunks_data:
                logger.warning(f"Worker {worker_id}: No valid chunks in file {file_path_str}, skipping.")
                progress.update(file_task_id, advance=1)
                continue

            logger.info(
                f"Worker {worker_id}: Processing file {claim_id}.json ({i + 1}/{len(file_paths)}) with {len(chunks_data)} chunks"
            )

            # Process chunks in batches using slicing
            all_embeddings = []

            # Process chunks in batches using slices
            for start_idx in range(0, len(chunks_data), config.batch_size):
                if stop_event.is_set():
                    break

                # Get the current batch of chunks
                end_idx = min(start_idx + config.batch_size, len(chunks_data))
                batch_chunks = chunks_data[start_idx:end_idx]
                batch_size = len(batch_chunks)

                # Prepare texts for embedding
                batch_texts = [f"{prefix}{chunk['content']}" for chunk in batch_chunks]

                # Get embeddings
                embeddings = await embed(
                    client, batch_texts, config.embedding_model.model_id, semaphore, config.embedding_model.max_retries
                )

                if len(embeddings) != len(batch_texts):
                    logger.error(
                        f"Worker {worker_id}: Expected {len(batch_texts)} embeddings but got {len(embeddings)}. "
                        f"Filling missing with zeros."
                    )
                    # Fill missing embeddings with zeros
                    while len(embeddings) < len(batch_texts):
                        embeddings.append([0.0] * config.embedding_model.emb_dim)

                # Order is maintained because we're processing in sequence
                all_embeddings.extend(embeddings)
                total_chunks_processed += batch_size

            # Verify all chunks have embeddings
            if len(all_embeddings) != len(chunks_data):
                logger.error(
                    f"Worker {worker_id}: Mismatch in embeddings count for file {claim_id}. "
                    f"Expected {len(chunks_data)}, got {len(all_embeddings)}."
                )

            # Save embeddings immediately for this file
            with open(output_path, "wb") as f:
                pickle.dump(all_embeddings, f)

            # Store success in results
            results[claim_id] = len(all_embeddings)
            logger.info(
                f"Worker {worker_id}: Finished and saved file {claim_id}.pkl with {len(all_embeddings)} embeddings"
            )

            # Update overall file progress
            progress.update(file_task_id, advance=1)

        except Exception as e:
            logger.error(f"Worker {worker_id}: Error processing {file_path_str}: {e}", exc_info=True)
            results[claim_id] = -1  # Mark as error
            # Update progress even on error
            progress.update(file_task_id, advance=1)
            if not stop_event.is_set():
                # Don't stop everything for a single file error
                continue

    await client.close()
    return results  # Return summary of processed files


async def run_embedding(config: EmbedConfig):
    """Main function to run the embedding process"""
    logger.info("Starting parallel embedding process...")

    # Ensure results directory exists
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    servers = []
    stop_event = asyncio.Event()

    try:
        # Start vLLM servers
        servers = start_servers(config)
        if servers:
            install_signal_handlers(servers)
        else:
            logger.error("No vLLM servers could be initialized. Aborting.")
            return

        # Wait for servers to be ready
        ready_servers = wait_for_servers(servers)
        if not ready_servers:
            logger.error("Aborting: Not all required servers became healthy.")
            raise RuntimeError("Server startup failed")

        # Get all chunk files
        chunks_dir = Path(config.chunks_dir)
        if not chunks_dir.is_dir():
            raise FileNotFoundError(f"Chunks directory not found: {config.chunks_dir}")

        chunked_evidence_fpaths = sorted(chunks_dir.glob("*.json"), key=lambda p: int(p.stem))
        if not chunked_evidence_fpaths:
            raise FileNotFoundError(f"No JSON files found in {config.chunks_dir}")

        logger.info(f"Found {len(chunked_evidence_fpaths)} chunk files to process.")

        # Distribute files among workers
        num_workers = len(ready_servers)
        worker_tasks = []

        # Create progress tracking objects
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            # Create a main task for overall file progress
            file_task_id = progress.add_task("[green]Files Processed", total=len(chunked_evidence_fpaths))

            logger.info(f"Distributing files across {num_workers} workers...")
            for i, server in enumerate(ready_servers):
                # Distribute files to workers (each worker gets every nth file)
                worker_files = [str(fpath) for j, fpath in enumerate(chunked_evidence_fpaths) if j % num_workers == i]

                if worker_files:
                    server_semaphore = asyncio.Semaphore(config.rate_limit)
                    worker_tasks.append(
                        asyncio.create_task(
                            embedding_worker(
                                worker_id=i,
                                server=server,
                                file_paths=worker_files,
                                config=config,
                                semaphore=server_semaphore,
                                stop_event=stop_event,
                                progress=progress,
                                file_task_id=file_task_id,
                            ),
                            name=f"Worker-{i}",
                        )
                    )

            # Wait for all workers to complete
            if worker_tasks:
                worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

                # Process results to count successes and failures
                successful_files = 0
                failed_files = 0
                skipped_files = 0

                for result in worker_results:
                    if isinstance(result, dict):
                        # Count successful, failed and skipped files
                        for claim_id, count in result.items():
                            if count > 0:
                                successful_files += 1
                            elif count == 0:
                                skipped_files += 1
                            else:
                                failed_files += 1
                    elif isinstance(result, Exception):
                        logger.error(f"Worker failed: {result}", exc_info=result)

                logger.info("Embedding process summary:")
                logger.info(f"  - Successfully embedded and saved: {successful_files} files")
                logger.info(f"  - Skipped (already existing): {skipped_files} files")
                logger.info(f"  - Failed to process: {failed_files} files")

                # Check whether all files were successful
                total_expected = len(chunked_evidence_fpaths)
                if successful_files + skipped_files + failed_files != total_expected:
                    logger.warning(
                        f"Embedding process completed with discrepancy: "
                        f"Expected {total_expected} files, but accounted for {successful_files + skipped_files + failed_files}"
                    )

                if failed_files > 0:
                    logger.warning(f"Some files ({failed_files}) failed to process. Check logs for details.")
                else:
                    logger.info("All files were processed successfully or skipped (already existed).")
            else:
                logger.warning("No worker tasks were created!")

    except Exception as e:
        logger.error(f"Embedding process failed: {e}", exc_info=True)
        stop_event.set()
    finally:
        # Clean up
        if servers:
            logger.info("Cleaning up embedding servers...")
            cleanup_servers(servers)

        logger.info("Embedding process finished.")


@hydra.main(version_base=None, config_path="../conf", config_name="embed")
def main(config: EmbedConfig):
    try:
        asyncio.run(run_embedding(config))
    except KeyboardInterrupt:
        logger.warning("Embedding process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
