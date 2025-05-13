import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from timeit import default_timer as timer
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
import hydra
import instructor
import pandas as pd
import requests
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from openai import APIConnectionError, AsyncOpenAI
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from veritas import agen, judge, qgen, rrank
from veritas.kb import pg

logger = logging.getLogger("veritas")


# Global list to hold server info for signal handler
_server_list_for_signal_handler: List[Dict] = []


@dataclass
class ModelConfig:
    model_id: str
    base_url: str
    api_key: str
    max_retries: int = 3
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: Optional[int] = 512
    gpu_id: int = 0


@dataclass
class DBConfig:
    name: str = "veritas"
    user: str = "dsouzars"
    password: str = "POSTGRES"
    host: str = "localhost"
    port: int = 5432
    connections: int = 4
    embedding_dim: int = 1024


@dataclass
class DevConfig:
    limit: Optional[int] = None
    ids: Optional[List[int]] = None
    bundled: bool = False


@dataclass
class RunConfig:
    claim_fpath: str
    reasoning_model: ModelConfig
    embedding_model: ModelConfig
    reranking_model: Optional[ModelConfig]
    db: DBConfig
    qgen: qgen.QGenConfig
    agen: agen.AGenConfig
    judge: judge.JudgeConfig
    retrieval: pg.RetrievalConfig
    reranking: rrank.RerankingConfig
    dev: DevConfig
    chunk_size: int = 2048
    save_dir: str = "./results"
    save_fname: str = "results.json"
    save_intermediate: bool = True
    save_every: int = 60


cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)


def start_servers(
    reasoner: ModelConfig, embedder: ModelConfig, reranker: Optional[ModelConfig] = None
) -> List[subprocess.Popen]:
    servers = []
    # fmt: off
    reasoner_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", reasoner.model_id, "--port", str(urlparse(reasoner.base_url).port),
        "--gpu-memory-utilization", str(reasoner.gpu_memory_utilization),
        "--max-model-len", str(reasoner.max_model_len),
        "--trust-remote-code", # TODO May be a problem on an airgapped server
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes", # https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm
        "--disable-log-requests", "--disable-log-stats",
    ]
    # fmt: on
    if reasoner.max_num_seqs is not None:
        reasoner_cmd.extend(["--max-num-seqs", str(reasoner.max_num_seqs)])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(reasoner.gpu_id)
    env = os.environ.copy()
    logger.info(f"Starting reasoning server: {' '.join(reasoner_cmd)}")
    reasoner_process = subprocess.Popen(reasoner_cmd, env=env)
    servers.append(reasoner_process)

    # fmt: off
    embedder_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", embedder.model_id, "--port", str(urlparse(embedder.base_url).port),
        "--gpu-memory-utilization", str(embedder.gpu_memory_utilization),
        "--max-model-len", str(embedder.max_model_len),
        "--trust-remote-code", # TODO May be a problem on an airgapped server
        "--disable-log-requests", "--disable-log-stats",
    ]
    # fmt: on
    if embedder.max_num_seqs is not None:
        embedder_cmd.extend(["--max-num-seqs", str(embedder.max_num_seqs)])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(embedder.gpu_id)
    env = os.environ.copy()
    logger.info(f"Starting embedding server: {' '.join(embedder_cmd)}")
    embedder_process = subprocess.Popen(embedder_cmd, env=env)
    servers.append(embedder_process)

    if reranker is not None:
        # fmt: off
        reranker_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", reranker.model_id, "--port", str(urlparse(reranker.base_url).port),
            "--gpu-memory-utilization", str(reranker.gpu_memory_utilization),
            "--max-model-len", str(reranker.max_model_len),
            "--trust-remote-code", # TODO May be a problem on an airgapped server
            "--disable-log-requests", "--disable-log-stats",
        ]
        # fmt: on
        if reranker.max_num_seqs is not None:
            reranker_cmd.extend(["--max-num-seqs", str(reranker.max_num_seqs)])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(reranker.gpu_id)
        env = os.environ.copy()
        logger.info(f"Starting reranking server: {' '.join(reranker_cmd)}")
        reranker_process = subprocess.Popen(reranker_cmd, env=env)
        servers.append(reranker_process)

    return servers


def wait_for_servers(
    reasoning_url: str,
    embedding_url: str,
    reranking_url: Optional[str] = None,
    max_wait_time: int = 300,
    check_interval: int = 5,
):
    logger.info("Waiting for vLLM servers to start...")

    start_time = time.time()
    servers_ready = False
    console = Console()

    with console.status(f"Waiting for server... 0/{max_wait_time}s", spinner="dots") as status:
        while time.time() - start_time < max_wait_time:
            try:
                # Try to connect to both servers
                reasoner_ready = requests.get(f"{reasoning_url}/health", timeout=2).status_code == 200
                embedder_ready = requests.get(f"{embedding_url}/health", timeout=2).status_code == 200
                if reranking_url:
                    reranker_ready = requests.get(f"{reranking_url}/health", timeout=2).status_code == 200
                else:
                    reranker_ready = None
                if reasoner_ready and embedder_ready and (reranker_ready is None or reranker_ready):
                    servers_ready = True
                    break
            except (requests.RequestException, ConnectionError):
                pass

            # Update progress bar
            elapsed = min(check_interval, max_wait_time - int(time.time() - start_time))
            status.update(f"Waiting for server... {elapsed}/{max_wait_time}s")
            time.sleep(check_interval)

    if servers_ready:
        logger.info("✓ All vLLM servers are ready!")
    else:
        logger.error(f"❌ Servers failed to start within {max_wait_time} seconds")
    return servers_ready


def cleanup_servers(servers: List):
    """Clean up all server processes"""
    logger.info(f"Shutting down {len(servers)} vLLM server processes...")
    for server in servers:
        if server is not None and server.poll() is None:
            pid = server.pid
            pgid = os.getpgid(pid)
            logger.info(f"Sending SIGTERM to process group {pgid} for server with PID {pid}")
            try:
                os.killpg(pgid, signal.SIGTERM)
                if server is not None:
                    server.terminate()
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server not found.")
            except Exception as e:
                logger.error(f"Error terminating process group {pgid}: {e}")

    # Wait for processes to terminate gracefully
    time.sleep(5)

    # Force kill any remaining processes
    for server in servers:
        if server is not None and server.poll() is None:
            pid = server.pid
            pgid = os.getpgid(pid)
            logger.warning(
                f"Process group {pgid} for server with PID {pid} did not terminate gracefully. Sending SIGKILL."
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
                if server is not None:
                    server.kill()
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} for server already gone.")
            except Exception as e:
                logger.error(f"Error killing process group {pgid}: {e}")
        server = None  # Mark as cleaned up


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


@dataclass
class Result:
    claim_id: int
    claim: str
    pred_label: Optional[str]
    evidence: List[agen.EvidenceItem]


async def run_single(
    claim_id: int,
    claim: str,
    db: pg.DB,
    reasoning_client: AsyncOpenAI,
    embedding_client: AsyncOpenAI,
    reranking_client: Optional[httpx.AsyncClient],
    config: RunConfig,
) -> Result:
    mstart = timer()
    # Generate questions & queries
    logger.debug(f"[CLAIM {claim_id}] Generating questions and queries")
    qq = await qgen.generate_questions_and_queries(
        claim=claim,
        client=reasoning_client,
        model_id=config.reasoning_model.model_id,
        max_retries=config.reasoning_model.max_retries,
        cfg=config.qgen,
    )

    # Retrieve
    logger.debug(f"[CLAIM {claim_id}] Retrieving evidence")
    retrieved = {}
    for question, queries in qq.items():
        # Retrieve evidence for each query
        # Also add the original question to the list of queries
        chunks, _ = await db.retrieve(
            embedding_client=embedding_client,
            embedding_model=config.embedding_model.model_id,
            queries=["fact check " + question] + queries,
            table=f"claim_{claim_id}",
            cfg=config.retrieval,
        )
        # If reranking is enabled, rerank the chunks
        if reranking_client and config.reranking_model:
            chunks = await rrank.rerank_chunks(
                query=question,
                chunks=chunks,
                client=reranking_client,
                model_id=config.reranking_model.model_id,
                cfg=config.reranking,
            )

        retrieved[question] = chunks
    logger.debug(f"[CLAIM {claim_id}][RETRIEVED CHUNKS]\n{retrieved}")

    # Answer
    logger.debug(f"[CLAIM {claim_id}] Synthesizing answers")
    answer_tasks = []
    for question, chunks in retrieved.items():
        # Create tasks for all questions to process in parallel
        task = asyncio.create_task(
            agen.synthesize_answer(
                question=question,
                chunks=chunks,
                client=reasoning_client,
                model_id=config.reasoning_model.model_id,
                max_retries=config.reasoning_model.max_retries,
                cfg=config.agen,
            )
        )
        answer_tasks.append((question, task))

    # Wait for all answer tasks to complete
    qa: List[Dict[str, str]] = []
    evidence: List[agen.EvidenceItem] = []
    for question, task in answer_tasks:
        evidence_item: agen.EvidenceItem = await task
        evidence.append(evidence_item)
        qa.append({"question": question, "answer": evidence_item.answer or "[EMPTY]"})

    # Judge
    logger.debug(f"[CLAIM {claim_id}] Generating final verdict")
    verdict = await judge.generate_verdict(
        claim=claim,
        qa_pairs=qa,
        client=reasoning_client,
        model_id=config.reasoning_model.model_id,
        max_retries=config.reasoning_model.max_retries,
        cfg=config.judge,
    )

    mend = timer()
    logger.debug(f"[CLAIM {claim_id}] Finished in {mend - mstart:.2f}s.")
    return Result(
        claim_id=int(claim_id),
        claim=claim,
        evidence=evidence,
        pred_label=verdict,
    )


async def run(config: RunConfig):
    start = timer()

    # Read JSON file with claims
    try:
        with open(config.claim_fpath, "r") as f:
            data = json.load(f)
        # Add claim_id to each record
        # The current assumption is that the claim_id is the index of the record
        data = [{**record, "claim_id": i} if "claim_id" not in record else record for i, record in enumerate(data)]
        if config.dev.ids is not None:
            data = [record for i, record in enumerate(data) if i in config.dev.ids]
        elif config.dev.limit is not None:
            data = data[: config.dev.limit]
    except FileNotFoundError:
        logger.error(f"File not found: {config.claim_fpath}")
        raise

    # Get clients
    try:
        reasoning_client = instructor.patch(
            AsyncOpenAI(base_url=config.reasoning_model.base_url, api_key=config.reasoning_model.api_key)
        )
        embedding_client = AsyncOpenAI(base_url=config.embedding_model.base_url, api_key=config.embedding_model.api_key)
        if config.reranking_model:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.reranking_model.api_key}",
            }
            reranking_client = httpx.AsyncClient(base_url=config.reranking_model.base_url, headers=headers)
        else:
            reranking_client = None
    except APIConnectionError as e:
        logger.error(f"Failed to connect to the API: {repr(e)}")
        return

    # Create directory if it doesn't exist
    os.makedirs(config.save_dir, exist_ok=True)
    save_fpath = os.path.join(config.save_dir, config.save_fname)
    intermediate_save_fpath = os.path.join(config.save_dir, "results_autosave.json")

    # Setup autosaving
    results: List[Dict] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def autosave():
        logger.info("[AUTOSAVE] Thread started")
        while not stop_event.is_set():
            stop_event.wait(config.save_every)
            if stop_event.is_set():
                break
            with lock:
                if results:
                    try:
                        with open(intermediate_save_fpath, "a") as f:
                            json.dump(results[-1], f)
                            f.write("\n")
                        logger.info(f"[AUTOSAVE] Saved progress to {intermediate_save_fpath}")
                    except Exception as e:
                        logger.error(f"[AUTOSAVE] Failed to save progress: {repr(e)}")

        # Save remaining results before exiting
        with lock:
            if results:
                try:
                    with open(intermediate_save_fpath, "a") as f:
                        json.dump(results, f)
                        f.write("\n")
                    logger.info(f"[AUTOSAVE] Final save to {intermediate_save_fpath}")
                except Exception as e:
                    logger.error(f"[AUTOSAVE] Failed to save progress: {repr(e)}")
        logger.info("[AUTOSAVE] Thread finished")

    # Start autosave thread if enabled
    save_daemon = None
    if config.save_every > 0:
        save_daemon = threading.Thread(target=autosave, daemon=True)
        save_daemon.start()

    failed = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as progress:
        async with pg.DB(
            db_name=config.db.name,
            db_user=config.db.user,
            db_password=config.db.password,
            db_host=config.db.host,
            db_port=config.db.port,
            embedding_dim=config.db.embedding_dim,
        ) as db:
            pbar = progress.add_task(description="Fact checking claims", total=len(data))
            for idx, record in enumerate(data):
                # Don't use idx here, as it may not be the same as the index in the data if we select specific IDs
                claim_id = record["claim_id"]
                claim = record["claim"]
                try:
                    result = await run_single(
                        claim_id=claim_id,
                        claim=claim,
                        db=db,
                        reasoning_client=reasoning_client,
                        embedding_client=embedding_client,
                        reranking_client=reranking_client,
                        config=config,
                    )
                    with lock:
                        results.append(asdict(result))
                        progress.update(pbar, advance=1)
                except Exception as e:
                    logger.error(f"[CLAIM {claim_id}] Error processing claim: {repr(e)}", exc_info=True)
                    with lock:
                        failed.append((claim_id, claim))
                        results.append(
                            {
                                "claim_id": int(claim_id),
                                "claim": claim,
                                "evidence": [],
                                "label": None,
                            }
                        )
                    progress.update(pbar, advance=1)

    if failed:
        logger.info(f"Failed to process {len(failed)} claims with IDs: {', '.join([str(f[0]) for f in failed])}")

    # Save final results to JSON
    try:
        results = [asdict(r) if isinstance(r, Result) else r for r in results]
        results_df = pd.DataFrame(results)
        results_df.to_json(save_fpath, orient="records")
        logger.info(f"✨ Results saved to {save_fpath}")
    except Exception as e:
        logger.error(f"Failed to save results: {repr(e)}")
    finally:
        # Stop autosave thread
        if save_daemon is not None:
            stop_event.set()
            save_daemon.join(timeout=60)

    # Timing
    end = timer()
    logger.info(f"Fact-checked {len(data)} claims in {end - start:.2f}s | AVG: {(end - start) / len(data):.2f}s/claim")


@hydra.main(version_base=None, config_path="../conf", config_name="run")
def main(config: RunConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"Current working directory: {os.getcwd()}")

    # If servers aren't started for us (bundled=false), start them
    if not config.dev.bundled:
        servers = start_servers(
            reasoner=config.reasoning_model, embedder=config.embedding_model, reranker=config.reranking_model
        )
    else:
        servers = None

    def get_host(url):
        # Note: These URLs are of the format http://0.0.0.0:1771 to which /health will be appended
        return f"{urlparse(url).scheme}://{urlparse(url).hostname}:{urlparse(url).port}"

    # Wait for servers to be ready
    servers_ready = wait_for_servers(
        reasoning_url=get_host(config.reasoning_model.base_url),
        embedding_url=get_host(config.embedding_model.base_url),
        reranking_url=get_host(config.reranking_model.base_url) if config.reranking_model else None,
    )

    # Close & Exit if servers are not ready
    if not servers_ready:
        logger.error("Servers not ready after 300s")

    try:
        asyncio.run(run(config))
    finally:
        logger.info("Shutting down vLLM servers...")
        if servers:
            cleanup_servers(servers)


if __name__ == "__main__":
    main()
