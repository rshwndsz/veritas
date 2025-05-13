import json
import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from kb.chunk import Chunk, Chunker, Document, chunk_document
from semantic_text_splitter import TextSplitter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer  # type: ignore

# Setup logging
logger = logging.getLogger(__name__)


def tokenize(batch, tokenizer, max_length: int = 8192) -> Dict[str, torch.Tensor]:
    return tokenizer(batch["text"], max_length=max_length, padding=True, truncation=True, return_tensors="pt")


def embed_chunks(chunks: List[Chunk], model, tokenizer, batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    # Convert text in chunks into a Dataset
    d = Dataset.from_list([{"text": chunk.content} for chunk in chunks])
    d = d.map(lambda batch: tokenize(batch, tokenizer), batched=True, batch_size=1024, desc="Building dataset")
    dataset = d.remove_columns(["text"]).with_format("torch")

    # Build dataloader
    embeddings = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)  # type: ignore

    # Compute embeddings
    pbar = tqdm(total=len(dataset), desc="Embedding chunks")
    for batch in dataloader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            _embeddings = output.last_hidden_state[:, 0]
            _embeddings = F.normalize(_embeddings, p=2, dim=1)
            embeddings.append(_embeddings.cpu())
            pbar.update(len(batch["input_ids"]))
    embeddings = torch.cat(embeddings).detach().cpu().numpy()
    return embeddings


def build_index(
    claim_id: int,
    documents: list[Document],
    text_splitter: Chunker,
    model,
    tokenizer,
    embedding_dir: str | Path,
    batch_size: int,
    device: str,
):
    # Convert each document into chunks
    chunks = []
    for document in documents:
        _chunks = chunk_document(document, text_splitter)
        chunks.extend(_chunks)
    logger.info(f"Found {len(chunks)} chunks from {len(documents)} documents")

    # Embed chunks
    embeddings = embed_chunks(chunks, model=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    # Store embeddings in FAISS vector store
    with console.status("Building FAISS index", spinner="monkey"):
        res = faiss.StandardGpuResources()
        dimension = embeddings.shape[1]
        index = faiss.IndexIDMap2(faiss.GpuIndexFlatIP(res, dimension))
        index.add_with_ids(embeddings, np.arange(len(chunks)))  #  type: ignore

    # Save chunks & embeddings
    index_fpath = Path(embedding_dir) / f"faiss_index_{claim_id}.index"
    faiss.write_index(faiss.index_gpu_to_cpu(index), str(index_fpath))

    metadata_fpath = Path(embedding_dir) / f"faiss_metadata_{claim_id}.pkl"
    with open(metadata_fpath, "wb") as meta_file:
        pickle.dump(chunks, meta_file)

    logger.info(f"Saved index and chunks in {embedding_dir}.")


def main(
    evidence_dir: str | Path,
    claim_ids: list[int],
    embedding_dir: str | Path,
    chunk_size: int = 2048,
    model_id: str = "Alibaba-NLP/gte-base-en-v1.5",
    batch_size: int = 64,
    device: str = "cuda",
    force: bool = False,
):
    # Setup
    # Check if embedding_dir exists
    if os.path.exists(embedding_dir) and not force:
        logger.error(f"Embedding directory {embedding_dir} already exists. Use --force to overwrite.")
        return
    elif os.path.exists(embedding_dir) and force:
        logger.warning(f"Overwriting embeddings in {embedding_dir}. Make sure you want to do this.")
    else:
        logger.info(f"Creating embeddings directory at {embedding_dir}")
    os.makedirs(embedding_dir, exist_ok=True)

    # Collect evidence
    logger.info(f"Collecting evidence from {evidence_dir}")
    evidence = {}
    evidence_dir = Path(evidence_dir)
    for claim_id in claim_ids:
        with open(evidence_dir / f"{claim_id}.json", "r", encoding="utf-8") as f:
            documents = [Document(**json.loads(line)) for line in f]
            evidence[claim_id] = documents

    # Load tokenizer & model
    logger.info(f"Loading model & tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    # Enable multi-GPU processing
    # model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Build semantic text splitter
    text_splitter = TextSplitter(chunk_size)

    for i, (claim_id, documents) in enumerate(evidence.items()):
        logger.info(f"Processing evidence for claim {claim_id} [{i + 1}/{len(claim_ids)}]")
        start = timer()
        build_index(
            claim_id,
            documents,
            text_splitter=text_splitter,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            embedding_dir=embedding_dir,
            device=device,
        )
        end = timer()
        logger.info(f"✨ Processed claim {claim_id} in {end - start:.2f}s.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--evidence_dir",
        type=str,
        default="/data/3/datasets/AVeriTeC/data_store/knowledge_store/train/train_0_999",
    )
    parser.add_argument(
        "--claim",
        type=int,
        nargs="+",
        default=[0],  # Hunder Biden :)
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="/data/3/embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-base-en-v1.5",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing embeddings.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        evidence_dir=args.evidence_dir,
        claim_ids=args.claim,
        embedding_dir=args.embedding_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        model_id=args.model,
        device=args.device,
        force=args.force,
    )
