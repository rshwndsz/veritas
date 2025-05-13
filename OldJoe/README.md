# 🧠 Veritas
This repository contains the full pipeline is a part of the 8th addition of the Fever workshop for fact checking and claim verification.
## Docker Container
This Docker container has multiple components:
  - PostgreSQL setup
  - Embedding model downloaded and ready to use: jinaai/jina-embeddings-v3.
  - Reasoning model downloaded and ready to use: Qwen/Qwen3-14B-FP8.
  - PostgreSQL database content copied into the container.
  - App code.

## Evidence knowledge Base construction
To build the knowldge base we conducted mutiple steps:
  - chuncked the evidence documents
  - Used the jinaai embedding model to create the embeddings
  - Used pgvector library to enable efficient storage and indexing of vector data directly in postgres
  - Used postgres to store the evidence knowldge store chuncks and their embedding
  
## Veritas Fact-Checking Pipeline

---

### 🚀 Quickstart

#### 🛠️ Setup Local Environment

##### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
##### 2. Install dependencies & activate new virtualenv
```bash
uv sync
source .venv/bin/activate
```

#### 📚 Build Knowledge Base

##### 1. Chunk Documents
```bash
python -m veritas.scripts.chunk batch_size=8192 evidence_dir=/data/3/datasets/averitec_test/ks/test_2025
```
##### 2. Embed Chunks
```bash
python -m veritas.scripts.embed hydra.run.dir=/data/3/results/20250429-225800_embed chunks_dir=/data/3/results/20250429-225418_chunk/results n_gpus=7 "gpu_ids=[1,2,3,4,5,6,7]" n_processes_per_gpu=8 base_port=3889
```
##### 3. Merge Chunks & Embeddings
```bash
python -m veritas.scripts.merge hydra.run.dir=/data/3/results/20250430-001700_merge chunks_dir=/data/3/results/20250429-225418_chunk/results embeddings_dir=/data/3/results/20250429-225800_embed/results num_workers=96
```
##### 4. Insert into Database
```bash
./veritas/scripts/insert.sh  /data/3/results/20250430-001700_merge/results veritas_test_final dsouzars POSTGRES localhost 5432 128 1024
```
##### Dump DB Snapshot
```bash
./veritas/scripts/dump.sh
```

#### 🧪 Run Fact-Checking

```bash
python -m veritas.scripts.run
```

Override options specified in RunConfig using the `arg=value` format.

To run the whole pipeline on database `veritas_test_final` with 20 chunks being retrieved for each question on 50 claims in total,
```bash
python -m veritas.scripts.run db.name=veritas_test_final retrieval_k=20 limit=50
```

#### 🐳 Docker Usage
Put your HF Token in `docker/hf.token`.
Place your database dump at `docker/db/test.tar.gz`.
Place the claims file at `docker/AVeriTec/test.json`.

```bash
HF_TOKEN=$(cat ./docker/hf.token) docker build --build-arg DB_FILE=./docker/db/test.tar.gz -t veritas:test --secret type=env,id=token,src=HF_TOKEN .
```

#### 🐳Run Docker

```bash
docker run --runtime=nvidia --gpus '"device=3"' veritas:dev claim_fpath=/home/ubuntu/AVeriTeC/dev.json save_fname="dev_veracity_prediction.json"
```
