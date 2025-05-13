#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)

# Variables
OUTPUT_DIR="/data/3/results"
DB_NAME=""
DB_USER=""
DB_PASSWORD=""
DB_HOST=""
DB_PORT=""
NUM_WORKERS=""
EMBEDDING_DIM=""
N_GPUS=""
N_PROCESSES=""
BATCH_SIZE=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --evidence-dir) EVIDENCE_DIR="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --db-name) DB_NAME="$2"; shift ;;
        --db-user) DB_USER="$2"; shift ;;
        --db-password) DB_PASSWORD="$2"; shift ;;
        --db-host) DB_HOST="$2"; shift ;;
        --db-port) DB_PORT="$2"; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift ;;
        --embedding-dim) EMBEDDING_DIM="$2"; shift ;;
        --n-gpus) N_GPUS="$2"; shift ;;
        --n-processes) N_PROCESSES="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "ARGS: OUTPUT_DIR=$OUTPUT_DIR, DB_NAME=$DB_NAME, DB_USER=$DB_USER, DB_HOST=$DB_HOST, DB_PORT=$DB_PORT, NUM_WORKERS=$NUM_WORKERS, EMBEDDING_DIM=$EMBEDDING_DIM, N_GPUS=$N_GPUS, N_PROCESSES=$N_PROCESSES, BATCH_SIZE=$BATCH_SIZE" 

python -m veritas.scripts.chunk hydra.run.dir="${OUTPUT_DIR}/${timestamp}_chunk" evidence_dir=${EVIDENCE_DIR} batch_size=${BATCH_SIZE}
python -m veritas.scripts.embed hydra.run.dir="${OUTPUT_DIR}/${timestamp}_embed" chunks_dir="${OUTPUT_DIR}/${timestamp}_chunk"  n_gpus=${N_GPUS} n_processes_per_gpu=${N_PROCESSES} batch_size=${BATCH_SIZE}
python -m veritas.scripts.merge hydra.run.dir="${OUTPUT_DIR}/${timestamp}_merge" chunks_dir="${OUTPUT_DIR}/${timestamp}_chunk" embeddings_dir="${OUTPUT_DIR}/${timestamp}_embed" num_workers="${NUM_WORKERS}"
./veritas/scripts/insert.sh --no-unique "${OUTPUT_DIR}/${timestamp}_merge" "${DB_NAME}" "${DB_USER}" "${DB_PASSWORD}" "${DB_HOST}" "${DB_PORT}" "${NUM_WORKERS}" "${EMBEDDING_DIM}"
./veritas/scripts/dump.sh "${OUTPUT_DIR}/${timestamp}_dump" "${DB_NAME}" "${DB_USER}" "${DB_PASSWORD}" "${DB_HOST}" "${DB_PORT}" "${NUM_WORKERS}"
