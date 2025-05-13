FROM ubuntu:22.04 AS dbbuilder
ARG DEBIAN_FRONTEND=noninteractive
ARG DB_FILE=./data/db/dev.tar.gz
ENV DB_FILE=${DB_FILE}

RUN apt-get update && apt-get install -y wget gnupg2 && \
    echo "deb http://apt.postgresql.org/pub/repos/apt jammy-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list && \
    wget -qO - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libpq-dev \
    sudo \
    pigz \
    postgresql-16 \
    postgresql-client-16 \
    postgresql-server-dev-all \
    postgresql-16-pgvector \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy DB & Extract
COPY ${DB_FILE} /home/ubuntu/db.tar.gz
RUN echo "localhost:5432:veritas:dsouzars:POSTGRES" > /home/ubuntu/.pgpass && \
    chmod 600 /home/ubuntu/.pgpass && mkdir -p /home/ubuntu/db && \
    pigz -p 32 -dc -1 "/home/ubuntu/db.tar.gz" | tar -C "/home/ubuntu/db" --strip-components 1 -xf -

# Restore DB
COPY --chown=postgres:postgres veritas/scripts/restore.sh /home/ubuntu/restore.sh
RUN echo "Starting PostgreSQL for restore..." && \
    pg_ctlcluster 16 main start && \
    echo "Waiting for PostgreSQL readiness..." && \
    timeout 120 bash -c 'until pg_isready -h localhost -p 5432 -U postgres; do echo "Waiting..."; sleep 2; done' && \
    echo "PostgreSQL ready. Running restore script..." && \
    bash /home/ubuntu/restore.sh veritas dsouzars POSTGRES localhost 5432 /home/ubuntu/db || true && \
    echo "Restore script finished. Stopping PostgreSQL..." && \
    pg_ctlcluster 16 main stop && \
    echo "PostgreSQL stopped. Cleaning up dump files..." && \
    rm -rf /home/ubuntu/db /home/ubuntu/db.tar.gz /home/ubuntu/.pgpass


FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive


# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV HF_HOME=/home/ubuntu/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y wget gnupg2 && \
    echo "deb http://apt.postgresql.org/pub/repos/apt jammy-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list && \
    wget -qO - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    unzip \
    git \
    wget \
    ssh \
    vim \
    htop \
    tmux \
    screen \
    net-tools \
    iputils-ping \
    libpq-dev \
    sudo \
    pigz \
    postgresql-16 \
    postgresql-client-16 \
    postgresql-server-dev-all \
    postgresql-16-pgvector \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create ubuntu user and set up home directory
RUN useradd -m -d /home/ubuntu -s /bin/bash ubuntu && chown -R ubuntu:ubuntu /home/ubuntu

RUN echo "ubuntu ALL=(postgres) NOPASSWD: ALL" >> /etc/sudoers

# Switch to the ubuntu user
USER ubuntu
WORKDIR /home/ubuntu

ENV EC2_HOME=/home/ubuntu

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV UV_CUDA=124

# Install dependencies
COPY pyproject.toml .
COPY .python-version .
RUN uv sync --extra eval

# 'Activate' virtual environment
# Place executables in the environment at the front of the path
# https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="./.venv/bin:$PATH"

# Download models
RUN --mount=type=secret,id=token,env=TOKEN huggingface-cli download Qwen/Qwen3-14B-FP8 --token=${TOKEN}
RUN --mount=type=secret,id=token,env=TOKEN huggingface-cli download jinaai/jina-embeddings-v3 --token=${TOKEN}
RUN --mount=type=secret,id=token,env=TOKEN huggingface-cli download Alibaba-NLP/gte-reranker-modernbert-base --token=${TOKEN}

# Copy in pre-populated PGDATA
COPY --from=dbbuilder /var/lib/postgresql/16/main /var/lib/postgresql/16/main
RUN sudo -u postgres chown -R postgres:postgres /var/lib/postgresql/16/main && sudo -u postgres chmod -R 700 /var/lib/postgresql/16/main

# Copy code & claims data
RUN mkdir -p /home/ubuntu/data
COPY --chown=ubuntu:ubuntu ./*.sh /home/ubuntu/
COPY --chown=ubuntu:ubuntu ./*.py /home/ubuntu/
COPY --chown=ubuntu:ubuntu README.md /home/ubuntu/
COPY --chown=ubuntu:ubuntu veritas /home/ubuntu/veritas
COPY --chown=ubuntu:ubuntu ./data/AVeriTeC/data /home/ubuntu/data/
RUN chmod +x /home/ubuntu/*sh && chmod +x /home/ubuntu/veritas/scripts/*.sh

# Entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["./run_system.sh"]
