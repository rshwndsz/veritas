#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CUDA=124
uv venv
uv sync --frozen --no-install-project --no-dev
./veritas/scripts/restore.sh "${DBNAME}" "${DBUSER}" "${DBPASSWORD}" "${DBHOST}" "${DBPORT}" "${DUMPARCHIVE}" "${DUMPDIR}"