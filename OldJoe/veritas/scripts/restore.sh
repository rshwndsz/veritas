#!/bin/bash
set -euo pipefail

timestamp=$(date +%Y%m%d-%H%M%S)
SECONDS=0

DBNAME="${1:-veritas}"
DBUSER="${2:-dsouzars}"
DBPASSWORD="${3:-POSTGRES}"
DBHOST="${4:-localhost}"
DBPORT="${5:-5432}"
DUMPDIR="${6:-/home/ubuntu/db}"

NUM_WORKERS=32

# PostgreSQL is assumed to be running by the entrypoint script.

# Create user dsouzars if it doesn't exist
# Note: Standard CREATE USER doesn't have IF NOT EXISTS. We rely on the entrypoint check.
# If running this script standalone, add checks or ignore errors.
sudo -u postgres psql -c "CREATE USER ${DBUSER} WITH PASSWORD '${DBPASSWORD}';" || echo "User ${DBUSER} likely already exists."
sudo -u postgres psql -c "ALTER USER ${DBUSER} WITH SUPERUSER;"
# Create database veritas if it doesn't exist
# Note: Standard CREATE DATABASE doesn't have IF NOT EXISTS. We rely on the entrypoint check.
sudo -u postgres psql -c "CREATE DATABASE ${DBNAME} OWNER ${DBUSER};" || echo "Database ${DBNAME} likely already exists."
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DBNAME} TO ${DBUSER};"
# Load pgvector extension
sudo -u postgres psql -d "${DBNAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Restore db
echo "Restoring DB from ${DUMPDIR} with pg_restore using DBNAME=${DBNAME} USER=${DBUSER}, HOST=${DBHOST}, PORT=${DBPORT}"
PGPASSWORD="${DBPASSWORD}" pg_restore -j "${NUM_WORKERS}" -v -w -Fd -O -d "${DBNAME}" -U "${DBUSER}" -h "${DBHOST}" -p "${DBPORT}" --no-owner --no-acl "${DUMPDIR}"

echo "PostgreSQL DB successfully restored."
duration=$SECONDS
echo "Total time taken: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
