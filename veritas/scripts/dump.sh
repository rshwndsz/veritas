#!/bin/bash
set -euo pipefail

# Dump a PostgreSQL database using pg_dump with maximum compression.

timestamp=$(date +%Y%m%d-%H%M%S)

save_dir="${1}"
DBNAME="${2}"
DBUSER="${3:-dsouzars}"
DBPASSWORD="${4:-POSTGRES}"
DBHOST="${5:-localhost}"
DBPORT="${6:-5432}"
NUM_WORKERS="${7:-128}"

# Validate inputs
if [[ -z "${save_dir}" || -z "$DBNAME" || -z "$DBUSER" || -z "$DBPASSWORD" || -z "$DBHOST" || -z "$DBPORT" ]]; then
  echo "Usage: $0 [dump_dir] DBNAME DBUSER DBPASSWORD DBHOST DBPORT"
  exit 1
fi

# Create save_dir if needed
mkdir -p "$save_dir"
echo "Dumping database '$DBNAME' to: ${save_dir}"

# Name the dump directory
dump_dir="${save_dir}/${DBNAME}_${timestamp}"

# Execute pg_dump
# https://serverfault.com/a/1081643
# https://postgrespro.com/list/thread-id/2055062 
# -Fd: directory format
# -Z0: no compression
# -j: number of jobs to run in parallel
# --no-comments: do not include comments in the dump
# --no-owner: do not include ownership information
# --no-acl: do not include access privileges
# -f: output file name
if PGPASSWORD="$DBPASSWORD" pg_dump \
  -h "$DBHOST" -p "$DBPORT" -U "$DBUSER" -d "$DBNAME" \
  -Z0 -j${NUM_WORKERS} -Fd \
  --no-comments --no-owner --no-acl \
  -f "$dump_dir"; then
  echo "Database dump completed successfully."
else
  echo "pg_dump failed!" >&2
  exit 1
fi

# Compress the dump directory
archive_name="${save_dir}/${DBNAME}_${timestamp}.tar.gz"
tar -cf - -C "$save_dir" "$(basename "$dump_dir")" | pigz -p "$NUM_WORKERS" > "$archive_name"

gzip -t "$archive_name"
echo "Dump directory compressed successfully: $archive_name"
