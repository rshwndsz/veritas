#!/bin/bash

set -euo pipefail

# Parse arguments
NUKE=false
DISABLE_CONSTRAINTS=false

if [[ "$1" == "--nuke" ]]; then
  NUKE=true
  shift
fi

if [[ "$1" == "--no-unique" ]]; then
  DISABLE_CONSTRAINTS=true
  shift
fi

CSV_DIR=$1
DB_NAME=$2
DB_USER=$3
DB_PASSWORD=$4
DB_HOST=$5
DB_PORT=$6
NUM_WORKERS=$7
EMBEDDING_DIM=$8

echo "ARGS: DB_NAME=$DB_NAME, DB_USER=$DB_USER, DB_PASSWORD=$DB_PASSWORD, DB_HOST=$DB_HOST, DB_PORT=$DB_PORT, NUM_WORKERS=$NUM_WORKERS, EMBEDDING_DIM=$EMBEDDING_DIM"
# Export common vars for parallel
export DB_NAME DB_USER DB_PASSWORD DB_HOST DB_PORT EMBEDDING_DIM DISABLE_CONSTRAINTS

# Function to get the list of indices from CSV filenames
get_indices() {
  find "$CSV_DIR" -maxdepth 1 -name '*.csv' -printf "%f\n" | sed 's/\.csv$//' | sort -n
}

# Verify the indices found
echo "Found the following indices in $CSV_DIR:"
get_indices | paste -sd ',' -
echo "---"


# Create DBUSER
# Spagetti code, but it works
# https://stackoverflow.com/a/49672442
# echo "Creating user $DB_USER"
# sudo -u postgres psql -h localhost -tc  "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'" | grep -q 1 || psql -h localhost -U postgres -c "CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASSWORD';"

# Create DBNAME
# https://stackoverflow.com/a/36591842
# echo "Creating database $DB_NAME"
# sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}';" | grep -q 1 || psql -U postgres -c "CREATE DATABASE ${DB_NAME} WITH OWNER ${DB_USER};"
# sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"

# Assume DBUSER & DBNAME exist
# Easier :)

# Enable pgvector extension
echo "Enabling pgvector extension..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Drop tables in parallel if requested
if [ "$NUKE" = true ]; then
  drop_table() {
    idx=$1
    echo "Dropping claim_${idx}"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "DROP TABLE IF EXISTS claim_${idx} CASCADE;"
  }
  export -f drop_table
  get_indices | parallel -j "$NUM_WORKERS" drop_table {}
fi

# Create tables in parallel (no indexes, no UNIQUE)
echo "Creating tables (no indexes yet)..."
create_table() {
  idx=$1
  echo "Creating claim_${idx}"
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE TABLE IF NOT EXISTS claim_${idx} (
      id BIGSERIAL PRIMARY KEY,
      doc_id TEXT,
      source_url TEXT,
      chunk_index TEXT,
      content TEXT,
      embedding VECTOR(${EMBEDDING_DIM})
    );
  "
}
export -f create_table
get_indices | parallel -j "$NUM_WORKERS" create_table {}

# Parallel insert with COPY (much faster than INSERT)
echo "Loading CSV data in parallel with $NUM_WORKERS workers..."
insert_csv() {
  idx=$1
  csv_file="${CSV_DIR}/${idx}.csv"
  table_name="claim_${idx}"

  echo "Inserting $csv_file into $table_name"
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
    "\COPY ${table_name} (doc_id, source_url, chunk_index, content, embedding) FROM '${csv_file}' WITH (FORMAT csv, HEADER true);"
  echo "✓ Done with $csv_file"
}
export -f insert_csv
export CSV_DIR
get_indices | parallel --lb -j "$NUM_WORKERS" insert_csv {}

# Temporarily disable constraints if requested for faster performance
if [ "$DISABLE_CONSTRAINTS" = true ]; then
  echo "Disabling constraints temporarily to speed up insertion..."
  disable_constraints() {
    idx=$1
    echo "Disabling constraints for claim_${idx}"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
      ALTER TABLE claim_${idx} DISABLE TRIGGER ALL;
    "
  }
  export -f disable_constraints
  get_indices | parallel -j "$NUM_WORKERS" disable_constraints {}
fi

# Add indexes after inserts
echo "Creating GIN (FTS) + HNSW (vector) indexes..."
create_indexes() {
  idx=$1
  echo "Indexing claim_${idx}..."
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE INDEX IF NOT EXISTS claim_${idx}_fts_idx ON claim_${idx} USING GIN (to_tsvector('english', content));
    CREATE INDEX IF NOT EXISTS claim_${idx}_vector_idx ON claim_${idx} USING hnsw (embedding vector_cosine_ops);
  "
}
export -f create_indexes
get_indices | parallel -j "$NUM_WORKERS" create_indexes {}

# Re-enable constraints after insert if they were disabled
if [ "$DISABLE_CONSTRAINTS" = true ]; then
  echo "Re-enabling constraints..."
  enable_constraints() {
    idx=$1
    echo "Re-enabling constraints for claim_${idx}"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
      ALTER TABLE claim_${idx} ENABLE TRIGGER ALL;
    "
  }
  export -f enable_constraints
  get_indices | parallel -j "$NUM_WORKERS" enable_constraints {}
fi

# Add UNIQUE constraints after insert
echo "Adding UNIQUE(doc_id, chunk_index) constraints..."
add_unique_constraint() {
  idx=$1
  echo "Adding UNIQUE constraint to claim_${idx}"
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    ALTER TABLE claim_${idx}
    ADD CONSTRAINT claim_${idx}_doc_chunk_unique UNIQUE (doc_id, chunk_index);
  "
}
export -f add_unique_constraint
get_indices | parallel -j "$NUM_WORKERS" add_unique_constraint {}

echo "All done!"
exit 0