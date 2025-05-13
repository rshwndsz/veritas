#!/bin/bash
set -euo pipefail

DUMPARCHIVE="/home/ubuntu/db.tar.gz"
DUMPDIR="/home/ubuntu/db"
DBNAME="veritas"
DBUSER="dsouzars"
DBPASSWORD="POSTGRES"
DBHOST="localhost"
DBPORT=5432
PGDATA_DIR="/var/lib/postgresql/16/main"


# Start PostgreSQL
echo "Ensuring PostgreSQL is running..."
if ! sudo -u postgres pg_ctlcluster 16 main status > /dev/null; then
    echo "PostgreSQL not running, starting..."
    sudo -u postgres pg_ctlcluster 16 main start
    # Wait for the server to be ready to accept commands
    until pg_isready -h localhost -p 5432 -U postgres &>/dev/null; do
      echo "Waiting for PostgreSQL service..."
      sleep 2
    done
else
    echo "PostgreSQL already running."
fi
echo "PostgreSQL is running."

# Check if the target database exists by querying the default 'postgres' database
echo "Checking if database '${DBNAME}' exists..."
if sudo -u postgres psql -lqt postgres | cut -d \| -f 1 | grep -qw "${DBNAME}"; then
    echo "Database '${DBNAME}' found. Skipping restore script."
    # Optional: Wait until the specific user can connect if needed
    until pg_isready -h localhost -p 5432 -U ${DBUSER} &>/dev/null; do
        echo "Waiting for PostgreSQL to accept connections for user ${DBUSER}..."
        sleep 2
    done
else
    echo "Database '${DBNAME}' not found. Running restore script..."
    # Restore the database (restore.sh handles user/db creation and data load)
    ./veritas/scripts/restore.sh "${DBNAME}" "${DBUSER}" "${DBPASSWORD}" "${DBHOST}" "${DBPORT}" "${DUMPDIR}"
    echo "Restore script finished."
fi

# Continue with the app
echo "Executing command: $@"
exec "$@"