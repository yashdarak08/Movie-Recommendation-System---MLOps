#!/bin/bash
# Initialize the PostgreSQL database with schema

set -e

# Path to SQL initialization file
SQL_FILE="/docker-entrypoint-initdb.d/init.sql"

# Check if the SQL file exists
if [ -f "$SQL_FILE" ]; then
    echo "Running database initialization script..."
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$SQL_FILE"
    echo "Database initialization complete."
else
    echo "Error: SQL initialization file not found at $SQL_FILE"
    exit 1
fi