#!/bin/bash
# Create required project directories

# Base directories
mkdir -p configs
mkdir -p data/movielens
mkdir -p models
mkdir -p infrastructure/db
mkdir -p monitoring
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p logs
mkdir -p notebooks

# Copy configuration files to appropriate locations
cp configs/train_config.yaml configs/
cp configs/tune_config.yaml configs/
cp configs/infer_config.yaml configs/

# Copy database initialization script
cp db-init.sql infrastructure/db/init.sql

# Create empty __init__.py files for proper Python module structure
touch src/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "Directory structure created successfully!"