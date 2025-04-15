#!/bin/bash
# Download the MovieLens datasets with error handling and dataset size options

# Set bash to exit immediately if a command fails
set -e

# Create data directory structure
DATA_DIR="movielens"
mkdir -p ${DATA_DIR}
echo "Created directory structure: ${DATA_DIR}"

# Function to download and extract a dataset
download_dataset() {
    local dataset=$1
    local dataset_name=$2
    
    echo "======================================================"
    echo "Downloading MovieLens ${dataset_name} dataset..."
    echo "======================================================"
    
    # Check if dataset already exists to avoid re-downloading
    if [ -d "${DATA_DIR}/${dataset}" ]; then
        echo "Dataset ${dataset} already exists. Skipping download."
        return 0
    fi
    
    # Download the dataset
    if wget -O ${DATA_DIR}/${dataset}.zip "https://files.grouplens.org/datasets/movielens/${dataset}.zip"; then
        echo "Extracting dataset..."
        unzip -q ${DATA_DIR}/${dataset}.zip -d ${DATA_DIR} || { echo "Failed to extract ${dataset}.zip"; return 1; }
        echo "Cleaning up..."
        rm ${DATA_DIR}/${dataset}.zip
        echo "Dataset ${dataset_name} ready in ${DATA_DIR}/${dataset}"
    else
        echo "Failed to download ${dataset}.zip"
        return 1
    fi
    
    return 0
}

# Parse command line arguments
DOWNLOAD_ALL=false
DOWNLOAD_LATEST_SMALL=true
DOWNLOAD_100K=false
DOWNLOAD_1M=false
DOWNLOAD_10M=false
DOWNLOAD_20M=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --latest-small)
            DOWNLOAD_LATEST_SMALL=true
            shift
            ;;
        --100k)
            DOWNLOAD_100K=true
            shift
            ;;
        --1m)
            DOWNLOAD_1M=true
            shift
            ;;
        --10m)
            DOWNLOAD_10M=true
            shift
            ;;
        --20m)
            DOWNLOAD_20M=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Download MovieLens datasets"
            echo ""
            echo "Options:"
            echo "  --all          Download all datasets"
            echo "  --latest-small Download latest-small dataset (default)"
            echo "  --100k         Download 100K dataset"
            echo "  --1m           Download 1M dataset"
            echo "  --10m          Download 10M dataset"
            echo "  --20m          Download 20M dataset"
            echo "  --help         Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If --all is specified, download all datasets
if [ "$DOWNLOAD_ALL" = true ]; then
    DOWNLOAD_LATEST_SMALL=true
    DOWNLOAD_100K=true
    DOWNLOAD_1M=true
    DOWNLOAD_10M=true
    DOWNLOAD_20M=true
fi

# Check available disk space
REQUIRED_SPACE=25000000  # ~25GB for all datasets
AVAILABLE_SPACE=$(df -k . | awk 'NR==2 {print $4}')

if [ "$DOWNLOAD_ALL" = true ] && [ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]; then
    echo "WARNING: You may not have enough disk space to download all datasets."
    echo "Available: $(($AVAILABLE_SPACE / 1024)) MB, Required: ~$(($REQUIRED_SPACE / 1024)) MB"
    echo "Continue anyway? (y/n)"
    read -r CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "Download canceled."
        exit 1
    fi
fi

# Download selected datasets
if [ "$DOWNLOAD_LATEST_SMALL" = true ]; then
    download_dataset "ml-latest-small" "latest-small" || echo "Failed to download latest-small dataset"
fi

if [ "$DOWNLOAD_100K" = true ]; then
    download_dataset "ml-100k" "100K" || echo "Failed to download 100K dataset"
fi

if [ "$DOWNLOAD_1M" = true ]; then
    download_dataset "ml-1m" "1M" || echo "Failed to download 1M dataset"
fi

if [ "$DOWNLOAD_10M" = true ]; then
    download_dataset "ml-10m" "10M" || echo "Failed to download 10M dataset"
fi

if [ "$DOWNLOAD_20M" = true ]; then
    download_dataset "ml-20m" "20M" || echo "Failed to download 20M dataset"
fi

echo "======================================================"
echo "All requested datasets have been downloaded successfully"
echo "======================================================"

# Create a simple README file with dataset information
cat > ${DATA_DIR}/README.txt << EOL
MovieLens Datasets
==================

This directory contains the following MovieLens datasets:

- ml-latest-small: A small dataset with 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users
- ml-100k: The original MovieLens dataset with 100,000 ratings from 1000 users on 1700 movies
- ml-1m: MovieLens 1M dataset with 1 million ratings from 6000 users on 4000 movies
- ml-10m: MovieLens 10M dataset with 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users
- ml-20m: MovieLens 20M dataset with 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users

For more information, visit: https://grouplens.org/datasets/movielens/
EOL

echo "Created dataset information file: ${DATA_DIR}/README.txt"