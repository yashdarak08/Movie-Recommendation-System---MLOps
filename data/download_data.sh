#!/bin/bash
# Download the MovieLens latest-small dataset

DATA_DIR="../data/movielens"
mkdir -p ${DATA_DIR}
echo "Downloading MovieLens latest-small dataset..."
wget -O ${DATA_DIR}/ml-latest-small.zip http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
echo "Extracting dataset..."
unzip ${DATA_DIR}/ml-latest-small.zip -d ${DATA_DIR}
echo "Dataset ready in ${DATA_DIR}"
echo "Cleaning up..."
rm ${DATA_DIR}/ml-latest-small.zip
echo "Done."
# Download the MovieLens 100k dataset
echo "Downloading MovieLens 100k dataset..."
wget -O ${DATA_DIR}/ml-100k.zip http://files.grouplens.org/datasets/movielens/ml-100k.zip
echo "Extracting dataset..."
unzip ${DATA_DIR}/ml-100k.zip -d ${DATA_DIR}
echo "Dataset ready in ${DATA_DIR}"
echo "Cleaning up..."
rm ${DATA_DIR}/ml-100k.zip
echo "Done."
# Download the MovieLens 1M dataset
echo "Downloading MovieLens 1M dataset..."
wget -O ${DATA_DIR}/ml-1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
echo "Extracting dataset..."
unzip ${DATA_DIR}/ml-1m.zip -d ${DATA_DIR}
echo "Dataset ready in ${DATA_DIR}"
echo "Cleaning up..."
rm ${DATA_DIR}/ml-1m.zip
echo "Done."
# Download the MovieLens 10M dataset
echo "Downloading MovieLens 10M dataset..."
wget -O ${DATA_DIR}/ml-10m.zip http://files.grouplens.org/datasets/movielens/ml-10m.zip
echo "Extracting dataset..."
unzip ${DATA_DIR}/ml-10m.zip -d ${DATA_DIR}
echo "Dataset ready in ${DATA_DIR}"
echo "Cleaning up..."
rm ${DATA_DIR}/ml-10m.zip
echo "Done."
# Download the MovieLens 20M dataset
echo "Downloading MovieLens 20M dataset..."
wget -O ${DATA_DIR}/ml-20m.zip http://files.grouplens.org/datasets/movielens/ml-20m.zip
echo "Extracting dataset..."
unzip ${DATA_DIR}/ml-20m.zip -d ${DATA_DIR}
echo "Dataset ready in ${DATA_DIR}"
echo "Cleaning up..."
rm ${DATA_DIR}/ml-20m.zip
echo "Done."