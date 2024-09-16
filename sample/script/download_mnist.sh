#!/bin/bash -e

if [ $# -eq 0 ]; then
    echo "Download and decompress the MNIST dataset"
    echo "Usage: $0 DOWNLOAD_DIRECTORY"
    exit 1
fi

# Create a dest. directory
dst_dir=${1}
mkdir -p ${dst_dir}

# Download MNIST binary data
# Training images/labels
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz -P ${dst_dir}
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz -P ${dst_dir}
# Testing images/labels
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -P ${dst_dir}
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -P ${dst_dir}

# Uncompress downloaded data
gunzip ${dst_dir}/train-images-idx3-ubyte.gz
gunzip ${dst_dir}/train-labels-idx1-ubyte.gz
gunzip ${dst_dir}/t10k-images-idx3-ubyte.gz
gunzip ${dst_dir}/t10k-labels-idx1-ubyte.gz
