#!/bin/bash
# Preprocess papers100M: undirected + symmetric normalization
# No double permutation (transductive training).
#
# NOTE: to_undirected doubles edge count (~1.6B -> ~3.2B), expect high memory usage.

set -euo pipefail

module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate 

RAW_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/arxiv/raw
BASE_OUT=/pscratch/sd/c/cunyang/gnn/plexus/dataset/arxiv/processed

echo "========================================"
echo "Arxiv: undirected + symmetric normalization"
echo "========================================"

python scripts/preprocess_mag.py \
    --name arxiv \
    --input_dir "${RAW_DIR}" \
    --output_dir "${BASE_OUT}" \
    --force_undirected \
    --no_double_perm \
    --no_build_train_adj \

echo "Arxiv done."

