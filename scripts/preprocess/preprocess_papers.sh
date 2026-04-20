#!/bin/bash
# Preprocess papers100M: undirected + symmetric normalization
# No double permutation (transductive training).
#
# NOTE: to_undirected doubles edge count (~1.6B -> ~3.2B), expect high memory usage.

set -euo pipefail
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate 
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

RAW_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers/raw
BASE_OUT=/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers
PARTITIONS=16

P2_PROCESSED=${BASE_OUT}/undirected_sym/processed
P2_PARTITIONED=${BASE_OUT}/undirected_sym/papers_part${PARTITIONS}

echo "========================================"
echo "P2: undirected + symmetric normalization"
echo "========================================"

if [ -f "${P2_PROCESSED}/processed_papers.pt" ]; then
    echo "P2 preprocessed file already exists, skipping preprocessing."
else
    mkdir -p "${P2_PROCESSED}"
    python scripts/preprocess_mag.py \
        --name papers \
        --input_dir "${RAW_DIR}" \
        --output_dir "${P2_PROCESSED}" \
        --force_undirected \
        --no_double_perm \
        --no_build_train_adj
fi

if [ -f "${P2_PARTITIONED}/metadata.pt" ]; then
    echo "P2 partitioned data already exists, skipping partitioning."
else
    mkdir -p "${P2_PARTITIONED}"
    python -c "
import sys; sys.path.insert(0, '.')
from plexus.utils.dataset_mag import partition_graph_2d
partition_graph_2d(
    file_path='${P2_PROCESSED}/processed_papers.pt',
    num_partitions=${PARTITIONS},
    output_dir='${P2_PARTITIONED}',
    num_workers=8,
)
"
fi
echo "P2 done."

echo "Partitioned data: ${P2_PARTITIONED}"
