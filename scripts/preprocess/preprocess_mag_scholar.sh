#!/bin/bash
# Preprocess MAG-Scholar: SVD reduces 2.78M sparse BoW -> 128 dense features.
# Graph is already undirected; no --force_undirected needed.
# Two-phase: preprocess -> partition (same pattern as papers100M).

set -euo pipefail
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

MAG_RAW_DIR=/pscratch/sd/c/cunyang/gnn/plexus/data/mag
BASE_OUT=/pscratch/sd/c/cunyang/gnn/plexus/dataset/mag_scholar

PARTITIONS=16

# ============================================================
# MAG-Scholar Coarse (10.5M nodes, 265M edges, 8 classes)
# ============================================================
COARSE_PROCESSED=${BASE_OUT}/coarse/processed
COARSE_PARTITIONED=${BASE_OUT}/coarse/mag_coarse_part${PARTITIONS}

echo "========================================"
echo "MAG-Scholar Coarse: preprocessing"
echo "========================================"

if [ -f "${COARSE_PROCESSED}/processed_mag_coarse.pt" ]; then
    echo "Coarse preprocessed file already exists, skipping preprocessing."
else
    mkdir -p "${COARSE_PROCESSED}"
    python scripts/preprocess_mag.py \
        --name mag_coarse \
        --input_dir "${MAG_RAW_DIR}" \
        --output_dir "${COARSE_PROCESSED}" \
        --no_double_perm \
        --no_build_train_adj \
        --norm_type symmetric
fi

echo "========================================"
echo "MAG-Scholar Coarse: 2D partitioning (${PARTITIONS}x${PARTITIONS})"
echo "========================================"

if [ -f "${COARSE_PARTITIONED}/metadata.pt" ]; then
    echo "Coarse partitioned data already exists, skipping partitioning."
else
    mkdir -p "${COARSE_PARTITIONED}"
    python -c "
import sys; sys.path.insert(0, '.')
from plexus.utils.dataset_mag import partition_graph_2d
partition_graph_2d(
    file_path='${COARSE_PROCESSED}/processed_mag_coarse.pt',
    num_partitions=${PARTITIONS},
    output_dir='${COARSE_PARTITIONED}',
    num_workers=4,
)
"
fi
echo "MAG-Scholar Coarse done."

# ============================================================
# MAG-Scholar Fine (12.4M nodes, 345M edges, 253 classes)
# ============================================================
FINE_PROCESSED=${BASE_OUT}/fine/processed
FINE_PARTITIONED=${BASE_OUT}/fine/mag_fine_part${PARTITIONS}

echo "========================================"
echo "MAG-Scholar Fine: preprocessing"
echo "========================================"

if [ -f "${FINE_PROCESSED}/processed_mag_fine.pt" ]; then
    echo "Fine preprocessed file already exists, skipping preprocessing."
else
    mkdir -p "${FINE_PROCESSED}"
    python scripts/preprocess_mag.py \
        --name mag_fine \
        --input_dir "${MAG_RAW_DIR}" \
        --output_dir "${FINE_PROCESSED}" \
        --no_double_perm \
        --no_build_train_adj \
        --norm_type symmetric
fi

echo "========================================"
echo "MAG-Scholar Fine: 2D partitioning (${PARTITIONS}x${PARTITIONS})"
echo "========================================"

if [ -f "${FINE_PARTITIONED}/metadata.pt" ]; then
    echo "Fine partitioned data already exists, skipping partitioning."
else
    mkdir -p "${FINE_PARTITIONED}"
    python -c "
import sys; sys.path.insert(0, '.')
from plexus.utils.dataset_mag import partition_graph_2d
partition_graph_2d(
    file_path='${FINE_PROCESSED}/processed_mag_fine.pt',
    num_partitions=${PARTITIONS},
    output_dir='${FINE_PARTITIONED}',
    num_workers=4,
)
"
fi
echo "MAG-Scholar Fine done."

echo "Partitioned data:"
echo "  Coarse: ${COARSE_PARTITIONED}"
echo "  Fine:   ${FINE_PARTITIONED}"
