#!/bin/bash
# Preprocess IGB Homogeneous dataset: directed + symmetric normalization.
# No double permutation (transductive training).
#
# Usage: bash preprocess_igb.sh [size] [num_classes] [--resume]
#   e.g.  bash preprocess_igb.sh small 19
#         bash preprocess_igb.sh medium 2983
#         bash preprocess_igb.sh large 19 --resume

set -euo pipefail
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

IGB_ROOT=/pscratch/sd/c/cunyang/gnn/IGB-Datasets
IGB_SIZE=${1:-medium}
NUM_CLASSES=${2:-19}
RESUME_FLAG=""
if [ "${3:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi
PARTITIONS=16

BASE_OUT=/pscratch/sd/c/cunyang/gnn/plexus/dataset/igb_${IGB_SIZE}
PROCESSED=${BASE_OUT}/test/processed
PARTITIONED=${BASE_OUT}/test/igb_${IGB_SIZE}_${NUM_CLASSES}_part${PARTITIONS}

echo "========================================"
echo "IGB ${IGB_SIZE} (${NUM_CLASSES} classes)"
echo "  directed + symmetric normalization"
echo "========================================"

# For large/full: streaming mode does preprocess + partition in one step.
# For smaller sizes: two-step (preprocess .pt, then partition separately).
if [ "${IGB_SIZE}" = "large" ] || [ "${IGB_SIZE}" = "medium" ]; then
    mkdir -p "${PROCESSED}" "${PARTITIONED}"
    python scripts/preprocess_igb.py \
        --igb_root "${IGB_ROOT}" \
        --igb_size "${IGB_SIZE}" \
        --num_classes "${NUM_CLASSES}" \
        --output_dir "${PROCESSED}" \
        --num_partitions "${PARTITIONS}" \
        --partition_output_dir "${PARTITIONED}" \
        --partition_workers 32 \
        --no_double_perm \
        --no_build_train_adj \
        --norm_type symmetric \
        ${RESUME_FLAG}
else
    if [ -f "${PROCESSED}/processed_igb_${IGB_SIZE}_${NUM_CLASSES}.pt" ]; then
        echo "Preprocessed file already exists, skipping preprocessing."
    else
        mkdir -p "${PROCESSED}"
        python scripts/preprocess_igb.py \
            --igb_root "${IGB_ROOT}" \
            --igb_size "${IGB_SIZE}" \
            --num_classes "${NUM_CLASSES}" \
            --output_dir "${PROCESSED}" \
            --no_double_perm \
            --no_build_train_adj \
            --norm_type symmetric
    fi

    if [ -f "${PARTITIONED}/metadata.pt" ]; then
        echo "Partitioned data already exists, skipping partitioning."
    else
        mkdir -p "${PARTITIONED}"
        python -c "
import sys; sys.path.insert(0, '.')
from plexus.utils.dataset_mag import partition_graph_2d
partition_graph_2d(
    file_path='${PROCESSED}/processed_igb_${IGB_SIZE}_${NUM_CLASSES}.pt',
    num_partitions=${PARTITIONS},
    output_dir='${PARTITIONED}',
    num_workers=16,
)
"
    fi
fi

echo "Done."
echo "Partitioned data: ${PARTITIONED}"
