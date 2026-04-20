#!/bin/bash
#flux: -N 2
#flux: -x
#flux: -t 20m
#flux: -q pbatch
#flux: --output "job-logs/strong_protein_8m_{{nnodes}}N_{{id}}.log"

module purge
module load cpe/24.11
rocm_version="6.2.4"
module load PrgEnv-cray
module load rocm/${rocm_version}
module load craype-accel-amd-gfx942
module load cray-python/3.10.10
source /usr/WS1/$USER/distributed-gnn/plexus/my-venv/bin/activate

NNODES=$(flux --parent jobs -no "{nnodes}" "$FLUX_ENCLOSING_ID")
GPUS_PER_NODE=4
GPUS=$(( NNODES * GPUS_PER_NODE ))

## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL="PHB"

# Cray MPI depends on additional CCE runtime libraries that are staged via
# CRAY_LD_LIBRARY_PATH rather than the shorter default LD_LIBRARY_PATH.
if [ -n "${CRAY_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="/usr/WS1/$USER/distributed-gnn/plexus/aws-ofi-nccl/lib:$LD_LIBRARY_PATH"

export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1


G_INTRA_R=${1:-2}
G_INTRA_C=${2:-2}
G_INTRA_D=${3:-2}
G_DATA=${4:-1}
ratio=${5:-0.05}

TRAIN_FILE=examples/train_mini.py
PARTITIONS_PER_DIM=16
PARTITIONED_DATA_DIR=./dataset/protein_8m/protein_part${PARTITIONS_PER_DIM}

export CXX=CC 
export CC=cc
export PYTHONPATH="$PYTHONPATH:."
export PLEXUS_COMPACT_COLMAP_CACHE=1
export PLEXUS_COMPACT_TRANSPOSE_T=1


lr=0.005
allreduce_lowp=1
allreduce_lowp_dtype=bf16

LOWP_FLAGS=""
if [ "${allreduce_lowp}" -eq 1 ]; then
    LOWP_FLAGS="--allreduce_lowp --allreduce_lowp_dtype ${allreduce_lowp_dtype}"
fi

chmod +x ./get_rank.sh

SCRIPT="$TRAIN_FILE \
    --G_intra_r ${G_INTRA_R} \
    --G_intra_c ${G_INTRA_C} \
    --G_intra_d ${G_INTRA_D} \
    --G_data ${G_DATA} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --num_epochs 10 \
    --hidden_size 256 \
    --lr ${lr} \
    --minibatch_ratio ${ratio} \
    --minibatch_unbiased \
    --minibatch_compact \
    --overlap_samp \
    --overlap_fwd_comm \
    --overlap_bwd_comm \
    --overlap_linear_bwd \
    --fuse_norm_activation \
    --vectorize_dp_grad \
    --tune_gemms \
    --timing_end_epoch 5 \
    --data_dir $PARTITIONED_DATA_DIR \
    ${LOWP_FLAGS}"

run_cmd="flux run -N $NNODES -x -n $GPUS ./get_rank.sh \
    python -u $SCRIPT"

echo $run_cmd
eval $run_cmd > results/proteins_8m_G_INTRA_R${G_INTRA_R}_G_INTRA_C${G_INTRA_C}_G_INTRA_D${G_INTRA_D}_G_DATA${G_DATA}_ratio${ratio}.log 2>&1
