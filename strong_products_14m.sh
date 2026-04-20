#!/bin/bash
#SBATCH -q premium
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=4
#SBATCH -A m5083
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu

module load nccl/2.24.3
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate 

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
GPUS_PER_NODE=4

## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=0

G_INTRA_R=${1:-4}
G_INTRA_C=${2:-2}
G_INTRA_D=${3:-2}
G_DATA=${4:-1}
ratio=${5:-0.05}


TRAIN_FILE=examples/train_mini.py
PARTITIONS_PER_DIM=32
PARTITIONED_DATA_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/amazon_14m/amazon_part${PARTITIONS_PER_DIM}

export CXX=CC 
export CC=cc
export PYTHONPATH="$PYTHONPATH:."
export PLEXUS_COMPACT_COLMAP_CACHE=1
export PLEXUS_COMPACT_TRANSPOSE_T=1


lr=0.005
allreduce_lowp=1
allreduce_lowp_dtype=bf16
use_bf16_spmm=0
use_bf16_gemm=0

LOWP_FLAGS=""
if [ "${allreduce_lowp}" -eq 1 ]; then
    LOWP_FLAGS="--allreduce_lowp --allreduce_lowp_dtype ${allreduce_lowp_dtype}"
fi
if [ "${use_bf16_spmm}" -eq 1 ]; then
    LOWP_FLAGS="${LOWP_FLAGS} --bf16_spmm"
fi
if [ "${use_bf16_gemm}" -eq 1 ]; then
    LOWP_FLAGS="${LOWP_FLAGS} --bf16_gemm"
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
    --data_dir $PARTITIONED_DATA_DIR \
    ${LOWP_FLAGS}"

    run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh \
        python -u $SCRIPT"

    echo $run_cmd
    eval $run_cmd > strong/products_14m_sampopt/products_14m_G_INTRA_R${G_INTRA_R}_G_INTRA_C${G_INTRA_C}_G_INTRA_D${G_INTRA_D}_G_DATA${G_DATA}_ratio${ratio}.log 2>&1