#!/bin/bash
#SBATCH -q premium
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -A m5083
#SBATCH --nodes=16
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

G_INTRA_R=2
G_INTRA_C=2
G_INTRA_D=2


TRAIN_FILE=examples/train_mini.py
PARTITIONS_PER_DIM=8
PARTITIONED_DATA_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/products/products_part${PARTITIONS_PER_DIM}

export CXX=CC 
export CC=cc
export PYTHONPATH="$PYTHONPATH:."
export PLEXUS_COMPACT_COLMAP_CACHE=1
export PLEXUS_COMPACT_TRANSPOSE_T=1

ratio="${1:-0.05}"
G_DATA="${2:-8}"
lr="${3:-0.002}"


LOWP_FLAGS="--allreduce_lowp --allreduce_lowp_dtype bf16"
chmod +x ./get_rank.sh

for dropout in 0.0 0.1 0.2 0.3; do
# dropout=0.0
    SCRIPT="$TRAIN_FILE \
        --G_intra_r ${G_INTRA_R} \
        --G_intra_c ${G_INTRA_C} \
        --G_intra_d ${G_INTRA_D} \
        --G_data ${G_DATA} \
        --gpus_per_node ${GPUS_PER_NODE} \
        --num_epochs 100 \
        --hidden_size 256 \
        --eval \
        --lr ${lr} \
        --minibatch_ratio ${ratio} \
        --minibatch_unbiased \
        --minibatch_compact \
        --overlap_samp \
        --eval \
        --eval_every 1 \
        --fuse_norm_activation \
        --overlap_fwd_comm \
        --overlap_bwd_comm \
        --overlap_linear_bwd \
        --dropout ${dropout} \
        --data_dir $PARTITIONED_DATA_DIR "


    run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh \
        python -u $SCRIPT"

    echo $run_cmd
    eval $run_cmd > products/products_c${G_INTRA_R}_${G_INTRA_C}_${G_INTRA_D}_${G_DATA}_${ratio}_${lr}_${dropout}.log 2>&1

done
