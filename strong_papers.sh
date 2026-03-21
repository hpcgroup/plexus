#!/bin/bash
#SBATCH -p batch
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load cray-mpich/8.1.31
module load amd-mixed/6.2.4
module load cpe/24.11
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
source <path/to/venv/bin/activate>

NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8
GPUS=$(( NNODES * GPUS_PER_NODE ))

## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL="PHB"

export LD_LIBRARY_PATH=<path/to/aws-ofi-rccl/lib> # Placeholder for path to AWS OFI RCCL plugin lib folder

export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47

CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"


G_INTRA_R=${1:-2}
G_INTRA_C=${2:-2}
G_INTRA_D=${3:-2}
G_DATA=${4:-1}
ratio=${5:-0.05}

TRAIN_FILE=examples/train_mini.py
PARTITIONS_PER_DIM=16
PARTITIONED_DATA_DIR=./dataset/papers/undirected_sym/papers_part${PARTITIONS_PER_DIM}

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

    run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2 ./get_rank.sh \
        python -u $SCRIPT"

    echo $run_cmd
    eval $run_cmd > results/papers_G_INTRA_R${G_INTRA_R}_G_INTRA_C${G_INTRA_C}_G_INTRA_D${G_INTRA_D}_G_DATA${G_DATA}_ratio${ratio}.log 2>&1