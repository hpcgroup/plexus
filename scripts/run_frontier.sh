#!/bin/bash
#SBATCH -p batch
#SBATCH -A <account> # Placeholder for account
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=8
#SBATCH --nodes=<number-of-nodes> # Placeholder for number of nodes
#SBATCH --ntasks-per-node=8

module load cray-mpich/8.1.31
module load amd-mixed/6.2.4
module load cpe/24.11
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
source <path/to/venv/bin/activate> # Placeholder for virtual environment activation

# Check for required arguments
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <G_INTRA_R> <G_INTRA_C> <G_INTRA_D> <TRIAL_NUM> <train_file> <data_dir> <result_dir>"
    exit 1
fi

G_INTRA_R=$1
G_INTRA_C=$2
G_INTRA_D=$3
TRIAL_NUM=$4
TRAIN_FILE=$5
DATA_DIR=$6
RESULT_DIR=$7

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
export CUDA_DEVICE_MAX_CONNECTIONS=1
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

SCRIPT="$TRAIN_FILE --G_intra_r ${G_INTRA_R} --G_intra_c ${G_INTRA_C} --G_intra_d ${G_INTRA_D} --gpus_per_node ${GPUS_PER_NODE} --num_epochs 10"
SCRIPT="$SCRIPT --data_dir $DATA_DIR"

run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2 ./get_rank.sh python -u $SCRIPT > ${RESULT_DIR}/X${G_INTRA_R}Y${G_INTRA_C}Z${G_INTRA_D}_${TRIAL_NUM}.txt 2>&1"

echo $run_cmd
eval $run_cmd
