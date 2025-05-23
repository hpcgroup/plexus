#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=4
#SBATCH -A <account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu 

module load nccl
module load cudatoolkit/12.4
source <path_to_venv/bin/activate>

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
GPUS_PER_NODE=4

## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
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

# Check for required arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <G_INTRA_R> <G_INTRA_C> <G_INTRA_D> <TRIAL_NUM>"
    exit 1
fi

G_INTRA_R=$1
G_INTRA_C=$2
G_INTRA_D=$3
TRIAL_NUM=$4

SCRIPT="train.py --G_intra_r ${G_INTRA_R} --G_intra_c ${G_INTRA_C} --G_intra_d ${G_INTRA_D} --gpus_per_node ${GPUS_PER_NODE} --num_epochs 10"
SCRIPT="$SCRIPT --data_dir <path_to_dataset>"

run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u $SCRIPT > <path_to_save_results_to>/<output_file_name>_X${G_INTRA_R}Y${G_INTRA_C}Z${G_INTRA_D}_${TRIAL_NUM}.txt 2>&1"

echo $run_cmd
eval $run_cmd
