#!/bin/bash
# papers100M accuracy runs with new samplers (undirected graph, eval + degree buckets).
# Baseline to beat: result/dp_papers/paper_ratio0.02_dp1_lr0.01.log (TEST 62.8%, uniform).
#
# Usage:
#   sbatch -N 32 run_papers_acc.sh <G_R> <G_C> <G_D> <G_DATA> <ratio> <sampler> <perm> [seed]
#     sampler: uniform | hub | degree
#     perm:    0 | 1  (epoch permutation)
# Examples (see bottom of file for the standard three submissions).
#SBATCH -q regular
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -A m5083
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu

module load nccl/2.24.3
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate

NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}"
GPUS_PER_NODE=4
GPUS=$(( NNODES * GPUS_PER_NODE ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}
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
export PYTHONPATH="$PYTHONPATH:."
export PLEXUS_COMPACT_COLMAP_CACHE=1
export PLEXUS_COMPACT_TRANSPOSE_T=1

G_INTRA_R=${1:-4}
G_INTRA_C=${2:-4}
G_INTRA_D=${3:-8}
G_DATA=${4:-1}
ratio=${5:-0.02}
sampler=${6:-uniform}
perm=${7:-0}
seed=${8:-0}
hidden=${9:-128}
# overlap=1 adds the async-comm perf flags (extra buffers; too big for hidden 256)
overlap=${10:-0}
# March baseline (62.8%) predates the dropout feature entirely -> default 0.0
dropout=${11:-0.0}
hubfrac=${12:-0.2}
clip=${13:-0}       # >0 enables --grad_clip_value
sched=${14:-constant}  # constant | cosine
tsteps=${15:-10}    # sampler=train_hub: steps per epoch
fast=${16:-0}       # 1 adds --bf16_spmm --bf16_gemm
lr=${17:-0.01}
wu=${18:-0}      # linear LR warmup epochs (cosine only)
dsrc=${19:-0}    # 1: DP groups draw diverse uniform sources (train_hub)

EXTRA_FLAGS="--lr_schedule ${sched}"
if [ "${clip}" != "0" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --grad_clip_value ${clip}"
fi
if [ "${fast}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --bf16_spmm --bf16_gemm"
fi
if [ "${wu}" != "0" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --lr_warmup_epochs ${wu}"
fi
SAMPLER_FLAGS="--minibatch_sampler ${sampler}"
if [ "${sampler}" = "hub" ]; then
    SAMPLER_FLAGS="${SAMPLER_FLAGS} --minibatch_hub_frac ${hubfrac}"
fi
if [ "${sampler}" = "degree" ]; then
    SAMPLER_FLAGS="${SAMPLER_FLAGS} --minibatch_degree_alpha 1.0"
fi
if [ "${sampler}" = "train_hub" ]; then
    SAMPLER_FLAGS="${SAMPLER_FLAGS} --minibatch_hub_frac ${hubfrac} --minibatch_train_steps ${tsteps}"
    if [ "${dsrc}" = "1" ]; then
        SAMPLER_FLAGS="${SAMPLER_FLAGS} --minibatch_dp_diverse_src"
    fi
fi
PERM_FLAG=""
name_perm=""
if [ "${perm}" = "1" ]; then
    PERM_FLAG="--minibatch_epoch_perm"
    name_perm="_perm"
fi

TRAIN_FILE=examples/train_mini.py
PARTITIONS_PER_DIM=16
PARTITIONED_DATA_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers/undirected_sym/papers_part${PARTITIONS_PER_DIM}

OVERLAP_FLAGS=""
if [ "${overlap}" = "1" ]; then
    OVERLAP_FLAGS="--overlap_fwd_comm --overlap_bwd_comm --overlap_linear_bwd --fuse_norm_activation --vectorize_dp_grad"
fi

OUT_DIR=result/papers_acc
mkdir -p ${OUT_DIR}
name_hf=""
if [ "${sampler}" = "hub" ]; then
    name_hf="_hf${hubfrac}"
fi
if [ "${sampler}" = "train_hub" ]; then
    name_hf="_hf${hubfrac}_T${tsteps}"
fi
NAME=papers_${sampler}${name_perm}${name_hf}_r${ratio}_g${G_INTRA_R}x${G_INTRA_C}x${G_INTRA_D}_dp${G_DATA}_lr${lr}_h${hidden}_do${dropout}_c${clip}_${sched}_wu${wu}_ds${dsrc}_ep${EPOCHS:-100}_s${seed}

chmod +x ./get_rank.sh

SCRIPT="$TRAIN_FILE \
    --G_intra_r ${G_INTRA_R} \
    --G_intra_c ${G_INTRA_C} \
    --G_intra_d ${G_INTRA_D} \
    --G_data ${G_DATA} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --num_epochs ${EPOCHS:-100} \
    --eval --eval_every ${EVAL_EVERY:-5} \
    --eval_degree_buckets \
    --hidden_size ${hidden} \
    --dropout ${dropout} \
    --lr ${lr} \
    --seed ${seed} \
    --minibatch_ratio ${ratio} \
    --minibatch_unbiased \
    --minibatch_compact \
    ${SAMPLER_FLAGS} \
    ${PERM_FLAG} \
    ${EXTRA_FLAGS} \
    --overlap_samp \
    ${OVERLAP_FLAGS} \
    --allreduce_lowp --allreduce_lowp_dtype bf16 \
    --data_dir $PARTITIONED_DATA_DIR"

run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=${GPUS_PER_NODE} ./get_rank.sh \
    python -u $SCRIPT"

echo "NAME=${NAME} NNODES=${NNODES} GPUS=${GPUS}"
echo $run_cmd
eval $run_cmd > ${OUT_DIR}/${NAME}.log 2>&1
echo "exit=$? -> ${OUT_DIR}/${NAME}.log"
grep -E "TEST: acc" ${OUT_DIR}/${NAME}.log | tail -5
