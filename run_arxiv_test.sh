#!/bin/bash
# Smoke-test new samplers on ogbn-arxiv (1 node, 4 GPUs, run inside salloc).
# Usage: bash run_arxiv_test.sh

module load nccl/2.24.3
module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29513
export WORLD_SIZE=4
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

DATA_DIR=/pscratch/sd/c/cunyang/gnn/plexus/dataset/arxiv/arxiv_part8
OUT_DIR=result/arxiv_test
mkdir -p ${OUT_DIR}
chmod +x ./get_rank.sh

COMMON="examples/train_mini.py \
    --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 --G_data 1 \
    --gpus_per_node 4 \
    --num_epochs 60 --eval --eval_every 10 \
    --hidden_size 256 --num_gcn_layers 3 --lr 0.01 \
    --minibatch_ratio 0.1 \
    --minibatch_unbiased --minibatch_compact --overlap_samp \
    --data_dir ${DATA_DIR}"

run_one () {
    name=$1; shift
    echo "=== ${name} ==="
    srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh \
        python -u ${COMMON} "$@" > ${OUT_DIR}/${name}.log 2>&1
    echo "--- ${name}: exit=$? ---"
    grep -E "TEST:|by-degree|\[info\] sampler|OGB|Error|error|Traceback" ${OUT_DIR}/${name}.log | tail -12
}

run_one uniform      --eval_degree_buckets
run_one epoch_perm   --minibatch_epoch_perm --eval_degree_buckets
run_one hub02        --minibatch_sampler hub --minibatch_hub_frac 0.2 --eval_degree_buckets
run_one degree_a1    --minibatch_sampler degree --minibatch_degree_alpha 1.0 --eval_degree_buckets

echo "ALL DONE"
