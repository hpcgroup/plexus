#!/bin/bash
cd /pscratch/sd/c/cunyang/gnn/plexus
module load nccl/2.24.3; module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29522 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,2,1,0 MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONPATH="$PYTHONPATH:."
chmod +x ./get_rank.sh
SRUN="srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u examples/train_mini.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 --G_data 1 --gpus_per_node 4 --minibatch_unbiased --minibatch_compact --overlap_samp"

echo "########## A: variance sweep (arxiv, 30ep warmup, K=100) ##########"
for r in 0.02 0.05 0.1 0.2 0.5; do
  for s in uniform hub; do
    tag="var_${s}_r${r}"
    timeout 500 ${SRUN} --num_epochs 30 --hidden_size 256 --lr 0.01 \
      --minibatch_ratio ${r} --minibatch_sampler ${s} \
      --grad_variance_samples 100 \
      --data_dir /pscratch/sd/c/cunyang/gnn/plexus/dataset/arxiv/arxiv_part8 \
      > result/arxiv_test/${tag}.log 2>&1
    echo "${tag}: exit=$? $(grep -o 'relative_var=[0-9.]*' result/arxiv_test/${tag}.log)"
  done
done

echo "########## B: yelp (multilabel f1_micro) ##########"
for cfg in "uniform 0.1" "uniform_perm 0.1" "uniform 0.25"; do
  set -- $cfg; s=$1; r=$2
  PERM=""; [ "$s" = "uniform_perm" ] && PERM="--minibatch_epoch_perm"
  tag="yelp_${s}_r${r}"
  timeout 900 ${SRUN} --num_epochs 100 --eval --eval_every 10 --hidden_size 256 --lr 0.01 \
    --minibatch_ratio ${r} --minibatch_sampler uniform ${PERM} \
    --multilabel_metric f1_micro \
    --data_dir /pscratch/sd/c/cunyang/gnn/plexus/dataset/yelp/yelp_part4 \
    > result/small_acc/${tag}.log 2>&1
  echo "${tag}: exit=$? best_f1=$(grep -oE 'f1_micro [0-9.]+' result/small_acc/${tag}.log | awk '{print $2}' | sort -rn | head -1)"
done

echo "########## C: proteins (rocauc) ##########"
tag="proteins_uniform_r0.1"
timeout 900 ${SRUN} --num_epochs 100 --eval --eval_every 10 --hidden_size 256 --lr 0.01 \
  --minibatch_ratio 0.1 --minibatch_sampler uniform \
  --multilabel_metric rocauc \
  --data_dir /pscratch/sd/c/cunyang/gnn/plexus/dataset/proteins/proteins_part4 \
  > result/small_acc/${tag}.log 2>&1
echo "${tag}: exit=$? best_rocauc=$(grep -oE 'rocauc [0-9.]+' result/small_acc/${tag}.log | awk '{print $2}' | sort -rn | head -1)"
echo "ALL DONE"
