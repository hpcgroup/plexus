#!/bin/bash
cd /pscratch/sd/c/cunyang/gnn/plexus
module load nccl/2.24.3; module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29523 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,2,1,0 MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONPATH="$PYTHONPATH:."
chmod +x ./get_rank.sh
run_one () {
  tag=$1; shift
  timeout 1100 srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh \
    python -u examples/train_mini.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 --G_data 1 \
    --gpus_per_node 4 --num_epochs 200 --eval --eval_every 20 \
    --minibatch_unbiased --minibatch_compact --overlap_samp \
    --multilabel_metric f1_micro --lr_schedule cosine --grad_clip_value 1.0 \
    --data_dir /pscratch/sd/c/cunyang/gnn/plexus/dataset/yelp/yelp_part4 \
    "$@" > result/small_acc/yelp_${tag}.log 2>&1
  echo "yelp_${tag}: exit=$? best_f1=$(grep -oE 'TEST: f1_micro [0-9.]+' result/small_acc/yelp_${tag}.log | awk '{print $3}' | sort -rn | head -1)"
}
run_one h512_do0_uni   --hidden_size 512 --dropout 0.0 --lr 0.01 --minibatch_ratio 0.1 --minibatch_sampler uniform
run_one h512_do0_hub   --hidden_size 512 --dropout 0.0 --lr 0.01 --minibatch_ratio 0.1 --minibatch_sampler hub --minibatch_hub_frac 0.3
run_one h512_do01_uni  --hidden_size 512 --dropout 0.1 --lr 0.01 --minibatch_ratio 0.1 --minibatch_sampler uniform
run_one h512_do0_lr3   --hidden_size 512 --dropout 0.0 --lr 0.003 --minibatch_ratio 0.1 --minibatch_sampler uniform
echo "ALL DONE"
