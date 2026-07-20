#!/bin/bash
cd /pscratch/sd/c/cunyang/gnn/plexus
module load nccl/2.24.3; module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29524 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,2,1,0 MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONPATH="$PYTHONPATH:."
chmod +x ./get_rank.sh
run_one () {
  tag=$1; r=$2; s=$3; shift 3
  timeout 900 srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh \
    python -u examples/train_mini.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 --G_data 1 \
    --gpus_per_node 4 --num_epochs 100 --eval --eval_every 10 \
    --hidden_size 256 --dropout 0.0 --lr 0.002 \
    --lr_schedule cosine --grad_clip_value 1.0 \
    --minibatch_ratio ${r} --minibatch_sampler ${s} \
    --minibatch_unbiased --minibatch_compact --overlap_samp --eval_degree_buckets \
    --data_dir /pscratch/sd/c/cunyang/gnn/plexus/dataset/reddit/processed_part4 \
    "$@" > result/small_acc/reddit_${tag}.log 2>&1
  echo "reddit_${tag}: exit=$? best=$(grep 'TEST: acc' result/small_acc/reddit_${tag}.log | awk '{print $3}' | tr -d ',' | sort -rn | head -1)"
}
echo "== dense regime: ratio 0.1 (coverage ~49) =="
run_one r0.1_uni  0.1  uniform
run_one r0.1_hub  0.1  hub --minibatch_hub_frac 0.3
run_one r0.1_perm 0.1  uniform --minibatch_epoch_perm
echo "== starved regime: ratio 0.01 (coverage ~4.9) =="
run_one r0.01_uni 0.01 uniform
run_one r0.01_hub 0.01 hub --minibatch_hub_frac 0.3
echo "== starved regime: ratio 0.005 (coverage ~2.5) =="
run_one r0.005_uni 0.005 uniform
run_one r0.005_hub 0.005 hub --minibatch_hub_frac 0.3
echo "ALL DONE"
