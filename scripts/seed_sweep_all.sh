#!/bin/bash
# Chained seed top-ups + variance-v2. Run via: bash scripts/seed_sweep_all.sh (login node; chains sallocs)
cd /pscratch/sd/c/cunyang/gnn/plexus
cat > /tmp/seed_inner_$$.sh << 'INNER'
#!/bin/bash
cd /pscratch/sd/c/cunyang/gnn/plexus
module load nccl/2.24.3; module load cudatoolkit/12.4
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29526 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,2,1,0 MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONPATH="$PYTHONPATH:."
chmod +x ./get_rank.sh
BASE="srun -N 1 -n 4 -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u examples/train_mini.py --G_intra_r 2 --G_intra_c 2 --G_intra_d 1 --G_data 1 --gpus_per_node 4 --num_epochs 100 --eval --eval_every 10 --hidden_size 256 --dropout 0.0 --lr_schedule cosine --grad_clip_value 1.0 --minibatch_unbiased --minibatch_compact --overlap_samp"
r1 () { tag=$1; shift; timeout 1000 ${BASE} "$@" > result/small_acc/${tag}.log 2>&1; echo "${tag}: exit=$? best=$(grep -oE '(TEST: acc|TEST: f1_micro|TEST: rocauc) [0-9.]+' result/small_acc/${tag}.log | awk '{print $3}' | sort -rn | head -1)"; }
PHASE=$1
if [ "$PHASE" = "A" ]; then  # reddit seeds 1,2 for table cells
  for s in 1 2; do
    r1 reddit_r0.1_uni_s$s   --lr 0.002 --seed $s --minibatch_ratio 0.1   --minibatch_sampler uniform --data_dir dataset/reddit/processed_part4
    r1 reddit_r0.1_hub_s$s   --lr 0.002 --seed $s --minibatch_ratio 0.1   --minibatch_sampler hub --minibatch_hub_frac 0.3 --data_dir dataset/reddit/processed_part4
    r1 reddit_r0.005_uni_s$s --lr 0.002 --seed $s --minibatch_ratio 0.005 --minibatch_sampler uniform --data_dir dataset/reddit/processed_part4
    r1 reddit_r0.005_hub_s$s --lr 0.002 --seed $s --minibatch_ratio 0.005 --minibatch_sampler hub --minibatch_hub_frac 0.3 --data_dir dataset/reddit/processed_part4
  done
elif [ "$PHASE" = "B" ]; then  # products seeds 1,2
  for s in 1 2; do
    r1 products_r0.05_uni_s$s --lr 0.002 --seed $s --minibatch_ratio 0.05 --minibatch_sampler uniform --data_dir dataset/products/products_part8
    r1 products_r0.05_hub_s$s --lr 0.002 --seed $s --minibatch_ratio 0.05 --minibatch_sampler hub --minibatch_hub_frac 0.3 --data_dir dataset/products/products_part8
    r1 products_r0.01_uni_s$s --lr 0.002 --seed $s --minibatch_ratio 0.01 --minibatch_sampler uniform --data_dir dataset/products/products_part8
    r1 products_r0.01_hub_s$s --lr 0.002 --seed $s --minibatch_ratio 0.01 --minibatch_sampler hub --minibatch_hub_frac 0.3 --data_dir dataset/products/products_part8
  done
elif [ "$PHASE" = "C" ]; then  # arxiv seeds + proteins seeds + yelp seeds + variance-v2
  for s in 1 2; do
    r1 arxiv_uni_s$s  --lr 0.01 --seed $s --minibatch_ratio 0.1 --minibatch_sampler uniform --data_dir dataset/arxiv/arxiv_part8
    r1 arxiv_hub_s$s  --lr 0.01 --seed $s --minibatch_ratio 0.1 --minibatch_sampler hub --minibatch_hub_frac 0.2 --data_dir dataset/arxiv/arxiv_part8
    r1 arxiv_perm_s$s --lr 0.01 --seed $s --minibatch_ratio 0.1 --minibatch_sampler uniform --minibatch_epoch_perm --data_dir dataset/arxiv/arxiv_part8
    r1 proteins_uni_s$s --lr 0.01 --seed $s --minibatch_ratio 0.1 --minibatch_sampler uniform --multilabel_metric rocauc --data_dir dataset/proteins/proteins_part4
    r1 yelp_uni_s$s --lr 0.01 --seed $s --hidden_size 512 --minibatch_ratio 0.1 --minibatch_sampler uniform --multilabel_metric f1_micro --data_dir dataset/yelp/yelp_part4
  done
  # variance v2: one frozen model (30ep @0.1), sweep all ratios at same params
  r1 var_v2_arxiv --lr 0.01 --seed 0 --num_epochs 30 --minibatch_ratio 0.1 --minibatch_sampler uniform \
     --grad_variance_samples 100 --grad_variance_ratios 0.02,0.05,0.1,0.2,0.5 --data_dir dataset/arxiv/arxiv_part8
  grep "grad-variance" result/small_acc/var_v2_arxiv.log
fi
echo "PHASE $PHASE DONE"
INNER
for P in A B C; do
  salloc -N 1 --ntasks-per-node=4 --gpus-per-node=4 -C gpu -q interactive -t 175 -A m5083 bash /tmp/seed_inner_$$.sh $P
done
echo "ALL PHASES DONE"
