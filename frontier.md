# Running Plexus on Frontier

## Step 1: Create the Python Environment

Run the environment setup script:

```bash
bash create_python_env_frontier.sh
```

This creates a virtual environment at `/lustre/orion/csc547/scratch/$USER/my-venv` and installs PyTorch (ROCm 6.2.4), torch_geometric, numpy, axonn, ogb, and the AWS OFI RCCL plugin.

## Step 2: Update SLURM Scripts with Environment Paths

In the following four scripts:
- `strong_papers.sh`
- `strong_products_2m.sh`
- `strong_products_14m.sh`
- `strong_protein_8m.sh`

Change **line 13** from:
```bash
source <path/to/venv/bin/activate>
```
to:
```bash
source /lustre/orion/csc547/scratch/$USER/my-venv/bin/activate
```

Change **line 29** from:
```bash
export LD_LIBRARY_PATH=<path/to/aws-ofi-rccl/lib>
```
to:
```bash
export LD_LIBRARY_PATH=/lustre/orion/csc547/scratch/$USER/aws-ofi-rccl/lib
```

## Step 3: Verify with a Test Run

Submit a single-node test job:

```bash
sbatch -N 1 strong_products_2m.sh 2 2 2 1 0.05
```

Check the output log. The last few lines should look similar to:

```
Epoch: 006, Train Loss: 0.5623
Epoch: 007, Train Loss: 0.5487
Epoch: 008, Train Loss: 0.5353
Epoch: 009, Train Loss: 0.5280
rank 0 Peak GPU memory: 3.22 GB
rank 5 Peak GPU memory: 3.14 GB
rank 1 Peak GPU memory: 3.14 GB
rank 4 Peak GPU memory: 3.15 GB
rank 2 Peak GPU memory: 3.15 GB
rank 6 Peak GPU memory: 3.15 GB
rank 3 Peak GPU memory: 3.14 GB
rank 7 Peak GPU memory: 3.22 GB
```

If you see decreasing train loss across epochs followed by per-rank GPU memory usage, the setup is working correctly.

## Step 4: Submit All Jobs

Once the test run succeeds, submit all scaling experiments:

```bash
bash submit.sh
```
