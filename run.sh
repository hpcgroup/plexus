# Create directories for serial data and logs
mkdir -p ./data/serial
mkdir -p ./data/no_double_perm
mkdir -p ./data/double_perm
mkdir -p ./log

# Run the serial baseline
echo "Running PyG serial baseline..."
python benchmarking/pyg_serial.py --download_path ./data/serial --num_epochs 1000 > log/serial.txt

# Preprocess data for our model
echo "Preprocessing Reddit dataset without double permutation..."
python scripts/preprocess_reddit.py --no_double_perm --input_dir ./no_double_perm/raw --output_dir ./no_double_perm/processed

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./no_double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 1 --G_intra_d 4 --num_epochs 1000 > log/r1c1d4.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./no_double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 2 --G_intra_d 2 --num_epochs 1000 > log/r1c2d2.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./no_double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 4 --G_intra_d 1 --num_epochs 1000 > log/r1c4d1.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./no_double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 2 --G_intra_c 1 --G_intra_d 2 --num_epochs 1000 > log/r2c1d2.txt


echo "Preprocessing Reddit dataset with double permutation..."
python scripts/preprocess_reddit.py --double_perm --input_dir ./double_perm/raw --output_dir ./double_perm/processed

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 1 --G_intra_d 4 --num_epochs 1000 > log/double_perm_r1c1d4.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 2 --G_intra_d 2 --num_epochs 1000 > log/double_perm_r1c2d2.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 1 --G_intra_c 4 --G_intra_d 1 --num_epochs 1000 > log/double_perm_r1c4d1.txt

torchrun --nproc_per_node=4 --master_port=29500 examples/train.py --data_dir ./double_perm/processed --block_aggregation --tune_gemms --gpus_per_node 4 --G_intra_r 2 --G_intra_c 1 --G_intra_d 2 --num_epochs 1000 > log/double_perm_r2c1d2.txt

# Generate plot from logs
echo "Generating loss plot..."
python scripts/plot_loss.py --log_dir ./log

echo "All experiments completed."

