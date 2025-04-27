#!/bin/bash

# Loop through all directories in the current folder
for gpu_dir in */; do
    gpu_dir=${gpu_dir%/}  # Remove trailing slash
    
    # Check if it is a directory
    if [ -d "$gpu_dir" ]; then
        
        # Loop through subdirectories 0, 1, and 2
        for sub_dir in 0 1 2; do
            dir="./$gpu_dir/$sub_dir"
            
            # Check if the subdirectory exists
            if [ -d "$dir" ]; then
                # Run the Python script inside the subdirectory
                (cd "$dir" && python ../../process_comm_model.py)
                
                # Copy and rename times.npy to the top-level directory
                npy_file="$dir/times.npy"
                if [ -f "$npy_file" ]; then
                    cp "$npy_file" "./times_${gpu_dir}_${sub_dir}.npy"
                    echo "Copied and renamed $npy_file to ./times_${gpu_dir}_${sub_dir}.npy"
                fi
            else
                echo "Subdirectory $dir not found, skipping."
            fi
        done
    fi
done

# Run the final plotting script
python plot_comm_model_avg.py
