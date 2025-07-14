#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read training log files from the log folder and plot loss curves over epochs
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """
    Parse log file to extract epoch and loss data
    
    Args:
        file_path: log file path
        
    Returns:
        tuple: (epochs, losses) two lists
    """
    epochs = []
    losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Use regex to match "Epoch: XXX, Train Loss: Y.YYYY" format
            match = re.match(r'Epoch:\s*(\d+),\s*Train Loss:\s*([\d.]+)', line.strip())
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return epochs, losses

def plot_loss_curves(log_dir):
    """
    Plot loss curves for all log files
    
    Args:
        log_dir: log folder path
    """
    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Color list, ensure enough colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get all txt files
    log_path = Path(log_dir)
    txt_files = list(log_path.glob('*.txt'))
    
    if not txt_files:
        print(f"No txt files found in {log_dir}")
        return
    
    not_drawn_files = []  # Track files that were not drawn and reasons
    # Draw a line for each file
    for i, file_path in enumerate(sorted(txt_files)):
        try:
            epochs, losses = parse_log_file(file_path)
            
            if not epochs or not losses:
                reason = f"No valid epoch data found (format mismatch or empty content)"
                not_drawn_files.append((file_path.name, reason))
                print(f"Warning: {file_path.name} {reason}")
                continue
            
            # Filename (without .txt extension) as legend
            legend_name = file_path.stem
            
            # Select color
            color = colors[i % len(colors)]
            
            # Draw line
            plt.plot(epochs, losses, label=legend_name, color=color, linewidth=2, alpha=0.8)
            
            print(f"Processed: {file_path.name} - {len(epochs)} epochs")
            
        except Exception as e:
            not_drawn_files.append((file_path.name, f"Parse error: {e}"))
            print(f"Error processing file {file_path.name}: {e}")
    
    if not_drawn_files:
        print("\nThe following files were not drawn:")
        for fname, reason in not_drawn_files:
            print(f"  {fname}: {reason}")
    else:
        print("\nAll files were successfully drawn!")
    
    # Set chart properties
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    # plt.title('', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to ensure legend is not clipped
    plt.tight_layout()
    
    # Save image
    output_path = log_path / 'loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")
    
    # Show chart
    plt.show()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot loss curves from training log files')
    parser.add_argument('--log_dir', type=str, default="/home/cc/plexus/log",
                       help='Path to the log directory (default: /home/cc/plexus/log)')
    
    args = parser.parse_args()
    log_dir = args.log_dir
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} does not exist")
        return
    
    print("Starting to process log files...")
    plot_loss_curves(log_dir)
    print("Processing completed!")

if __name__ == "__main__":
    main() 