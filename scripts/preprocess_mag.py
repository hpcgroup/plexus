#!/usr/bin/env python
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plexus.utils.dataset_mag import preprocess_graph
from plexus.utils.general import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess graph dataset')
    parser.add_argument('--name', type=str, default='reddit', 
                       help='Dataset name (default: reddit)')
    parser.add_argument('--input_dir', type=str, default='./data/raw',
                       help='Input directory path (default: ./data/raw)')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory path (default: ./data/processed)')
    parser.add_argument('--double_perm', action='store_true', default=True,
                       help='Use double permutation (default: True)')
    parser.add_argument('--no_double_perm', dest='double_perm', action='store_false',
                       help='Disable double permutation')
    parser.add_argument('--unsupervised', action='store_true', default=False,
                       help='Unsupervised mode (default: False)')
    parser.add_argument('--directed', action='store_true', default=False,
                       help='Directed graph mode (default: False)')
    parser.add_argument('--build_train_adj', action='store_true', default=True,
                       help='Precompute train-induced adjacency for --train_adj runs (default: True)')
    parser.add_argument('--no_build_train_adj', dest='build_train_adj', action='store_false',
                       help='Disable precomputing train-induced adjacency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no_undirected', action='store_true', default=False,
                       help='Skip to_undirected conversion for arxiv (keep directed)')
    parser.add_argument('--force_undirected', action='store_true', default=False,
                       help='Force to_undirected conversion for any dataset (e.g. papers100M)')
    parser.add_argument('--norm_type', type=str, default='symmetric',
                       choices=['symmetric', 'row'],
                       help='Normalization type: symmetric (D^-0.5 A D^-0.5) or row (D^-1 A) (default: symmetric)')

    return parser.parse_args()

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    print(f"Starting preprocessing for {args.name} dataset...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    preprocess_graph(
        name=args.name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        double_perm=args.double_perm,
        unsupervised=args.unsupervised,
        directed=args.directed,
        build_train_adj=args.build_train_adj,
        force_no_undirected=args.no_undirected,
        force_undirected=args.force_undirected,
        norm_type=args.norm_type,
    )
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()
