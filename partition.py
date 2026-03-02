from plexus.utils.dataset import partition_graph_2d

partition_graph_2d(
    file_path="/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers_adj/processed/processed_papers.pt",
    num_partitions=16,
    output_dir="/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers_adj/papers_part16",
    num_workers=16,
)