# from plexus.utils.dataset import partition_graph_2d

# partition_graph_2d(
#     file_path="/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers_adj/processed/processed_papers.pt",
#     num_partitions=16,
#     output_dir="/pscratch/sd/c/cunyang/gnn/plexus/dataset/papers_adj/papers_part16",
#     num_workers=16,
# )

# partition_graph_2d(
#     file_path="/pscratch/sd/c/cunyang/gnn/plexus/dataset/yelp/processed/processed_yelp.pt",
#     num_partitions=4,
#     output_dir="/pscratch/sd/c/cunyang/gnn/plexus/dataset/yelp/yelp_part4",
#     num_workers=16,
# )

# partition_graph_2d(
#     file_path="/pscratch/sd/c/cunyang/gnn/plexus/dataset/protein_8m/processed/processed_protein.pt",
#     num_partitions=16,
#     output_dir="/pscratch/sd/c/cunyang/gnn/plexus/dataset/protein_8m/protein_part16",
#     num_workers=16,
# )

partition_graph_2d(
    file_path="/pscratch/sd/c/cunyang/gnn/plexus/dataset/amazon_14m/processed/processed_amazon.pt",
    num_partitions=32,
    output_dir="/pscratch/sd/c/cunyang/gnn/plexus/dataset/amazon_14m/amazon_part32",
    num_workers=16,
)