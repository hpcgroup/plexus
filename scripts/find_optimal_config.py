import argparse
import sys
import os

# 确保脚本可以找到 performance 目录下的模块
# 这假设脚本在 plexus-AE 根目录下运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from performance.comm_model import compute_config_costs
    from performance.comp_model import comp_model
except ImportError:
    print("错误: 无法导入性能模型。")
    print("请确保此脚本位于'plexus-AE'项目的根目录中，")
    print("并且'performance/comm_model.py'和'performance/comp_model.py'文件存在。")
    sys.exit(1)


# 根据您提供的图片存储数据集的统计信息
DATASET_STATS = {
    "Reddit": {
        "nodes": 232965,
        "non_zeros": 114848857,
        "features": 602,
        "classes": 41,
    },
    "ogbn-products": {
        "nodes": 2449029,
        "non_zeros": 126167053,
        "features": 100,
        "classes": 47,
    },
    "Isolate-3-8M": {
        "nodes": 8745542,
        "non_zeros": 1317986044,
        "features": 128,
        "classes": 32,
    },
    "products-14M": {
        "nodes": 14249639,
        "non_zeros": 245036907,
        "features": 128,
        "classes": 32,
    },
    "europe_osm": {
        "nodes": 50912018,
        "non_zeros": 159021338,
        "features": 128,
        "classes": 32,
    },
    "ogbn-papers100M": {
        "nodes": 111059956,
        "non_zeros": 1726745828,
        "features": 100,
        "classes": 172,
    },
    "friendster": {
        "nodes": 65608366,
        "non_zeros": 1806067135,
        "features": 128,
        "classes": 32,
    },
    "mycielskian19": {
        "nodes": 393215,
        "non_zeros": 903194710,
        "features": 128,
        "classes": 32,
    },
    "nlpkkt200": {
        "nodes": 16240000,
        "non_zeros": 440225632,
        "features": 128,
        "classes": 32,
    },
    "nlpkkt160": {
        "nodes": 8345600,
        "non_zeros": 225422112,
        "features": 128,
        "classes": 32,
    },
    "ml_geer": {
        "nodes": 1504002,
        "non_zeros": 110686677,
        "features": 128,
        "classes": 32,
    },
    "hugebubbles-00000": {
        "nodes": 18318143,
        "non_zeros": 54940162,
        "features": 128,
        "classes": 32,
    },
    "ml_laplace": {
        "nodes": 377002,
        "non_zeros": 27582698,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale18-ef16": {
        "nodes": 262144,
        "non_zeros": 4194304,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale19-ef16": {
        "nodes": 524288,
        "non_zeros": 8388608,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale20-ef16": {
        "nodes": 1048576,
        "non_zeros": 16777216,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale21-ef16": {
        "nodes": 2097152,
        "non_zeros": 33554432,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale22-ef16": {
        "nodes": 4194304,
        "non_zeros": 67108864,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale23-ef16": {
        "nodes": 8388608,
        "non_zeros": 134217728,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale24-ef16": {
        "nodes": 16777216,
        "non_zeros": 268435456,
        "features": 128,
        "classes": 32,
    },
    "graph500-scale25-ef16": {
        "nodes": 33554432,
        "non_zeros": 536870912,
        "features": 128,
        "classes": 32,
    },
}

def is_power_of_two(n):
    """检查一个数是否是2的幂。"""
    return (n > 0) and (n & (n - 1) == 0)

def find_optimal_configuration(dataset_name, total_gpus, hidden_dim=128):
    """
    根据给定的数据集和GPU数量，计算并返回最优的3D并行配置。

    Args:
        dataset_name (str): 数据集的名称。
        total_gpus (int): 用于训练的GPU总数。
        hidden_dim (int): GNN模型的隐藏层维度。
    """
    if dataset_name not in DATASET_STATS:
        print(f"错误: 未知的数据集 '{dataset_name}'。")
        print(f"可用数据集: {', '.join(DATASET_STATS.keys())}")
        return

    if not is_power_of_two(total_gpus):
        print(f"错误: GPU总数 ({total_gpus}) 必须是2的幂。")
        return

    stats = DATASET_STATS[dataset_name]
    N = stats["nodes"]
    NNZ = stats["non_zeros"]
    num_features = stats["features"]
    num_classes = stats["classes"]

    print(f"正在为数据集 '{dataset_name}' 在 {total_gpus} 个GPU上寻找最优配置...")
    print(f"参数: N={N}, NNZ={NNZ}, Features={num_features}, Classes={num_classes}, HiddenDim={hidden_dim}")
    print("-" * 40)

    # 假设模型有3个GCN层，这是 train.py 中的默认设置
    # D_list for computation model (不包含最后的类别数)
    d_list_comp = [num_features, hidden_dim, hidden_dim]
    # D_list for communication model (包含最后的类别数)
    d_list_comm = [num_features, hidden_dim, hidden_dim, num_classes]

    # --- 运行模型 ---
    # 假设使用Perlmutter的经验带宽模型
    comm_costs = compute_config_costs(total_gpus, N, d_list_comm, version="v3", machine="perlmutter")
    # 使用默认系数运行计算模型
    comp_costs = comp_model(N, NNZ, total_gpus, d_list_comp)

    # --- 整合结果 ---
    total_costs = {}
    for config, comm_time in comm_costs.items():
        if config in comp_costs:
            total_costs[config] = comm_time + comp_costs[config]

    if not total_costs:
        print("未能计算任何配置的成本。请检查模型实现。")
        return

    # 对总时间进行排序
    sorted_configs = sorted(total_costs.items(), key=lambda item: item[1])

    # --- 打印结果 ---
    print("各3D配置的估算总成本 (通信 + 计算)，按优劣排序:")
    for config, total_time in sorted_configs:
        config_str = f"(X={config[0]}, Y={config[1]}, Z={config[2]})"
        print(f"  - 配置 {config_str:<20}: {total_time:.4f} ms")
        
    print("-" * 40)
    best_config, min_time = sorted_configs[0]
    best_config_str = f"(X={best_config[0]}, Y={best_config[1]}, Z={best_config[2]})"
    print(f"✅ 找到的最优配置是: {best_config_str}，估算总时间为 {min_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用性能模型为Plexus GNN训练找到最优的3D并行配置。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_STATS.keys()),
        help="要使用的数据集名称。"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        help="用于训练的GPU总数 (必须是2的幂)。"
    )
    
    args = parser.parse_args()
    
    find_optimal_configuration(args.dataset, args.gpus)
