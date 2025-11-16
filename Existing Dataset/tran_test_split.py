import json
import random
import time
from collections import Counter

import numpy as np
import torch

seed = 42
random.seed(seed)  # Python 的随机数
torch.manual_seed(seed)  # PyTorch CPU 随机数
torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机数

def count_sublist_lengths_3d(nested_3d_list):
    """
    统计三维列表中所有子列表的长度及其出现频率，按从大到小排列

    参数:
        nested_3d_list: 三维列表，例如 [[[1,2], [3]], [[4,5,6], [7], [8,9]]]

    返回:
        排序后的列表，格式为 [(长度, 数量), ...]
    """
    # 展平三维列表，获取所有子列表

    # 统计每个子列表的长度
    lengths = [len(sublist) for sublist in nested_3d_list]

    # 使用Counter统计各长度出现次数
    length_counts = Counter(lengths)

    # 按长度从大到小排序
    sorted_counts = sorted(length_counts.items(), key=lambda x: x[0], reverse=True)

    return sorted_counts
def load_and_preprocess_data(input_file_path):
    """
    加载并预处理数据

    Args:
        input_file_path: 输入数据文件路径
        instruction_path: 指令数据文件路径

    Returns:
        raw_data: 包含所有原始数据的列表
    """
    # 读取原始数据文件
    with open(input_file_path, 'r') as f:
        raw_data = json.load(f)
        print(raw_data[0].keys())

    # 提取基础特征
    features = [d['node_feature'] for d in raw_data]
    names = [d['contract_name'] for d in raw_data]
    edges = [d['edge'] for d in raw_data]
    labels = [d['label'] for d in raw_data]
    instructions = [d['instruction'] for d in raw_data]
    result = count_sublist_lengths_3d(features)
    return raw_data


def split_train_test(data, test_size=0.2, random_state=42):

    from sklearn.model_selection import train_test_split

    # 使用 stratified sampling 保持类别分布
    indices = np.arange(len(data))
    labels = [item['label'] for item in data]

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data






def prepare_train_test_data(input_file_path, batch_size=64, test_size=0.2):
    """
    端到端加载、处理数据并划分训练集/测试集以及批次

    Args:
        input_file_path: 输入数据文件路径
        instruction_path: 指令数据文件路径
        batch_size: 批次大小
        test_size: 测试集比例

    Returns:
        train_batches: 处理后的训练批次数据
        test_batches: 处理后的测试批次数据
    """

    # 1. 加载原始数据
    raw_data = load_and_preprocess_data(input_file_path)
    print(f"成功加载 {len(raw_data)} 条数据")
    print("——————————————————————————————————————————————————————————")
    train_data, test_data = split_train_test(raw_data)
    print("划分训练集和测试集...")
    # 2. 划分训练集和测试集

    # train_data, test_data = split_train_test(raw_data, test_size=test_size)
    print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
    return train_data, test_data

if __name__ == '__main__':
    data_file_path = 'DC/DC_fusion_data.json'

    train_data, test_data = prepare_train_test_data(
        data_file_path,
        batch_size=64
    )

    train_output_file = 'DC/train_dataset.json'
    test_output_file = 'DC/test_dataset.json'
    with open(train_output_file,'w') as f:
        json.dump(train_data, f)
        f.close()
    with open(test_output_file, 'w') as f:
        json.dump(test_data, f)
        f.close()

