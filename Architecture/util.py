import json
import logging
import random
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from AVP import *

def load_data(input_file_path):
    # 读取原始数据文件
    with open(input_file_path, 'r') as f:
        data = json.load(f)
        f.close()

    return data

def balance_samples(data):
    """
    对数据集进行正负样本均衡

    Args:
        data: 包含合约数据的列表

    Returns:
        balanced_data: 平衡后的数据集
    """
    # 分离正负样本
    pos_samples = [item for item in data if item['label'] == 1]  # 正样本
    neg_samples = [item for item in data if item['label'] == 0]  # 负样本

    n_neg = len(neg_samples)
    n_pos = len(pos_samples)

    logging.info(f"Original dataset: {n_neg} negative samples, {n_pos} positive samples")

    # 如果正样本数量少于负样本且存在正样本
    if n_pos < n_neg and n_pos > 0:
        # 计算需要复制的次数
        repeat_times = n_neg // n_pos
        remainder = n_neg % n_pos

        # 复制正样本（随机选择以满足数量）
        oversampled_pos = pos_samples * repeat_times
        oversampled_pos += random.sample(pos_samples, remainder)

        # 合并数据集
        balanced_data = neg_samples + oversampled_pos

        # 打乱顺序
        random.shuffle(balanced_data)

        logging.info(f"Balanced dataset: {len(neg_samples)} negative samples, {len(oversampled_pos)} positive samples")
    else:
        balanced_data = data
        logging.info(
            f"No balancing needed. Dataset remains unchanged with {n_neg} negative samples, {n_pos} positive samples")

    return balanced_data


def create_batches(data, batch_size=64):
    """
    将数据划分为批次

    Args:
        data: 输入数据列表
        batch_size: 批次大小

    Returns:
        batches: 划分后的批次列表
    """

    batches = []
    num_samples = len(data)

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]

        batches.append(batch)

    return batches


def prepare_train_data(input_file_path,batch_size):
    from  tqdm import tqdm
    logging.info("开始数据加载和预处理...")
    # 1. 加载原始数据
    raw_data = load_data(input_file_path)
    logging.info(f"成功加载 {len(raw_data)} 条train数据")
    logging.info("——————————————————————————————————————————————————————————")

    logging.info(f"训练集: {len(raw_data)} 条")
    logging.info("——————————————————————————————————————————————————————————")
    # 2. 提取批次中的vul_branch和cri_branch
    logging.info(f"提取训练集的降噪特征")
    processed_train = []
    for batch_idx, batch in enumerate(tqdm(raw_data)):
        data = extract_features_for_train(batch)
        processed_train.append(data)

    logging.info("对训练集进行正负样本均衡...")
    # 3. 对训练集进行正负样本均衡

    balanced_train_data = balance_samples(processed_train)
    logging.info(f"均衡后训练集大小: {len(balanced_train_data)}")
    logging.info("——————————————————————————————————————————————————————————")

    logging.info("划分批次...")
    # 4. 将平衡后的训练集和测试集划分为批次
    train_batches = create_batches(balanced_train_data, batch_size)
    logging.info("——————————————————————————————————————————————————————————")

    # 5. 提取批次中的vul_branch和cri_branch

    return train_batches

def extract_features_for_train(batch):

    label = batch['label']
    edges = batch['edge']

    node_features = batch['node_feature']
    instructions = batch['instruction']
    feature_dim = len(node_features[0])
    if len(edges) == 0:
        node_num = len(node_features)
        for i in range(node_num - 1):
            edges.append([i, i + 1])
    if label == 1:
        # ============ extract vulnerability branch ================
        vulner_branch_nodes = extract_vulner_branch(instructions, edges)  # [[],]
        denoising_features  = [[1]*feature_dim if i in vulner_branch_nodes else [0]*feature_dim for i in range(len(node_features))]
    else:
        denoising_features = None

    batch['denoising_feature'] = denoising_features
    return batch


def prepare_test_data(path):
    logging.info("开始测试数据加载...")
    test_data = load_data(path)
    for data in test_data:
        if len(data['edge'] ) == 0:
            node_num = len(data['node_feature'])
            for i in range(node_num - 1):
                data['edge'].append([i, i + 1])
    return test_data

def extract_vulner_branch(instruction, edge):
    vulner_branch = score_main(instruction, edge)
    return vulner_branch


def concatenate_tensors(tensor_list, dim=0):
    """
    将列表中的tensor沿着指定维度拼接

    Args:
        tensor_list: 包含tensor的列表
        dim: 拼接的维度，默认为0

    Returns:
        拼接后的tensor
    """
    if not tensor_list:
        return None

    # 使用torch.cat进行拼接
    return torch.cat(tensor_list, dim=dim)


def stage_train(model, data, first_optimizer, second_optimizer, pre_optimizer,
                first_fn, second_fn, pre_fn,device):
    for batch in data:
        batch_first_features = []
        batch_y = []
        first_stage_losses = 0.0
        for d in batch:
            y = torch.tensor(d['label']).to(device).unsqueeze(0)
            batch_y.append(y)
            edge = torch.tensor(d['edge']).to(device)
            node_feature = torch.Tensor(d['node_feature']).to(device)
            # ================= 第一阶段学生网络处理 =================
            first_stage_feature = model.student.first_stage(node_feature, edge)  # [1, hidden_dim]
            batch_first_features.append(torch.sum(first_stage_feature.detach(), dim=0, keepdim=True))
            target_dim = torch.ones(first_stage_feature.size()[0]).to(device)
            # ================= 全局降噪特征挖掘 =================
            if d['denoising_feature'] is not None:
                denoising_feature_tensor = torch.Tensor(d['denoising_feature']).to(device)
                denoising_feature = model.denoise_model(denoising_feature_tensor, edge)  # [1, hidden_dim*2]
                loss_1 = first_fn(first_stage_feature, denoising_feature, target_dim)
            else:
                # 当vulner_feature为None时，不计算损失，避免推动输出为零
                loss_1 = 0.0  # 或者使用非常小的权重，如 0.001 * torch.norm(first_stage_feature)

            first_stage_losses = first_stage_losses + loss_1

            # 后向传播等后续步骤
        first_optimizer.zero_grad()
        first_stage_losses.backward()
        first_optimizer.step()

        batch_first_features = torch.cat(batch_first_features, dim=0)
        # ================= 第二阶段学生网络处理 =================
        complete_feature, high_confidence, front_feature = model.student.second_stage(batch_first_features)
        # ================= 自蒸馏语义补充 =================
        complementary_feature = model.complement_model(batch_first_features)
        second_stage_losses = second_fn(high_confidence.log(), complementary_feature)
        second_optimizer.zero_grad()
        second_stage_losses.backward()
        second_optimizer.step()

        pre = model.student.predictor(complete_feature.detach())
        batch_y = torch.cat(batch_y, dim=0)
        pre_losses = pre_fn(pre, batch_y)
        pre_optimizer.zero_grad()
        pre_losses.backward()
        pre_optimizer.step()
    return first_stage_losses, second_stage_losses, pre_losses


def model_test(model, test_data,device):
    detection_pre = []
    truth_list = []
    for data in test_data:
        truth = data['label']
        truth_list.append(truth)
        edge = torch.tensor(data['edge']).to(device)
        node_feature = torch.Tensor(data['node_feature']).to(device)
        contract_pre = model.student(node_feature, edge).squeeze()
        _, predicted_class = torch.max(contract_pre, dim=0)
        detection_pre.append(predicted_class.item())
    detecting_evaluation(detection_pre, truth_list)

def detecting_evaluation(pre, y):
    # 将预测结果转换为类别


    # 计算准确率（Accuracy）
    accuracy = accuracy_score(y, pre)

    # 计算多分类的精确率、召回率和F1分数
    precision = precision_score(y, pre)
    recall = recall_score(y, pre)
    f1 = f1_score(y, pre)

    tn = sum((y_t == 0) and (y_p == 0) for y_t, y_p in zip(y, pre))
    tp = sum((y_t == 1) and (y_p == 1) for y_t, y_p in zip(y, pre))
    fp = sum((y_t == 0) and (y_p == 1) for y_t, y_p in zip(y, pre))
    fn = sum((y_t == 1) and (y_p == 0) for y_t, y_p in zip(y, pre))

    # # 计算混淆矩阵
    # cm = confusion_matrix(y, y_pred_class)

    # # 提取 TP, TN（假设二分类，0 为负类，1 为正类）
    # tn, fp, fn, tp = cm.ravel()  # 适用于二分类问题


    # 改为：
    logging.info(f'检测准确率 (Accuracy): {accuracy:.4f}')
    logging.info(f'检测精确率 (Precision): {precision:.4f}')
    logging.info(f'检测召回率 (Recall): {recall:.4f}')
    logging.info(f'检测F1 分数: {f1:.4f}')
    logging.info(f'TP : {tp}')
    logging.info(f'TN : {tn}')
    logging.info(f'FP : {fp}')
    logging.info(f'FN : {fn}')

def find_up_nodes(graph, vulner_nodes, edges):
    def dfs(node, visited, path):
        if node in visited:
            return
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, visited, path)
    visited = set()
    path = []
    dfs(vulner_nodes[0], visited, path)
    path.remove(vulner_nodes[0])
    forking_nodes = []
    for node in path:
        # 检查当前节点在原图中的出度（即有多少子节点）
        children_count = sum(1 for u, v in edges if u == node)
        if children_count == 2:  # 如果该节点有两个子节点，出度为2
            forking_nodes.append(node)
    return forking_nodes

def find_up_nodes(graph, vulner_nodes, edges):
    def dfs(node, visited, path):
        if node in visited:
            return
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, visited, path)
    visited = set()
    path = []
    dfs(vulner_nodes[0], visited, path)
    path.remove(vulner_nodes[0])
    forking_nodes = []
    for node in path:
        # 检查当前节点在原图中的出度（即有多少子节点）
        children_count = sum(1 for u, v in edges if u == node)
        if children_count == 2:  # 如果该节点有两个子节点，出度为2
            forking_nodes.append(node)
    return forking_nodes

# 找到所有的分叉节点
def find_forking_nodes(graph):
    return [node for node, children in graph.items() if len(children) > 1]

def dfs_branches(node, graph, path, branches, visited=None):
    if visited is None:
        visited = set()

    # 检测环路
    if node in path:
        return

    path.append(node)

    if node not in graph or not graph[node]:  # 叶节点
        branches.append(path[:])  # 将当前路径作为一个分支添加
    else:
        for child in graph[node]:
            dfs_branches(child, graph, path, branches, visited)

    path.pop()  # 回溯
def Condition_Gate(vulner_nodes: object, instructions: object, edges: object):
    """
    找到vulner_nodes上游节点的分支节点，判断该分支节点下的路径是否存在关键数据调用
    """
    graph = defaultdict(list)
    diver_graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        diver_graph[v].append(u)
    up_nodes = []
    for vulner_node in vulner_nodes:
        up_node = find_up_nodes(diver_graph, vulner_node, edges)
        up_nodes = up_nodes + up_node
        # 去除根节点
    up_nodes = list(set(up_nodes))
    up_nodes.remove(0)
    # 找出除去根节点之外所有分支，并找出每一个以up_nodes值为根节点的分支
    fork_nodes = find_forking_nodes(graph)
    fork_nodes.remove(0)
    all_branches = []
    for node in fork_nodes:
        branches = []
        dfs_branches(node, graph, [], branches)
        all_branches.extend(branches)  # 合并所有分支
    # 去除原有的漏洞分支
    all_branches = [
        sublist for sublist in all_branches
        if not any(
            all(num in sublist for num in subset)
            for subset in vulner_nodes
        )
    ]
    # remove sutb_pah

    critical_branches = [branch for branch in all_branches for start in up_nodes if branch[0] == start]
    critical_branches = [sublist for sublist in critical_branches if len(sublist) >= 3]
    critical_instruction = [
        [instructions[b] for b in critical_branch]
        for critical_branch in critical_branches
    ]
    _, valid_branch = Condition_gate_judge(critical_instruction, critical_branches)
    return valid_branch

import itertools
def Condition_gate_judge(branch_instructions, branch_edge):
    valid_branch_index = []
    valid_branch_instructions = []
    for i, branch_instruction in enumerate(branch_instructions):
        # 利用SS-SL模式进行识别分支下的关键数据利用
        flag = 0
        branch_all_instructions = list(itertools.chain(*branch_instruction))
        for call_instru in call_instruction:
            position = []
            if call_instru in branch_all_instructions:
                pos = branch_all_instructions.index(call_instru)
                slice_instruction = branch_all_instructions[pos:]
                for store_instru in log_instruction:
                    position = []
                    if store_instru in slice_instruction:
                        valid_branch_instructions.append(branch_instruction)
                        valid_branch_index.append(branch_edge[i])
                        break
                break
    return valid_branch_instructions, valid_branch_index

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_heatmap(tensor_1x512):
    """
    将 [1,512] 张量转换为 8×8 热力图

    步骤：
    1. 重塑为 [1,32,4,4]
    2. 应用平均池化
    3. 归一化处理
    4. 转换为热力图显示
    """
    # 步骤1: 重塑为 [1,32,4,4]
    reshaped_tensor = tensor_1x512.view(1, 16, 8, 4)

    # 步骤2: 应用平均池化，将每个 4×4 区域池化为 1×1
    # 使用 AdaptiveAvgPool2d 将 [1,32,4,4] 池化为 [1,32,1,1]
    adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
    pooled_tensor = adaptive_pool(reshaped_tensor)  # 形状: [1,32,1,1]

    # 重塑为 [8,8] 的热力图矩阵
    heatmap_matrix = pooled_tensor.view(4, 4)

    # 步骤3: 归一化到 [0,1] 范围
    heatmap_normalized = (heatmap_matrix - heatmap_matrix.min()) / (heatmap_matrix.max() - heatmap_matrix.min())

    return heatmap_normalized.detach().numpy()


def plot_heatmap(heatmap_data, title="Feature Heatmap", cmap="viridis", figsize=(8, 6)):
    """
    以美观的方式绘制热力图
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    im = ax.imshow(heatmap_data, cmap=cmap, interpolation='nearest')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Feature Intensity', rotation=-90, va="bottom")

    # 设置坐标轴
    ax.set_xticks(np.arange(heatmap_data.shape[1]))
    ax.set_yticks(np.arange(heatmap_data.shape[0]))
    ax.set_xticklabels(np.arange(1, heatmap_data.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, heatmap_data.shape[0] + 1))

    # 在每个单元格中显示数值
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                    ha="center", va="center", color="w", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')

    plt.tight_layout()
    return fig, ax


import seaborn as sns


def enhanced_heatmap_visualization(tensor_1x512, cmap="RdYlBu_r"):
    """
    增强版热力图可视化，使用 seaborn 获得更专业的外观
    """
    # 获取热力图数据
    heatmap_data = tensor_to_heatmap(tensor_1x512)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 使用 seaborn 绘制热力图
    ax = sns.heatmap(heatmap_data,
                     annot=True,  # 显示数值
                     fmt=".2f",  # 数值格式
                     cmap=cmap,  # 颜色映射
                     linewidths=0.5,  # 单元格边界线
                     linecolor='white',  # 边界线颜色
                     cbar_kws={'shrink': 0.8, 'label': 'Feature Intensity'})

    # 美化图形
    ax.set_title('8×8 Feature Distribution Heatmap\n(From 512D Tensor)',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Spatial Dimension X', fontsize=12)
    ax.set_ylabel('Spatial Dimension Y', fontsize=12)

    # 设置坐标轴标签
    ax.set_xticklabels([f'X{i + 1}' for i in range(8)])
    ax.set_yticklabels([f'Y{i + 1}' for i in range(8)])

    plt.tight_layout()
    return plt.gcf(), ax