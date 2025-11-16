import json
import logging
import itertools
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction
from AVP import *

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
def cfg_construct(blocks, graph):
    for edge in graph:
        source = edge[0]
        target = edge[1]
        blocks[source].add_callee(blocks[target])
    return blocks
def function_construct(instructions, edge):
    functions = []
    # 根据图的连接关系，构建function
    for i, instruction in enumerate(instructions):  # all instructions within a block
        block = BasicBlock()
        for insr in instruction:  # every instruction are included in BasicBlock
            block.add_instruction(parse_instruction(insr))  # adding an instruction in  BasicBlock
        functions.append(Function(block))
    functions = cfg_construct(functions, edge)
    return functions




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



def vulner_score(instructions: object, edges: object):
    """

    :rtype: critical_branches
    """
    vulner_nodes = score_main(instructions, edges)  # AVP 判别
    vulner_instru = []
    if len(vulner_nodes) ==0 :
        cri_branch = []
    else:
        cri_branch = list(Condition_Gate(vulner_nodes, instructions, edges))
    return cri_branch, vulner_nodes


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
def block_construct(instructions):
    Functions = []
    # 根据图的连接关系，构建function
    for instruction in instructions:  # all instructions within a block
        block = BasicBlock()
        for insr in instruction:  # every instruction are included in BasicBlock
            block.add_instruction(parse_instruction(insr))  # adding an instruction in  BasicBlock
        Functions.append(Function(block))
    for i in range(len(Functions)-1):
        Functions[i].add_callee(Functions[i+1])

    return Functions


def branch_feature_extraction(instructions: object, edges: object, embedding_model: object):

    cri_branch, vul_branch = vulner_score(instructions, edges)
    if len(vul_branch) == 0:
        vul_branch_feature = None # 漏洞分支的每一个节点特征
        cri_branch_feature = None # 关键变量分支的每一个节点特征
    else:
        vul_branch_instru = vul_branch[0]
        cri_branch_instru = cri_branch[0]

        vul_branch_feature = []
        for vbInstruction in vul_branch_instru:
            functions = block_construct(vbInstruction)
            feature = list(map(lambda f: embedding_model.to_vec(f), functions))
            vul_branch_feature.append(feature)

        cri_branch_feature = []
        for cbInstruction in cri_branch_instru:
            functions = block_construct(cbInstruction)
            feature = list(map(lambda f: embedding_model.to_vec(f), functions))
            cri_branch_feature.append(feature)

    return vul_branch_feature, cri_branch_feature

# 找到所有的分叉节点
def find_forking_nodes(graph):
    return [node for node, children in graph.items() if len(children) > 1]

# 使用深度优先搜索来查找分支
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
def embedding_process(instructions,edge,embedding_model):

    functions = function_construct(instructions, edge)
    feature = list(map(lambda f: embedding_model.to_vec(f), functions))
    converted_list = [arr.tolist() for arr in feature]
    return converted_list

import random



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

