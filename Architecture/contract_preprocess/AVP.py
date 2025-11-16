import json

call_instruction = ['CALL','CALLVALUE']
logits_instruction =['SUB','ADD', 'MUL', 'DIV', 'MOD']
log_instruction = ['SSTORE','RETURN','REVERT', 'MSTORE']
def score_process(result):
    '''
    评分为0标识没有记录指令
    标志位为1表示有记录指令
    标志位为2表示第一层邻居节点有逻辑指令
    标志位为3表示第二层邻居节点有逻辑指令
    评分为4表示在第一层邻居节点有逻辑指令并且在第二层邻居节点有call指令
    评分为5表示在第一层没有逻辑指令但在第二层邻居节点有逻辑指令，且在第三层邻居节点中有call指令
    '''
    score_list =[]
    for info in result:
        flag = 0 #评分标志位
        node_index = info[0]
        node_feature = info[1]
        adjacency_feature = info[2]
        score =[]
        # if node_index == 124:
        #     print()
        if len(adjacency_feature) <= 2:

            score.append({node_index:flag})
            score_list.append(score)
            continue
            _ = 0
        for i, ins in enumerate(node_feature):
            if ins in log_instruction:# 判断是否出现记录指令
                flag = 1
                _ = i
                if flag != 0:
                    break
        if flag == 0:
            score.append({node_index:flag})
            score_list.append(score)
            continue
        else:
            for i in range(_, len(node_feature)):
                if node_feature[i] in logits_instruction:
                    flag = 2
        if flag != 2:
            adjacency_first_1 = adjacency_feature[0][1]
            first_result = bool(set(adjacency_first_1) & set(logits_instruction))
            if first_result is True:
                flag = 2
            # 设置第二检验代码，用于判断第二层邻居节点是否出现记录逻辑指令，如果第二层邻居节点也没有出席那逻辑指令，则标志位为1
        if flag != 2:
            adjacency_first_2= adjacency_feature[1][1]
            if flag == 1:
                second_second_result = bool(set(adjacency_first_2) & set(logits_instruction))
                if second_second_result is True:
                    flag=3 # 如果出现逻辑指令，则标志位为3
        # 如果在两层邻居节点之后标志位仍然为1，则跳出循环，开始下一个节点
        if flag ==1:
            score.append({node_index:flag})
            score_list.append(score)
            continue
        '''
        标志位为2或者3，表示有记录指令和逻辑指令，但如果只是2或者3，则标志位为0
        如果在第一层邻居节点出现逻辑指令，则标志位为2，如果在第二层邻居节点出现逻辑指令，则标志位为3
        在第二层出现call指令，则标志位为4，如果在第三层出现call指令，则标志位为5
        '''

        if flag == 2:
            adjacency_second = adjacency_feature[1][1]
            second_result = bool(set(adjacency_second) & set(call_instruction))
            if second_result is True:
                flag = 4
                score.append({node_index: flag})
                score_list.append(score)
            else:
                adjacency_third = adjacency_feature[2][1]
                adjacency_ = adjacency_second+adjacency_third
                second_third_result = bool(set(adjacency_) & set(call_instruction))
                if second_third_result is True:
                    flag = 6
                    score.append({node_index: flag})
                    score_list.append(score)
            if flag == 2:
                flag = 0
                score.append({node_index:flag})
                score_list.append(score)
            continue
        elif flag ==3:
            adjacency_second = adjacency_feature[2][1]
            third_result = bool(set(adjacency_second) & set(call_instruction))
            if third_result is True:
                flag = 5
                score.append({node_index:flag})
                score_list.append(score)
                continue
        if flag ==2 or flag ==3 :
            flag = 0
            score.append({node_index:flag})
            score_list.append(score)
    final_score = []
    for j in range(len(score_list)):
        score_node = score_list[j][0][j]
        if score_node ==0:
            final_score.append(0)
        else:
            final_score.append(score_node)
    '''
    提取评分为4的索引，将该节点往上两层的索引提取出来并赋值为4
    提取评分为5的索引，将该节点往上一层的节点赋值为0，将该节点往上两层和三层的节点赋值为5
    提取评分为6的索引，将该节点往上一层的节点赋值为6，将该节点往上三层的节点赋值为6
    '''
    index_4 = [i for i, x in enumerate(final_score) if x == 4]
    index_5 = [i for i, x in enumerate(final_score) if x == 5]
    index_6 = [i for i, x in enumerate(final_score) if x == 6]
    return index_4, index_5, index_6
def build_adjacency_dict(adjacency_list):
    adjacency_dict = {}
    for source, target in adjacency_list:
        if target not in adjacency_dict:
            adjacency_dict[target] = set()
        adjacency_dict[target].add(source)
    return adjacency_dict
def find_neighbors_4(adjacency_dict,current_node):
    four_neighbors = []
    neighbors = set()
    # 第一层邻居节点
    if current_node in adjacency_dict:
        neighbors.update(adjacency_dict[current_node])
        first_neighbor = neighbors.pop()
        four_neighbors.append(first_neighbor)
        # 第二层节点
        if first_neighbor in adjacency_dict:
            neighbors.update(adjacency_dict[first_neighbor])
            second_neighbor = neighbors.pop()
            four_neighbors.append(second_neighbor)

    return four_neighbors
def find_neighbors_5(adjacency_dict,current_node):
    five_neighbors = []
    neighbors = set()
    # 第一层邻居节点
    if current_node in adjacency_dict:
        neighbors.update(adjacency_dict[current_node])
        first_neighbor = neighbors.pop()
        # five_neighbors.append(first_neighbor)
        # 第二层节点
        if first_neighbor in adjacency_dict:
            neighbors.update(adjacency_dict[first_neighbor])
            second_neighbor = neighbors.pop()
            five_neighbors.append(second_neighbor)
            # 第三层节点
            if second_neighbor in adjacency_dict:
                neighbors.update(adjacency_dict[second_neighbor])
                third_neighbor = neighbors.pop()
                five_neighbors.append(third_neighbor)
    return five_neighbors

def find_neighbors_6(adjacency_dict,current_node):
    six_neighbors = []
    neighbors = set()
    # 第一层邻居节点
    if current_node in adjacency_dict:
        neighbors.update(adjacency_dict[current_node])
        first_neighbor = neighbors.pop()
        six_neighbors.append(first_neighbor)
        # 第二层节点
        if first_neighbor in adjacency_dict:
            neighbors.update(adjacency_dict[first_neighbor])
            second_neighbor = neighbors.pop()
            if second_neighbor in adjacency_dict:
                neighbors.update((adjacency_dict[second_neighbor]))
                thrid_neighbor = neighbors.pop()
                six_neighbors.append(thrid_neighbor)
    return  six_neighbors

def find_three_layers_neighbors(adjacency_dict, node_features):
    result = []
    for node_index, node_feature in enumerate(node_features):# 遍历每个节点，并给出节点的索引
        neighbors = set()#预设值
        first_neighbor = []
        # 第一层邻居节点
        if node_index in adjacency_dict:
            neighbors.update(adjacency_dict[node_index])

            first_neighbor.append(neighbors.pop())
        # 第二层邻居节点
        try:
            # neighbors_second = list(neighbors)
            # neighbor_second = neighbors_second[-1]
            neighbor_second = first_neighbor[-1]
            if neighbor_second in adjacency_dict:
                neighbors.update(adjacency_dict[neighbor_second])

                first_neighbor.append(neighbors.pop())
        except IndexError:
            None
        # 第三层邻居节点
        try:
            # neighbors_third = list(neighbors)
            # neighbor_third = neighbors_third[-1]
            neighbor_third = first_neighbor[-1]
            if neighbor_third in adjacency_dict:
                neighbors.update(adjacency_dict[neighbor_third])

                first_neighbor.append(neighbors.pop())
        except IndexError:
            None
        neighbor_index = {node_index:first_neighbor}
        # 将邻居节点的索引和特征存入结果列表中
        # neighbors_data = [(index, node_features[index]) for index in neighbors]
        neighbors_data =[(index,node_features[index]) for index in first_neighbor]
        result.append((node_index, node_feature, neighbors_data))
    return result

def score_main(instructions, adjacency_list):
    # 构建邻接字典
    adjacency_dict = build_adjacency_dict(adjacency_list)

    # 寻找每个节点的三层邻居节点
    result = find_three_layers_neighbors(adjacency_dict, instructions)
    index_4, index_5, index_6 =score_process(result)
    expert_score_node = []
    four_score = []
    for i in index_4:
        four_score.append(i)
        neighbors_four = find_neighbors_4(adjacency_dict,i)
        four_score  = four_score+neighbors_four
    if four_score:
        four_score.reverse()
        expert_score_node.append(four_score)
    five_sore = []
    for j in index_5:
        five_sore.append(j)
        neighbors_five  = find_neighbors_5(adjacency_dict,j)
        five_sore = five_sore+neighbors_five
    if five_sore:
        five_sore.reverse()
        expert_score_node.append(five_sore)
    six_score = []
    for k in index_6:
        six_score.append(k)
        neighbors_six  = find_neighbors_6(adjacency_dict,k)
        six_score = six_score+neighbors_six
    if six_score:
        six_score.reverse()
        expert_score_node.append(six_score)
    return expert_score_node