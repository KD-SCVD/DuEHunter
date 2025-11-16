import  json
import os
import joblib
from tqdm import tqdm

from util import *

def contract_load(input_path):
    with open(input_file, 'r') as f:
        contents = json.load(f)
        f.close()
    return contents

def extract_vulner_branch(instruction, edge):
    vulner_branch = score_main(instruction, edge)
    return vulner_branch

def extract_critical_branch(vulner_nodes, edge):
    if len(vulner_nodes) ==0 :
        cri_branch = []
    else:
        cri_branch = list(Condition_Gate(vulner_nodes, instructions, edges))
    return cri_branch

if __name__ == '__main__':
    input_file = '../../Existing Dataset/RE/train_dataset.json'
    contracts = contract_load(input_file)
    for idx, contract in enumerate(tqdm(contracts)):
        instructions = contract['instruction']
        label = contract['label']
        edges = contract['edge']
        contract_name = contract['contract_name']
        node_features = contract['node_feature']

        if label == 1:
            # ============ extract vulnerability branch ================
            vulner_branch_nodes = extract_vulner_branch(instructions, edges) # [[],]
            for nodes in vulner_branch_nodes:
                print()
            # ============ extract critical branch  ===========
            critical_branch_nodes = extract_critical_branch(vulner_branch_nodes, edges) # [[],]

        else:
            vulner_branch_nodes = None
            critical_branch_nodes = None

        # ============ construct dictionary ===========
        contract['vulner_feature'] = vulner_branch_nodes
        contract['critical_feature'] = critical_branch_nodes

    with open('../../Existing Dataset/RE/RE_train_dataset.json', 'w') as f:
        json.dump(contracts, f)
        f.close()

