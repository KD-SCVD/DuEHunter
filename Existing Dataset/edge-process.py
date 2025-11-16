import json

input_file = 'RE/bytecode/RE_bytecode.json'
output_file = 'RE/bytecode/RE_bytecode.json'
f = open(input_file,'r')
CFG_ = json.load(f)
for j, contract in enumerate(CFG_):
    edge = contract['edge']
    instructions = contract['instruction']
    if not edge:
        edges = []
        node_num = len(instructions)
        for j in range(node_num-1):
            edges.append([j,j+1])
        contract['edge'] = edges
s = open(output_file,'w')
f = json.dump(CFG_,s)

