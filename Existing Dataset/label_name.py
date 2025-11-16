import json

with open('final_delegatecall_name.txt', 'r') as f:
    names = f.read().split('\n')
    f.close()
with open('final_delegatecall_label.txt', 'r') as f:
    labels = f.read().split('\n')
    f.close()
    vulner = []
for i, name in enumerate(names):
    label = int(labels[i])
    if label == 1:
        vulner.append(name)
with open('DC_label_name.json','w') as f:
    json.dump(vulner, f)
    f.close()
