import json

from fontTools.ttLib.tables.ttProgram import instructions

# 将data_RE_emb_asm2vecRE_d128.json中的instruction_label替换为RE_processed.json中的instruction
with open('TO/data_TO_emb_asm2vecRE_d128.json', 'r') as f :
    content_1 = json.load(f)
    f.close()
with open('TO/TO_processed.json', 'r') as f:
    content_2 = json.load(f)
    f.close()

# 从content_2中筛选出包含content_1的条目
filtered_content_2 = []
content_1_names = {item['contract_name']: idx for idx, item in enumerate(content_1)}
for item in content_2:
    if item.get('contract_name') in content_1_names:
        filtered_content_2.append(item)

# 确保filtered_content_2与content_1顺序一致
filtered_content_2.sort(key=lambda x: content_1_names[x['contract_name']])

# 执行原有的替换操作
for idx, content in enumerate(content_1):
    instruction = filtered_content_2[idx]
    content['instruction'] = instruction['instruction']
    del content['instruction_label']

with open('TO/TO_fusion_data.json', 'w') as f:
    json.dump(content_1, f)
    f.close()