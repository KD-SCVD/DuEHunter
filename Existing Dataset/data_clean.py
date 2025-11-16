import json
import re


def replace_instructions_with_data(instruction_lists):
    """
    替换指令并将操作数统一为"data"的函数

    参数:
        instruction_lists: 二维列表，每个子列表包含多条指令

    返回:
        处理后的新二维列表
    """

    # 编译正则表达式，匹配三种指令模式：
    patterns = {
        'PUSH': re.compile(r'^(PUSH)\d+(\s+.*)?$', re.IGNORECASE),
        'DUP': re.compile(r'^(DUP)\d+(\s+.*)?$', re.IGNORECASE),
        'SWAP': re.compile(r'^(SWAP)\d+(\s+.*)?$', re.IGNORECASE)
    }

    # 处理每个子列表
    new_lists = []
    for sublist in instruction_lists:
        new_sublist = []
        for instruction in sublist:
            if isinstance(instruction, str):
                matched = False
                # 检查所有模式
                for instr_type, pattern in patterns.items():
                    match = pattern.match(instruction)
                    if match:
                        # 统一替换为"data"作为操作数
                        new_instr = f"{instr_type} Data" if match.group(2) else instr_type
                        new_sublist.append(new_instr)
                        matched = True
                        break

                if not matched:
                    new_sublist.append(instruction)
            else:
                new_sublist.append(instruction)
        new_lists.append(new_sublist)

    return new_lists


if __name__ == '__main__':
    with open('demo_contract.json', 'r') as f:
        contents = json.load(f)
        f.close()
    for i, contract in enumerate(contents):
        instructions = contract['instruction']
        instruction_new = replace_instructions_with_data(instructions)
        contract['instruction'] = instruction_new
    with open('demo_contract.json', 'w') as f:
        json.dump(contents, f)
        f.close()
