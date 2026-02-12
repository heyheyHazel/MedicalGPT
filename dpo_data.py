'''
生成DPO适配的数据集格式。
将原始的 JSON 数据进行补全，
确保每条数据都包含 "system"、"history"、"question"、"response_chosen" 和 "response_rejected" 这几个字段。
它会从原始数据中提取这些字段，如果某个字段缺失，就用默认值（空字符串或空列表）来填充。
最后，它会将补全后的数据保存为 JSONL 格式，方便后续使用。
'''

import json

input_path = 'data/medical/reward/train/train.json' # 原始数据路径
output_path = 'data/reward/train.jsonl' # 输出路径

fixed_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line)
            # 补齐所有缺少的抽屉
            fixed_item = {
                "system": item.get("system", ""), # 如果没有就设为空字符串
                "history": item.get("history", []), # 如果没有就设为空列表
                "question": item.get("question") or item.get("prompt"), # 映射你的问题
                "response_chosen": item.get("response_chosen") or item.get("chosen"),
                "response_rejected": item.get("response_rejected") or item.get("rejected")
            }
            fixed_data.append(fixed_item)
        except: continue

with open(output_path, 'w', encoding='utf-8') as f:
    for item in fixed_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"数据补全完成！已经保存在：{output_path}")
