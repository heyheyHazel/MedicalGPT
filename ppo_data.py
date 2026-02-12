'''
只保留训练需要的字段，剔除了 metadata 字段，以确保数据的干净和一致性，方便后续PPO训练使用。
'''

import json
import os


input_file = "data/preference/preference_v1.jsonl"
output_file = "data/preference/preference_v1_clean.jsonl"


# 训练需要的 Key 列表
REQUIRED_KEYS = ["system", "history", "question", "response_chosen", "response_rejected"]

print(f"正在清理数据并对齐 Schema...")
with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        try:
            data = json.loads(line)
            # 仅保留训练需要的字段，彻底剔除 metadata 等不必要的字段
            clean_data = {k: data[k] for k in REQUIRED_KEYS if k in data}
            f_out.write(json.dumps(clean_data, ensure_ascii=False) + '\n')
        except:
            continue

print(f"处理完成！请使用 {output_file} 进行训练。")
