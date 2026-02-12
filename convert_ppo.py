import json
import os

# 修改为你真实的 PPO 训练数据路径
input_path = '/root/autodl-tmp/medical/MedicalGPT/data/preference/general_logic_dpo_v1.jsonl' 
output_path = '/root/autodl-tmp/medical/MedicalGPT/data/preference/general_logic_dpo_s.jsonl'

new_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line)
            # 无论原来是什么格式，统一封装进 conversations
            # 这里的 question 是你数据里的原始提问
            q = item.get("question") or item.get("prompt") or item.get("instruction")
            
            # PPO 训练只需要 question 即可，答案由模型自己生成
            sharegpt_item = {
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": ""} # PPO 会自动填充这里
                ]
            }
            new_data.append(sharegpt_item)
        except: continue

with open(output_path, 'w', encoding='utf-8') as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 转换完成！PPO 专用格式在: {output_path}")
