'''
将 Alpaca 格式的数据转换为 ShareGPT 格式。
Alpaca 格式示例：
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
ShareGPT 格式示例：
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."}
  ]
}
'''


import json

# 定义输入和输出文件路径
input_path = 'data/finetune/version3/curated_healthai_sft.jsonl'
output_path = 'data/finetune/version3/curated_healthai_sft-s.jsonl'

new_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        # 将 Alpaca 格式转换为 ShareGPT 格式
        # 把 instruction 和 input 拼在一起作为人类的提问
        human_text = item['instruction']
        if item.get('input'):
            human_text += "\n" + item['input']
            
        sharegpt_item = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": item['output']}
            ]
        }
        new_data.append(sharegpt_item)

with open(output_path, 'w', encoding='utf-8') as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"转换完成！新文件在: {output_path}")
