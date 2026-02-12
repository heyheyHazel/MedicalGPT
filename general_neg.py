'''
PPO数据工程：为distill_r1_110k通用推理数据生成具有推理功能的负样本，调用本地7B模型生成普通回答作为负样本。
输入数据格式（JSONL）：
{"input": "问题文本", "reasoning_content": "专家思考过程", "content": "专家回答"}
输出数据格式（JSONL）：
{"system": "你是一个严谨且有深度思考能力的AI助手。", "history": [], "question": "问题文本", "response_chosen": "<thought>\n专家思考过程\n</thought>\n专家回答", "response_rejected": "本地7B模型生成的普通回答"}
'''

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import random
import hashlib


# ====== 配置区 ======
INPUT_PATH = "data/general/distill_r1_110k.jsonl"
OUTPUT_PATH = "data/preference/general_logic_dpo_v1.jsonl"
MODEL_PATH = "models/base/medical-qwen-7b-sft-km-v3"


MAX_SAMPLES = 800   # 只采样800条
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RESUME = os.getenv("RESUME", "1").lower() in {"1", "true", "yes"}

# 加载本地模型（用于生成 Rejected 样本）
print("正在加载本地 SFT 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

def get_local_rejected(question):
    """让本地 7B 模型直接生成一个普通回答作为负样本"""
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _resume_keys_from_output(path: str):
    """从已生成的输出 JSONL 中提取断点 key（question 的 hash）。"""
    keys = set()
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                q = rec.get("question")
                if not q:
                    continue
                q = str(q)
                keys.add(hashlib.sha1(q.encode("utf-8")).hexdigest())
            except Exception:
                bad_lines += 1
                continue
    return keys, bad_lines

def main():
    # 2. 读取原始数据集并采样
    all_data = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(RANDOM_SEED)
    sampled_data = random.sample(all_data, min(len(all_data), MAX_SAMPLES))

    done_keys = set()
    bad_output_lines = 0
    if RESUME and os.path.exists(OUTPUT_PATH):
        try:
            done_keys, bad_output_lines = _resume_keys_from_output(OUTPUT_PATH)
        except Exception:
            done_keys, bad_output_lines = set(), 0
    
    # 3. 转换并生成
    print(
        f"开始转换格式并生成负样本 (共 {len(sampled_data)} 条)..."
        f" 断点续跑已完成 {len(done_keys)} 条 (bad_output_lines={bad_output_lines})"
    )
    mode = 'a' if RESUME else 'w'
    processed = 0
    skipped = 0
    bad_input = 0
    with open(OUTPUT_PATH, mode, encoding='utf-8') as f_out:
        for item in tqdm(sampled_data):
            # 获取原始字段
            question = item.get("input") or item.get("question")
            if not question:
                bad_input += 1
                continue
            if done_keys:
                q_key = hashlib.sha1(str(question).encode("utf-8")).hexdigest()
                if q_key in done_keys:
                    skipped += 1
                    continue
            expert_thought = item.get("reasoning_content", "")
            expert_answer = item.get("content", "")
            
            # --- 核心缝合：构造符合你要求的 Chosen 格式 ---
            response_chosen = f"<thought>\n{expert_thought}\n</thought>\n{expert_answer}"
            
            # --- 生成负样本：调用本地 7B 模型 ---
            response_rejected = get_local_rejected(question)
            
            # 4. 构造符合你脚本格式的最终 JSON
            final_record = {
                "system": "你是一个严谨且有深度思考能力的AI助手。",
                "history": [],
                "question": question,
                "response_chosen": response_chosen,
                "response_rejected": response_rejected
            }
            
            f_out.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            processed += 1

    print(f"完成：新增 {processed} 条，跳过 {skipped} 条，坏输入 {bad_input} 条。")

if __name__ == "__main__":
    main()
