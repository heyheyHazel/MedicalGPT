'''
评测C-Eval医学相关的验证集，计算模型在每个科目上的准确率的简易脚本。
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import glob
import os

# -------------------- 配置地址 --------------------
MODEL_PATH = "models/base/medical-qwen-7b-sft-km-v3"
DATA_DIR = "./data/ceval"



# -------------------- 加载模型 --------------------
print(f"正在加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
model.eval()



# -------------------- 模型推理 --------------------
def get_model_answer(question, A, B, C, D):
    """通过计算 A/B/C/D 四个 Token 的 Logits 概率来选择答案"""
    # 构建 Prompt
    prompt = f"<|im_start|>system\n你是一个专业的医生。<|im_end|>\n<|im_start|>user\n问题：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请直接给出正确答案的字母。<|im_end|>\n<|im_start|>assistant\n答案是："
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取最后一个预测位置的 Logits
        logits = outputs.logits[0, -1, :]
        
        # 提取 A, B, C, D 对应的 Token ID
        # 针对 Qwen 这种分词器，直接取字符的 ID 即可
        choices = ['A', 'B', 'C', 'D']
        choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
        
        # 提取这四个 ID 的得分并取最大值
        choice_logits = logits[choice_ids]
        best_choice_idx = torch.argmax(choice_logits).item()
        
        return choices[best_choice_idx]


# -------------------- 模型评测 --------------------
def start_eval():
    # 扫描所有医学相关的验证集
    files = glob.glob(os.path.join(DATA_DIR, "*val.parquet"))
    if not files:
        print("未在目录下找到 val.parquet 文件，请检查路径")
        return

    final_scores = {}

    for file_path in files:
        subject = os.path.basename(file_path).replace("-val.parquet", "")
        print(f"\n正在评测科目: {subject}")
        
        df = pd.read_parquet(file_path)
        correct = 0
        total = len(df)
        
        for i, row in df.iterrows():
            pred = get_model_answer(row['question'], row['A'], row['B'], row['C'], row['D'])
            actual = row['answer']
            
            if pred == actual:
                correct += 1
            
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{total} | 当前准度: {correct/(i+1)*100:.2f}%")
        
        score = (correct / total) * 100
        final_scores[subject] = score
        print(f"{subject} 评测完成! 最终得分: {score:.2f}%")

    print("\n" + "="*40)
    print(f"{'科目 (Subject)':<25} | {'得分 (Accuracy)':<15}")
    print("-" * 40)
    for sub, acc in final_scores.items():
        print(f"{sub:<25} | {acc:>14.2f}%")
    print("="*40)

if __name__ == "__main__":
    start_eval()