'''
增强SFT数据工程脚本
KnowledgeMatch-v2: 题目找语料
'''

import json
import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm



# ================= 配置区 =================
SFT_CORPUS_PATH = 'data/medical/finetune/train/train_zh_0.json' # sft原始数据194w
CEVAL_DIR = 'data/ceval'    # ceval题目地址
OUTPUT_PATH = 'data/finetune/km-train_zh_0.jsonl'
EMBED_MODEL_PATH = '/root/autodl-tmp/medical/MedicalGPT/models/text2vec_model'  # 向量模型
TOP_K = 100  # 每道考题去sft数据里找Top-K个最相关的对话

# ================= 加载模型 =================
print("1. 正在加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH)


# ================= 解析原文 =================
print("2. 正在载入 SFT 原始语料...")
corpus_json = []
corpus_texts = []

with open(SFT_CORPUS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # Alpaca 格式：
        full_text = data['instruction'] + " " + data.get('input', '') + " " + data.get('output', '')
        corpus_json.append(data)
        corpus_texts.append(full_text)

print(f"加载完成，总语料条数：{len(corpus_texts)}")


# ================= 词向量化 =================
print("3. 正在构建向量索引...")
# 分批编码
corpus_embeddings = embed_model.encode(
    corpus_texts, 
    batch_size=256, 
    show_progress_bar=True, 
    convert_to_numpy=True
)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings.astype('float32'))


print("4. 正在读取 C-Eval 考题...")
val_files = glob.glob(os.path.join(CEVAL_DIR, "*_val.parquet"))
all_questions = pd.concat([pd.read_parquet(f) for f in val_files])
queries = all_questions['question'].tolist()

print(f"5. 正在进行语义对齐筛选 (考题数: {len(queries)})...")
# 使用集合去重，因为不同的题目可能会捞到同一条 SFT 数据
selected_indices = set()

for query in tqdm(queries, desc="对齐进度"):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec.astype('float32'), k=TOP_K)
    for idx in I[0]:
        selected_indices.add(idx)

print(f"6. 正在导出筛选后的‘精益化’数据集...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for idx in selected_indices:
        f.write(json.dumps(corpus_json[idx], ensure_ascii=False) + '\n')

print(f"✨ 任务完成！")
print(f"原始数据: {len(corpus_texts)} 条")
print(f"精选数据: {len(selected_indices)} 条")
print(f"数据压缩率: {len(selected_indices)/len(corpus_texts)*100:.2f}%")
print(f"新数据集已保存至: {OUTPUT_PATH}")