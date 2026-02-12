'''
增强SFT数据工程脚本
Knowledge Match v3版本：语料找题目
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
SFT_POOL_PATH = 'data/medical/finetune/train/train_zh_0.json'   # 原始sft语料194w条
CEVAL_DIR = 'data/ceval'    # 题目
EMBED_MODEL_PATH = '/root/autodl-tmp/medical/MedicalGPT/models/text2vec_model'  # 向量模型
TOP_K_BENCHMARK = 5  
FINAL_TAKE = 10000


# ================= 向量化 ==================
print("加载模型与探测器题目...")
model = SentenceTransformer(EMBED_MODEL_PATH)
val_files = glob.glob(os.path.join(CEVAL_DIR, "*_val.parquet"))
ceval_questions = pd.concat([pd.read_parquet(f) for f in val_files])['question'].tolist()
benchmark_embeddings = model.encode(ceval_questions, convert_to_numpy=True)


# ============ 载入语料并向量化 ================
print("载入SFT语料中...")
corpus_json = []
corpus_texts = []
with open(SFT_POOL_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="读文件中"):
        try:
            data = json.loads(line)
            # 把指令和输出拼在一起
            text = data['instruction'] + " " + data.get('output', '')
            corpus_json.append(data)
            corpus_texts.append(text)
        except: continue

print(f"开始向量化 {len(corpus_texts)} 条语料...")
sft_embeddings = model.encode(corpus_texts, batch_size=256, show_progress_bar=True)

# ============== 语义对齐 ==================
# 计算每条语料相对于 Benchmark 的得分
print("进行语义对齐打分...")
d = benchmark_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(benchmark_embeddings.astype('float32'))

# 给原始数据逐条打分
all_scores = []
batch_size = 1024
for i in range(0, len(sft_embeddings), batch_size):
    batch_vecs = sft_embeddings[i : i + batch_size].astype('float32')
    # D 存储的是 L2 距离，距离越小相似度越高
    D, _ = index.search(batch_vecs, k=TOP_K_BENCHMARK)
    # 计算 Top-5 相似题目的平均距离作为该语料的“分值”
    batch_avg_dists = np.mean(D, axis=1)
    all_scores.extend(batch_avg_dists)


# =============== 得分排序 =================
# 根据得分（距离）从小到大排序，取前 N 条
sorted_indices = np.argsort(all_scores)[:FINAL_TAKE]


# ================ 导出 =================
OUTPUT_PATH = 'data/finetune/version3/curated_healthai_sft.jsonl'
os.makedirs('data/finetune/version3', exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    for idx in sorted_indices:
        f.write(json.dumps(corpus_json[idx], ensure_ascii=False) + '\n')

print(f"完成！已保存到 {OUTPUT_PATH}")