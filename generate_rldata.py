'''
调用DeepSeek-R1模型生成医疗推理对齐数据（Chosen/Rejcted），并支持断点续跑和性能优化。
输入：原始病历提问（JSONL，每行一个记录，包含 "question" 字段）
输出：包含重构病历、专家回答（Chosen）和实习生回答（Rejected）的 JSONL 文件，格式如下：
{
    "system": "你是一个专业的医疗AI助手，必须通过深度逻辑推理辅助诊断。",
    "history": [],
    "question": "重构后的病历提问",
    "response_chosen": "<thought>模型的思维链</thought>\n专家风格的回答",
    "response_rejected": "实习生风格的回答",
    "metadata": {
        "original_question": "原始病历提问",
        "latency": 12.34
    }
}
'''
import json
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

from openai import APIStatusError, OpenAI, OpenAIError
from tqdm import tqdm
import os


# ====== 配置区 ======
# 提前配置环境变量：export OPENAI_API_KEY="sk-xxx"
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
model = os.getenv("OPENAI_MODEL", "deepseek-chat")
stream = os.getenv("OPENAI_STREAM", "1").lower() in {"1", "true", "yes"}
INPUT_PATH = "data/reward/train/train.jsonl"
OUTPUT_PATH = "data/preference/preference_v1.jsonl" # 输出清洗后的推理对齐数据

# 性能相关配置（可用环境变量覆盖）
# - RL_GEN_MODE: two_call(默认) | three_call(原始逻辑)
# - MAX_WORKERS: 并发线程数（适当调大可显著提速，但可能触发限流）
# - OPENAI_TIMEOUT: 请求超时秒数（避免长时间卡死）
# - RESUME: 1 表示根据输出文件行数跳过已完成的输入行（断点续跑）
RL_GEN_MODE = os.getenv("RL_GEN_MODE", "two_call").lower()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "180"))
RESUME = os.getenv("RESUME", "1").lower() in {"1", "true", "yes"}


# 初始化 deepseek/openai compatible 客户端
client = OpenAI(api_key=api_key, base_url=base_url, timeout=OPENAI_TIMEOUT)

# 重构病历的Prompt 格式化病人病史特征
PROMPT_RECONSTRUCT = "你是一名病历录入员。请将以下患者的乱序提问重构为标准的临床病历格式（包括：性别、年龄、主诉、病史简述、核心问题），要求用词专业、精简，若有未提及到的则标记为“未知”。"

# chosen/rejected 的 Prompt 模板, 专家 vs 实习生
PROMPT_EXPERT = (
    "你是一名经验丰富的临床主任。现在你正在诊室面对一位焦虑的患者。\n"
    "要求：\n"
    "1. 语气要亲切自然，像医生查房说话，多用‘您’，避免AI味。\n"
    "2. 直接给出详细的诊断内容和3-5条最关键的行动建议，总字数严控在500字以内。\n"
    "3. 不要在回复里写‘内部思维’、‘分析’等标题，直接对话。"
)

# 3. 实习生 (Rejected)：生硬死板、只会背书、忽略情感
PROMPT_INTERN = (
    "你是一名刚毕业、只懂背书的实习医生。你对患者缺少同理心，语气生硬。\n"
    "要求：\n"
    "1. 语气机械，只会罗列医学名词，不擅长安慰患者。\n"
    "2. 给出大量不分主次的检查建议，让患者感到更加困惑和负担。\n"
    "3. 回答要简短但敷衍，给人一种‘我在应付工作’的感觉。"
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # 容错：模型有时会在 JSON 前后加说明文字
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# ====== 重构病历prompt ======
def call_ds_chat(prompt, model="deepseek-chat", temperature=0.7):
    """通用调用函数"""
    for _ in range(3): # 简单重试机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except (APIStatusError, OpenAIError) as e:
            print(f"API请求失败，重试中... {e}")
            time.sleep(2)
    return None

def call_ds_reasoner(system_prompt, user_q, temperature=0.7):
    """调用 R1 推理模型，捕获思维链"""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_q}
                ],
                temperature=temperature
            )
            # 组合 <thought> 标签和最终回复
            thought = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
            return f"<thought>\n{thought}\n</thought>\n{content}"
        except (APIStatusError, OpenAIError) as e:
            print(f"推理模型调用失败: {e}")
            time.sleep(2)
    return None

def _resume_keys(path):
    if not os.path.exists(path): return set()
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                oq = rec.get("metadata", {}).get("original_question", "")
                keys.add(hashlib.sha1(oq.encode("utf-8")).hexdigest())
            except: continue
    return keys


def process_one_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_q = item.get("question") or item.get("prompt")
    if not raw_q: return None
    
    t0 = time.time()
    
    # 1. 重构病历 (Chat模型)
    reconstructed_q = call_ds_chat(f"{PROMPT_RECONSTRUCT}\n原始提问：{raw_q}")
    if not reconstructed_q: return None
    
    # 2. 生成 Chosen (R1 专家模式 - 深度思考)
    chosen_res = call_ds_reasoner(PROMPT_EXPERT, reconstructed_q, temperature=0.1)
    if not chosen_res: return None
    
    # 3. 生成 Rejected (R1 实习生模式 - 浅层思考)
    rejected_res = call_ds_reasoner(PROMPT_INTERN, reconstructed_q, temperature=1.2)
    if not rejected_res: return None
    
    return {
        "system": "你是一个专业的医疗AI助手，必须通过深度逻辑推理辅助诊断。",
        "history": [],
        "question": reconstructed_q,
        "response_chosen": chosen_res,
        "response_rejected": rejected_res,
        "metadata": {
            "original_question": raw_q,
            "latency": round(time.time() - t0, 2)
        }
    }


def main():
    if not api_key: raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    
    done_keys = _resume_keys(OUTPUT_PATH)
    items = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question") or item.get("prompt")
            if hashlib.sha1(q.encode("utf-8")).hexdigest() not in done_keys:
                items.append(item)

    print(f"🚀 开始炼金！待处理：{len(items)} 条，已跳过：{len(done_keys)} 条")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(process_one_item, it): it for it in items}
            
            for future in tqdm(as_completed(future_to_item), total=len(items), desc="数据蒸馏中"):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()


if __name__ == "__main__":
    main()
