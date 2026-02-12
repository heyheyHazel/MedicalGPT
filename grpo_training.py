# -*- coding: utf-8 -*-
"""
GRPO Training with a single GPU.
Updated reward functions.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F


# =================== 配置区 =====================
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 大模型裁判
JUDGE_MODEL_PATH = "/root/autodl-tmp/medical/MedicalGPT/models/base/Qwen2.5-3B-Instruct"
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH)

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_PATH, 
    device_map="auto", # 也可以写死 {"": 0}
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 语言顺滑度裁判
PPL_CHECKER_PATH = "/root/autodl-tmp/medical/MedicalGPT/models/base/Qwen2.5-0.5B"
ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_CHECKER_PATH)
ppl_model = AutoModelForCausalLM.from_pretrained(
    PPL_CHECKER_PATH, 
    device_map="cuda:0",
    torch_dtype=torch.bfloat16
).eval()

# 向量模型
VECTOR_MODEL_PATH = "/root/autodl-tmp/medical/MedicalGPT/models/text2vec_model"
semantic_judge = SentenceTransformer(VECTOR_MODEL_PATH).to("cuda")


# 参数配置类
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


# =================== 工具函数区 =====================
def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def accuracy_reward(completions, answer, **kwargs):
    """奖励函数: 根据模型输出与标准答案的一致性计算奖励分数"""
    # 提取模型输出的内容 
    # completions 是一个列表，包含了这一组（Group）生成的多个候选答案
    # completion[0]["content"] 获取第 i 个生成的对话正文
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # 遍历每一个生成的回答内容和对应的标准答案 (sol)
    for content, sol in zip(contents, answer):
        if '####' in sol:
            # 针对 GSM8K 数学题集的特殊处理逻辑
            # 解析标准答案：取 #### 后的数字并进行标准化解析（parse）
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            # 解析模型答案：先调用 extract_answer 抠出模型吐出的数字，再解析
            answer_parsed = parse(extract_answer(content))
        else:
            # 针对非 GSM8K（通常是 LaTeX 或通用数学/医疗题）的处理逻辑
            # 使用 LatexExtractionConfig 尝试从标准答案中提取 LaTeX 格式的数学表达式
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            # 解析模型答案：要求提供正确的 LaTeX 格式（无格式错误的运算符）
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,  # # 允许不规范的运算符
                            basic_latex=True,   # 基础LaTeX识别
                            equations=True, # 方程识别
                            boxed="all",    # 强制优先寻找 \boxed{} 里的答案
                            units=True,     # 识别单位（如 mg, ml）
                        ),
                        # 对于非 GSM8K 的题目，模型输出中可能包含多个数学表达式（如多个步骤的解答），我们优先考虑被 <answer> 标签包裹的内容，如果没有，则按照顺序提取第一个符合条件的表达式作为答案进行验证
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        # 判定阶段：调用 math-verify 库进行“语义对齐”判定
        try:
            # verify函数能判断答案是否一致，并返回一个布尔值（True/False）。我们将其转换为浮点数（1.0/0.0）作为奖励分数
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0
        # 打印调试信息，方便在后台看模型到底答对了没
        logger.debug(f"predict_answer: {content}, \nground_truth: {sol}, \n"
                     f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n")
        rewards.append(reward)
    # 汇总这一组（Group）所有样本的奖励值并返回
    logger.debug(f'accuracy rewards: {rewards}')
    return rewards


def format_reward(completions, **kwargs):
    """奖励函数: 保证格式正确 (CoT)."""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"\n[SAMPLE OUTPUT]: {completions[0][0]['content'][:200]}...", flush=True)
    
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    # 计算并打印奖励分数
    rewards = [2.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards


def semantic_reward(completions, answer, **kwargs) -> list[float]:
    """奖励函数: 模型回答和标准答案的语义相似度"""
    # 获取回答
    responses = [extract_answer(c[0]["content"]) for c in completions]
    
    # 向量化当前 Batch 的所有回答和标准答案
    pred_embeddings = semantic_judge.encode(responses, convert_to_tensor=True)
    gold_embeddings = semantic_judge.encode(answer, convert_to_tensor=True)
    
    # 计算余弦相似度
    cosine_scores = util.cos_sim(pred_embeddings, gold_embeddings)
    # 提取对角线上的分值（即对应的样本对）
    scores = torch.diagonal(cosine_scores).tolist()
    logger.debug(f'semantic rewards: {scores}')
    return [float(s) for s in scores]


def anti_repetition_reward(completions, **kwargs) -> list[float]:
    """奖励函数: 惩罚简单的重复句子/段落"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # 以句子为单位，判断是否存在过度重复
        sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", content) if s.strip()]
        if len(sentences) >= 3:
            unique_sentences = set(sentences)
            max_repeat_ratio = max(sentences.count(s) for s in unique_sentences) / len(sentences)
            # 若某个句子占比过高且重复出现，视为刷重复
            if max_repeat_ratio > 0.4 and max(sentences.count(s) for s in unique_sentences) >= 2:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    logger.debug(f'anti repetition rewards: {rewards}')
    return rewards


def llm_judge_reward(completions, answer, **kwargs) -> list[float]:
    """奖励函数: 使用大模型给回答质量打分"""
    # 提取模型回答
    responses = [c[0]["content"] for c in completions]
    prompts = kwargs.get("prompts", [""] * len(responses)) # 获取原始提问
    rewards = []
    
    for i, (res, gold) in enumerate(zip(responses, answer)):
        # 先查看原文
        if os.environ.get("LOCAL_RANK", "0") == "0" and i == 0:
            print(f"\n --- [JUDGE DEBUG] --- \n", flush=True)
            print(f"【😆 学生原话】: \n {res}...", flush=True)
        
        # 构造标准chat_template
        messages = [
            {"role": "system", "content": "你是一名资深的医学教授。请评价下方学生对医疗问题的回答。"},
            {"role": "user", "content": f"""
             请从回答专业性、逻辑严密性等维度评价学生的医疗回答。
             要求：先直接给出分数，再简要说明理由，打分量程：0-10分。
             打分要求：
                1. 如果回答逻辑错误或有误导，给 0-3 分。
                2. 如果回答基本正确但有瑕疵，给 4-7 分。
                3. 如果回答完美且逻辑清晰，给 8-10 分。
             务必以 [[分值]] 的格式给出总分。
             【参考标准答案】：{gold}\n
             【学生生成的回答】：{res}\n"""}
        ]
        input_ids = judge_tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(judge_model.device)
        
        with torch.no_grad():
            outputs = judge_model.generate(
                **input_ids,    # ** 解包字典为关键字参数
                max_new_tokens=256, 
                do_sample=False,  # 进一步强制确定性
                pad_token_id=judge_tokenizer.pad_token_id
            )
            # 只解码模型新吐出来的部分
            prompt_len = input_ids["input_ids"].shape[1]
            new_tokens = outputs[0][prompt_len:]
            judge_response = judge_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 正则提取分数
        def flexible_score_parser(text):
            """正则化提取分数, 包含优先级"""
            # 1. 优先级最高：找标准的 [[8]]
            match = re.search(r"\[\[(\d+\.?\d*)\]\]", text)
            if match: return float(match.group(1))
            
            # 2. 优先级中等：找类似 **分数：2分** 或 分数: 2
            match = re.search(r"分数[:：]\s*(\d+\.?\d*)", text)
            if match: return float(match.group(1))
            
            # 3. 优先级最低：找字符串里的第一个数字
            match = re.search(r"(\d+\.?\d*)", text)
            if match: return float(match.group(1))
            return 0.0

        score = flexible_score_parser(judge_response)
        rewards.append(score / 10.0)    # 标准化

        # 查看裁判原文
        if os.environ.get("LOCAL_RANK", "0") == "0" and i == 0:
            print(f"【🤨 裁判原话】:\n {judge_response}", flush=True)
    logger.debug(f'llm judge rewards: {rewards}')
    return rewards



def ppl_penalty_reward(completions, **kwargs) -> list[float]:
    """奖励函数: 语言顺滑度惩罚 (PPL Penalty)"""
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for text in responses:
        if len(text.strip()) < 5: # 太短的数据不测，直接给 0
            rewards.append(0.0)
            continue

        # 1. 分词
        inputs = ppl_tokenizer(text, return_tensors="pt").to(ppl_model.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            # 2. 获取模型的 Logits
            outputs = ppl_model(input_ids, labels=input_ids)
            # 这里的 loss 实际上就是平均 Negative Log-Likelihood (NLL)
            nll = outputs.loss.item() 
        
        # 3. 映射为惩罚项
        # 正常人类语言的 NLL 通常在 1.0 - 4.0 之间
        # 如果 nll > 5.0，说明模型开始乱说话了，我们开始扣分
        if nll > 5.0:
            # 惩罚公式：超过 5 的部分，每多 1 点扣 0.5 分
            penalty = -0.5 * (nll - 5.0)
            rewards.append(max(penalty, -2.0)) # 设置扣分下限，防止 Loss 爆炸
        else:
            rewards.append(0.0) # 说人话，不扣分
    logger.debug(f'ppl penalty rewards: {rewards}')
    return rewards


# 修改医学任务适配的system prompt
SYSTEM_PROMPT = (
    "你是一个专业的医疗AI助手。用户会向你咨询医学问题，请你通过深度思考后给出准确的解答。\n"
    "【核心要求】：\n"
    "1. 你的回答必须包含‘思维链推理’和‘最终答案’两部分。\n"
    "2. 推理过程请放在 <think> 和 </think> 标签之间，详细分析病理、逻辑和鉴别诊断。\n"
    "3. 最终答案请放在 <answer> 和 </answer> 标签之间，给出精炼、专业的医学建议。\n"
    "格式示例：<think> 在这里进行深入推理... </think><answer> 最终结论和建议... </answer>"
)

# 获取检查点路径，如果存在的话，以便在训练中断后恢复训练
def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

# 模型有成百上千层，到底该把 LoRA 的“补丁”贴在哪里
def find_all_linear_names(peft_model, int4=False, int8=False):
    """寻找所有可注入 LoRA 的线性层名称"""
    # 设置目标层的初始类型为 PyTorch 官方的标准线性层
    cls = torch.nn.Linear
    # 如果开启了量化（4位或8位），需要切换查找的目标类型
    if int4 or int8:
        # 导入 bitsandbytes 库，它是实现量化微调的底层核心
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    # 创建一个集合，用来存储发现的线性层短名称，set 具有自动去重功能
    lora_module_names = set()
    # 遍历模型中的所有子模块
    # name 是完整路径（如 model.layers.0.mlp.gate_proj），module 是对应的层对象
    for name, module in peft_model.named_modules():
        # 检查当前的这个层是不是我们刚才定义的线性层类型
        if isinstance(module, cls):
            # lm_head 是预测单词的最后一层，通常不建议加 LoRA，以保持输出稳定
            if 'lm_head' in name:
                continue
            # 有的模型把输出层叫 output_layer，同样跳过
            if 'output_layer' in name:
                continue
            # 提取层的短名称 
            # 将 'model.layers.0.self_attn.q_proj' 按 '.' 切分
            names = name.split('.')
            # 如果名字只有一级就取 names[0]，否则取最后一部分 names[-1]（如 'q_proj'
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    '''完整的GRPO训练流程, 包括DDP、多卡、量化、LoRA'''
    # 分布式训练初始化
    is_main_process = training_args.local_rank in [-1, 0]

    # 判断是否主进程，仅主进程输出日志
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # 加载tokenizer，优先使用用户指定的 tokenizer，如果没有则使用模型自带的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    # 配置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集，优先从本地目录加载，如果没有则从 HuggingFace hub 加载
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # 从本地目录加载数据集
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # 从 HuggingFace hub 加载数据集
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)
    # 如果用户指定了训练样本数量，则随机打乱数据集并选取前 N 个样本进行训练
    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # 准备数据集
    with training_args.main_process_first(desc="Dataset preparation"):
        def extract_to_grpo(example):
            user_question = example.get("question", "")
            gold_answer = example.get("response_chosen", "")
            return {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_question}
                ],
                'answer': gold_answer
            }

        dataset = dataset.map(
            extract_to_grpo,
            num_proc=script_args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="Processing dataset" if is_main_process else None,
        )

    # 划分数据集
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # 模型初始化参数设置
    # model_args找不到torch_dtype 直接用training_args的bf16和fp16来自动识别torch_dtype
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    logger.info(f"根据训练配置，自动识别 torch_dtype 为: {torch_dtype}")

    # 设置分布式训练配置
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # 检查 QLoRA 兼容性
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # 检查量化设置，4bit 和 8bit 不能同时开启
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # 设置量化配置
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )
    
    # 分布式训练和多卡训练设置
    num_gpus = torch.cuda.device_count()
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        model_kwargs["device_map"] = device_map
        # Ensure gradient_accumulation_steps is at least 1 after division
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // world_size, 1)
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # 配置LoRA
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # Fixed FP16 ValueError for quantized models
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # 初始化GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            llm_judge_reward,
            ppl_penalty_reward,
            format_reward,
            semantic_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # 检查是否存在之前的检查点，如果存在则从检查点恢复训练
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 训练完成后，只有主进程负责记录和保存训练结果，以避免多进程重复写入日志和模型文件
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # 保存模型
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
