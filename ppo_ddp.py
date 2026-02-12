# -*- coding: utf-8 -*-
"""
PPO分布式训练脚本
ppo_training.py ddp version
"""

import os
from tqdm import tqdm
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from datasets import load_dataset
from loguru import logger
import torch.distributed as dist
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)

from trl import (
    PPOConfig,
    PPOTrainer,
    ModelConfig,
    get_peft_config,
    AutoModelForCausalLMWithValueHead
)
from peft import LoraConfig
from template import get_conv_template
import torch
import bitsandbytes as bnb
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel as DDP


os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class PPOArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})


@dataclass
class ScriptArguments:
    """
    Script-level arguments that are not part of PPOConfig/ModelConfig.
    """
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the SFT model."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reward model."})
    output_dir: str = field(default="outputs-ppo", metadata={"help": "Output directory to save checkpoints."})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    total_episodes: Optional[int] = field(default=None, metadata={"help": "Total PPO steps (alias for PPOConfig.steps)."})
    response_length: Optional[int] = field(default=None, metadata={"help": "Max response length (unused)."})
    missing_eos_penalty: Optional[float] = field(default=None, metadata={"help": "Penalty for missing EOS (unused)."})
    eval_strategy: Optional[str] = field(default=None, metadata={"help": "Evaluation strategy (unused)."})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Evaluation steps (unused)."})
    num_train_epochs: Optional[int] = field(default=None, metadata={"help": "Train epochs (unused)."})
    per_device_train_batch_size: Optional[int] = field(default=None, metadata={"help": "Per-device batch size (unused)."})
    report_to: Optional[str] = field(default=None, metadata={"help": "Logging backend, e.g. wandb (maps to PPOConfig.log_with)."})


def main():
    accelerator = Accelerator()
    local_rank = accelerator.process_index
    is_main_process = accelerator.is_main_process
    parser = HfArgumentParser((PPOArguments, PPOConfig, ModelConfig, ScriptArguments))
    args, training_args, model_args, script_args = parser.parse_args_into_dataclasses()

    # 分布式训练设置：获取当前进程的 local_rank，并基于此构建 device_map 实现显存隔离
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}
    is_main_process = local_rank == 0
    # 定义 QLoRA 压缩配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 开启 4-bit 量化
        bnb_4bit_use_double_quant=True, # 二次量化，进一步省显存
        bnb_4bit_quant_type="nf4",  # 工业界标准数据类型
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Only log on main process
    if is_main_process:
        logger.info(f"Parse args: {args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")

    if script_args.sft_model_path is None or script_args.reward_model_path is None:
        raise ValueError("Both --sft_model_path and --reward_model_path are required.")

    if script_args.total_episodes is not None:
        training_args.steps = script_args.total_episodes

    if script_args.report_to and training_args.log_with is None:
        training_args.log_with = script_args.report_to

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.sft_model_path, 
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left" # 注意：PPO 训练通常需要左侧填充
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.debug(f"Tokenizer: {tokenizer}")

    # 加载模型
    # 价值模型用于估计当前策略的价值函数
    # value_model = AutoModelForSequenceClassification.from_pretrained(
    #     script_args.reward_model_path, 
    #     num_labels=1,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device_map, # 强制隔离
    #     quantization_config=bnb_config,
    #     trust_remote_code=model_args.trust_remote_code
    # )

    # 奖励模型用于评估生成文本的质量
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_path, 
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=model_args.trust_remote_code
    )
    value_model = reward_model 
    logger.info("已启用 Reward-Value 参数共享模式，节省 5.5GB 显存。")
    peft_config = LoraConfig(
        r=model_args.lora_rank if hasattr(model_args, 'lora_rank') else 8,
        lora_alpha=model_args.lora_alpha if hasattr(model_args, 'lora_alpha') else 16,
        lora_dropout=model_args.lora_dropout if hasattr(model_args, 'lora_dropout') else 0.05,
        target_modules=["q_proj", "v_proj"], # 显存保命，只练这两个
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 这里我们直接在加载模型时注入 LoRA 配置，这样就不需要后续再调用 `peft_model = get_peft_model(policy, peft_config)` 了。
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        load_in_4bit=True, # 开启 QLoRA
        device_map=device_map, # 强制双卡隔离
        peft_config=peft_config, # <--- 关键：在这里注入 LoRA
        torch_dtype=torch.bfloat16,
    )

    # 强制开启梯度检查点
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
        logger.info("已通过代码强制开启 Gradient Checkpointing。")

    # ref_policy 作为 PPO 的参考模型，通常用于计算 KL 散度以稳定训练。如果使用 PEFT，我们可以直接共享底座模型的显存，因此不需要单独加载一个完整的 ref_policy。
    if model_args.use_peft:
        ref_policy = None
        logger.info("PEFT 模式已启用，ref_model 设为 None 以共享底座显存。")
    else:
        # 只有全量微调（不推荐）才需要加载第二个模型
        ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.sft_model_path, 
        trust_remote_code=model_args.trust_remote_code,
        load_in_4bit=True,
        device_map=device_map
        )

    # Get datasets
    prompt_template = get_conv_template(args.template_name)
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_train_split
        )
        eval_samples = 100
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        dataset = load_dataset(
            'json',
            data_files=data_files,
        )
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))
    logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length
    # 修改预处理函数以适配新的数据格式，确保能够正确提取 human 的问题并构造 Prompt
    def preprocess_function(examples):
        new_examples = {"input_ids": []}
    # 获取对话列表
        for i, source in enumerate(examples['conversations']):
            try:
            # 1. 直接提取人类的问题 (通常是第一个 message)
            # 不管后面有没有 gpt 的回答，我们只管拿人问的那句
                question_text = ""
                for msg in source:
                    if msg["from"] == "human":
                        question_text = msg["value"]
                        break # 拿到第一句 human 即可
            
                if not question_text:
                    continue
                
            # 2. 这里的逻辑要符合你选的模板 (qwen)
            # 我们直接手动拼接一个简单的 Prompt，确保 100% 能被模型读懂
                full_prompt = f"<|im_start|>system\n你是一个专业的医疗AI助手。<|im_end|>\n<|im_start|>user\n{question_text}<|im_end|>\n<|im_start|>assistant\n"
            
            # 3. 分词
                tokenized = tokenizer(full_prompt, truncation=True, max_length=args.max_source_length)
                new_examples["input_ids"].append(tokenized["input_ids"])
            
            except Exception as e:
                continue
            
        return new_examples

    # def preprocess_function(examples):
    #     new_examples = {"input_ids": []}
    #     roles = ["human", "gpt"]

    #     def get_dialog(examples):
    #         system_prompts = examples.get("system_prompt", "")
    #         for i, source in enumerate(examples['conversations']):
    #             if len(source) < 2:
    #                 continue
    #             data_role = source[0].get("from", "")
    #             if data_role not in roles or data_role != roles[0]:
    #                 # Skip the first one if it is not from human
    #                 source = source[1:]
    #             if len(source) < 2:
    #                 continue
    #             messages = []
    #             for j, sentence in enumerate(source):
    #                 data_role = sentence.get("from", "")
    #                 if data_role not in roles:
    #                     logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
    #                     break
    #                 if data_role == roles[j % 2]:
    #                     messages.append(sentence["value"])
    #             if len(messages) < 2 or len(messages) % 2 != 0:
    #                 continue
    #             # Convert the list to pairs of elements
    #             history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
    #             system_prompt = system_prompts[i] if system_prompts else None
    #             yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)

    #     for dialog in get_dialog(examples):
    #         for i in range(len(dialog) // 2):
    #             source_txt = dialog[2 * i]
    #             tokenized_question = tokenizer(source_txt, padding=False)
    #             new_examples["input_ids"].append(tokenized_question["input_ids"])

    #     return new_examples

    # Preprocess the dataset
    if is_main_process:
        # num_proc = getattr(training_args, "dataset_num_proc", 1)
        num_proc = 1
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with accelerator.main_process_first():
            tokenized_train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1, # 强制改成1
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            train_dataset = tokenized_train_dataset.filter(
            lambda x: len(x['input_ids']) > 0
            )
            logger.debug(f"Train samples tokenized top3: {train_dataset[:3]}")
            print(f"DEBUG: 过滤后的训练集大小为: {len(train_dataset)}")
            if len(train_dataset) == 0:
                raise ValueError("错误：数据全部被过滤了！请检查 preprocess_function 逻辑。")

            # Preprocess the dataset for evaluation
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            tokenized_eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1, 
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            eval_dataset = tokenized_eval_dataset.filter(
                lambda x: len(x['input_ids']) > 0
            )
            logger.debug(f"Eval samples tokenized top3: {eval_dataset[:3]}")
            print(f"DEBUG: 过滤后的验证集大小为: {len(eval_dataset)}")
            if len(eval_dataset) == 0:
                raise ValueError("错误：验证数据全部被过滤了！请检查 preprocess_function 逻辑。")
    
    
    # PPOTrainer 传入model + tokenizer + dataset
    trainer = PPOTrainer(
        config=training_args,
        tokenizer=tokenizer,
        model=policy,
        ref_model=ref_policy,
        # reward_model=reward_model,
        # value_model=value_model,
        dataset=train_dataset,
        data_collator=data_collator, 
        # eval_dataset=eval_dataset,
        # peft_config=peft_config

    )

    # Training
    # if script_args.do_train:
    #     if is_main_process:
    #         logger.info("*** Train ***")
    #     trainer.train()

    #     # Only log on main process
    #     if is_main_process:
    #         trainer.save_model(script_args.output_dir)

    # 手写PPO训练循环逻辑
    if script_args.do_train:
        if is_main_process:
            logger.info("*** 启动 PPO 强化学习迭代 ***")

        # 1. 设定训练总步目
        max_steps = training_args.steps if training_args.steps is not None else 10000
        
        for step, batch in enumerate(tqdm(trainer.dataloader, desc="PPO 迭代中")):
            if step >= max_steps:
                break

            # --- A. 采样阶段 (Sampling) ---
            # 获取当前 batch 的问题 Token
            query_tensors = [q for q in batch["input_ids"]] 

            # Actor (运动员) 生成回答
            # 注意：generate 方法在 DDP 下会自动处理
            response_tensors = trainer.generate(
                query_tensors,
                return_prompt=False,
                max_new_tokens=script_args.response_length,
                temperature=0.7,
                do_sample=True
            )
            
            # 解码成文字，准备给判官打分
            batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            batch["query"] = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]

            # --- B. 打分阶段 (Scoring) ---
            # 判官 (Reward Model) 给 [问题 + 回答] 的组合打分
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            print(f"\n[DEBUG] 模型刚刚吐出的前两条回复是: ", flush=True)
            print(f"1. {batch['response'][0][:100]}...", flush=True)
            print(f"2. {batch['response'][1][:100]}...", flush=True)   
            torch.cuda.empty_cache()          
            # 使用之前加载好的 reward_model 进行前向推理
            # 注意：要把输入搬到对应的 GPU 上
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{local_rank}")
            with torch.no_grad():
                reward_outputs = reward_model(**inputs)
                # 提取分数 (通常取 logits 的第一个维度)
                rewards = [torch.tensor(score).to(policy.device) for score in reward_outputs.logits[:, 0]]

            # --- C. 更新阶段 (Optimization) ---
            # 执行最核心的 PPO 梯度更新
            # 这步会自动计算 KL 散度、优势函数 (Advantage) 并更新参数
            stats = trainer.step(query_tensors, response_tensors, rewards)

            # --- D. 日志记录 (Logging) ---
            # 只有主进程负责上报指标到 WandB
            if is_main_process:
                trainer.log_stats(stats, batch, rewards)
                if step % script_args.logging_steps == 0:
                    logger.info(f"Step {step}: Loss={stats['ppo/loss/total']:.4f}, Reward={torch.mean(torch.stack(rewards)):.4f}")

        # --- E. 训练结束，保存模型 ---
        if is_main_process:
            logger.info(f"训练完成，正在保存模型至 {script_args.output_dir}")
            # 提醒：PPO 的保存需要特殊处理，通常保存 policy 即可
            trainer.save_model(script_args.output_dir)
    
    trainer.generate_completions()


if __name__ == "__main__":
    main()
