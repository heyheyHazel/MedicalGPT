# 优化的GRPO QLoRA训练脚本 - 解决显存不足问题
# 针对32k长文本的配置
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 grpo_training.py \
    --model_name_or_path ./models/base/medical-qwen-7b-sft-km-v3 \
    --train_file_dir data/grpo \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 13 \
    --output_dir ./models/rlhf/outputs-grpo-qwen-7b-v1 \
    --bf16 True \
    --report_to wandb \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --use_vllm False \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 16384 \
    --max_completion_length 512

echo "训练完成!"
