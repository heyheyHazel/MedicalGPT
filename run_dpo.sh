export PYTORCH_ALLOC_CONF=expandable_segments:True
fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 dpo_training.py \
    --model_name_or_path ./models/base/medical-qwen-7b-sft-km-v3 \
    --train_file_dir ./data/preference/train \
    --validation_file_dir ./data/preference/validation \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --use_peft True \
    --load_in_4bit True \
    --gradient_checkpointing False \
    --max_steps 100 \
    --max_source_length 256 \
    --max_target_length 256 \
    --output_dir ./models/rlhf/outputs-dpo-qwen-7b-v1 \
    --optim paged_adamw_32bit \
    --bf16 True \
    --fp16 False \
    --report_to wandb \
    --device_map None \
