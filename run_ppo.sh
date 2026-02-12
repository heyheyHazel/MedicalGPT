export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
fuser -k /dev/nvidia*
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 ppo_training.py \
    --sft_model_path ./models/base/medical-qwen-7b-sft-km-v3 \
    --reward_model_path ./models/rm/medical-qwen-7b-rm-merged \
    --template_name qwen \
    --torch_dtype bfloat16 \
    --train_file_dir ./data/rlhf/train \
    --validation_file_dir ./data/rlhf/validation \
    --max_source_length 512 \
    --response_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --batch_size 64 \
    --mini_batch_size 4 \
    --do_train \
    --total_episodes 30000 \
    --output_dir outputs-ppo-qwen-v1 \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 3 \
    --report_to wandb