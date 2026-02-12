CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 inference.py \
    --base_model models/base/medical-qwen-7b-dpo-v1 \
    --interactive