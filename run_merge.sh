python merge_peft_adapter.py \
    --base_model models/base/medical-qwen-7b-sft-km-v3 \
    --lora_model models/rlhf/outputs-dpo-qwen-7b-v1 \
    --output_dir models/base/medical-qwen-7b-dpo-v1