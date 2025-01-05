#!/bin/bash

python main.py \
    --run_name topksae_gpt2_layer8_tiny_shakespeare \
    --model_name_or_path openai-community/gpt2 \
    --dataset_name_or_path HuggingFaceFW/fineweb-edu \
    --shard_size 10_000 \
    --acts_size 768 \
    --expansion_factor 4 \
    --batch_size 2048 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --model_layer_index 8 \
    --device_id 0 \
    --l1_coef 1e-2 \
    --mode topk \
    --log_interval 100