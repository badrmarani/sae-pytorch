#!/bin/bash

python main.py \
    --model_name_or_path "path/to/pretrained/model" \
    --dataset_name_or_path "path/to/dataset" \
    --activations_path "path/to/activations" \
    --shard_size 1000 \
    --acts_size 500 \
    --dict_size 1000 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 10 \
    --model_layer_index 0 \
    --device_id 0 \
    --l1_coef 1.0 \
    --topk "soft" \
    --k 10