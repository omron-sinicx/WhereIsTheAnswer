#!/bin/bash
CUSTOM_COMMAND=${@:4}
export DATASET_DIR="dataset"
deepspeed --num_gpus=$1 eval_a100.py \
    --output_dir $2 \
    --init_ckpt $3 \
    --deepspeed ./configs/ds_config_zero3_demo_a100.json \
    --logging_steps 10 \
    --logging_strategy steps \
    --save_strategy steps \
    --do_train \
    --do_eval \
    --bf16 \
    --evaluation_strategy "steps" \
    --report_to wandb \
    --use_flash_attn False \
    --eval_qa \
    ${CUSTOM_COMMAND}