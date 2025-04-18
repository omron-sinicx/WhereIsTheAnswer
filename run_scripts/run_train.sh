#!/bin/bash
CUSTOM_COMMAND=${@:4}
export DATASET_DIR="dataset"
deepspeed --num_gpus=$1 --master_port 29600 train.py \
    --output_dir $2 \
    --deepspeed $3 \
    --logging_steps 10 \
    --logging_strategy steps \
    --save_strategy steps \
    --do_train \
    --do_eval \
    --bf16 \
    --evaluation_strategy "steps" \
    --report_to wandb \
    ${CUSTOM_COMMAND}    

