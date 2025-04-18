#!/bin/bash
## Train D-AR model
bash run_scripts/run_train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_1.jsonl --max_steps 3000 --save_steps 3000 --noise_ratio 0.2
## Train AR model
bash run_scripts/run_train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_1.jsonl --max_steps 3000 --save_steps 3000 --noise_ratio 0.0
## Train attention dropout
bash run_scripts/run_train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_1.jsonl --max_steps 3000 --save_steps 3000 --noise_ratio 0.0 --dropout 0.2

