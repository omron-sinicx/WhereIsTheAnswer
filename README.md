# Where is the Answer? An Empirical Study of Positional Bias for Parametric Knowledge Extraction in Language Model (NAACL2025 Long Paper, Oral)

This repository contains the code and experiments from the paper "Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model."

## Authors

- **Kuniaki Saito** (OMRON SINIC X)  
  *kuniaki.saito@sinicx.com*  
  All experiments are conducted by OMRON SINIC X.
  
- **Chen-Yu Lee** (Google Cloud AI Research)  
- **Kihyuk Sohn** (Work done at Google Research)  
- **Yoshitaka Ushiku**

## Requirements

To run the code, you'll need the following dependencies and some packages listed in requiremenets.txt:

- PyTorch (version compatible with your setup)
- Transformers
- Deepspeed

## Setup
1. Download [Wiki2023 plus](https://huggingface.co/datasets/omron-sinicx/wiki2023_plus)
2. Download [Synthetic Language](https://huggingface.co/datasets/omron-sinicx/synthetic_language)
3. Make a directory, dataset and put files (e.g., film_qa_test.jsonl, film_qa_train.jsonl, film_qa_val.jsonl, film_doc_all.jsonl, film_doc_all.jsonl, film_insert_1.jsonl, film_insert_3.jsonl, film_insert_5.jsonl)  under ./dataset. 
4. Make a directory, model and download pre-trained weight, e.g., llama-2-7b-chat, under ./model directory. base.yaml's "init_ckpt" will specify the path to pre-trained model.

## Running the Code

To run the experiments, use the following script:

```bash
## Train model on wiki-film, with denoising auto-regressive loss. 
bash run_scripts/train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_1.jsonl --max_steps 3000 --save_steps 3000
## Train model on wiki-film, D_k = 3 
bash run_scripts/train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_3.jsonl --max_steps 3000 --save_steps 3000
## Train model on wiki-film, D_k = 5
bash run_scripts/train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --path_doc_train film_insert_5.jsonl --max_steps 3000 --save_steps 3000

## Train model synthetic language with denoising auto-regressive loss. 
bash run_scripts/train.sh 8 $PATH_TO_SAVE_MODEL ./configs/ds_config_zero3.json --max_steps 3000 --save_steps 3000 --train_config ./train_configs/synth_language.yaml
```

Evaluation. Please make sure that you have a tokenizer in the $PATH_TO_EVAL_MODEL.
```bash
## Evaluation on wiki-film.
bash run_scripts/run_eval.sh 1 $PATH_TO_SAVE_RESULT $PATH_TO_EVAL_MODEL

bash run_scripts/run_eval.sh 1 $PATH_TO_SAVE_RESULT $PATH_TO_EVAL_MODEL --train_config ./train_configs/synth_language.yaml

## Evaluation on synthetic language
```

### Experiment Configuration

Configurations for the experiments are defined in the `train_configs/` folder. You can modify them as needed before running the experiments.

## Contribution

If you would like to contribute to this project, please feel free to fork the repository and submit a pull request with your improvements.

## Citation

If you use the code or results in this repository, please cite our paper:

```bibtex
@article{saito2024answer,
  title={Where is the answer? investigating positional bias in language model knowledge extraction},
  author={Saito, Kuniaki and Sohn, Kihyuk and Lee, Chen-Yu and Ushiku, Yoshitaka},
  journal={arXiv preprint arXiv:2402.12170},
  year={2024}
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
