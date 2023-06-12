# PLM-ICD: Automatic ICD Coding with Pretrained Language Models
- Original Paper. [ClinicalNLP 2022 Paper](https://aclanthology.org/2022.clinicalnlp-1.2/)

## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`
* If the specific versions could not be found in your distribution, you could simple remove the version constraint. Our code should work with most versions.

## Dataset
Follow the instruction in the [dataset repository](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing) to obtain the dataset.

## How to run
### Pretrained LMs
Please download the pretrained LMs you want to use from the following link:
- [BioLM](https://github.com/facebookresearch/bio-lm): RoBERTa-PM models

Other models  (which we have not evaluated) include:
- [BioBERT](https://github.com/dmis-lab/biobert)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract): you can also set `--model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` when training the model, the script will download the checkpoint automatically.

### Training
1. `cd src`
2. Run the following command to train a model on MIMIC-IV-9 full.
```bash
accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES run_icd.py \
    --code_file mimic4_all_icd9.txt \
    --train_file $MIMIC4_ICD9_DIR/train_full.csv \
    --validation_file $MIMIC4_ICD9_DIR/dev_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 20 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-full-icd9 \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5
```
3. Run the following command to train a model on MIMIC-IV-10 full.
```bash
accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES run_icd.py \
    --code_file ./mimic4_all_icd10.txt \
    --train_file $MIMIC4_ICD10_DIR/train_full.csv \
    --validation_file $MIMIC4_ICD10_DIR/dev_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-full-icd10 \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5
```
4. Run the following command to train a model on MIMIC-IV-9 50.
```bash
accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES run_icd.py \
    --code_file ./top50_icd9_code_list.txt \
    --train_file $MIMIC4_ICD9_DIR/train_50.csv \
    --validation_file $MIMIC4_ICD9_DIR/dev_50.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 5 \
    --num_warmup_steps 3000 \
    --output_dir ../roberta-mimic4-50-icd9-shorter \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 5e-5
 ```
6. Run the following command to train a model on MIMIC-IV-10 50.
```bash
accelerate launch --gpu_ids $CUDA_VISIBLE_DEVICES run_icd.py \
    --code_file ./top50_icd10_code_list.txt \
    --train_file $MIMIC4_ICD10_DIR/train_50.csv \
    --validation_file $MIMIC4_ICD10_DIR/dev_50.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 5 \
    --num_warmup_steps 3000 \
    --output_dir ../roberta-mimic4-50-icd10-shorter \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 5e-5
```


### Notes

### Inference
1. `cd src`
2. Run the following commands to evaluate a model on the test set of `MIMIC-IV-ICD9-full`,  `MIMIC-IV-ICD10-full`, `MIMIC-IV-ICD9-50` and `MIMIC-IV-ICD10-50` (in that order).
```bash
python run_icd.py \
    --code_file mimic4_all_icd9.txt \
    --train_file $MIMIC4_ICD9_DIR/train_full.csv \
    --validation_file $MIMIC4_ICD9_DIR/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../roberta-mimic4-full-icd9 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8  \
    --num_train_epochs 0 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-full-icd9 \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5
    
python run_icd.py \
    --code_file ./mimic4_all_icd10.txt \
    --train_file $MIMIC4_ICD10_DIR/train_full.csv \
    --validation_file $MIMIC4_ICD10_DIR/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../roberta-mimic4-full-icd10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-full-icd10 \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5
    
python run_icd.py \
    --code_file ./top50_icd9_code_list.txt \
    --train_file $MIMIC4_ICD9_DIR/train_50.csv \
    --validation_file $MIMIC4_ICD9_DIR/test_50.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../roberta-mimic4-50-icd9-shorter/ \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8  \
    --num_train_epochs 0 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-50-icd9-shorter/ \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5

python run_icd.py \
    --code_file ./top50_icd10_code_list.txt \
    --train_file $MIMIC4_ICD10_DIR/train_50.csv \
    --validation_file $MIMIC4_ICD10_DIR/test_50.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../roberta-mimic4-50-icd10-shorter/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --num_warmup_steps 1000 \
    --output_dir ../roberta-mimic4-50-icd10-shorter/ \
    --model_type roberta \
    --model_mode laat \
    --learning_rate 7e-5
```
