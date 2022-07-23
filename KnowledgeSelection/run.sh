#!/bin/bash

dataroot=$(dirname "$PWD")/data
echo $dataroot

path=$PWD
echo $path


python NameEntityRecognition/preprocess.py \
	--input_path=$dataroot \
	--output_path=$path/data \

python NameEntityRecognition/main.py --mode="train" \
	--train_file=$path/data/ner_train.json \
	--dev_file=$path/data/ner_valid.json \
	--tag_file=$path/data/tag.txt \
	--pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
	--save_model_path=$path/model/ner \
	--epochs=10 \
	--validate_every=1

python IntentExtraction/preprocess.py \
        --input_path=$dataroot \
        --output_path=$path/data \

python IntentExtraction/main.py --mode="train" \
        --train_file=$path/data/extractor_valid.json \
        --dev_file=$path/data/extractor_valid.json \
        --kb_file=$dataroot/kg.json \
        --pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
        --save_model_path=$path/model/intent \
        --epochs=10 \
        --validate_every=1

python inference.py --mode="valid" \
	--tag_file=$path/data/tag.txt \
	--ner_pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
	--ner_save_model_path=$path/model/ner \
        --extractor_pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
	--extractor_save_model_path=$path/model/intent \
	--kb_file=$dataroot/kg.json \
	--valid_file=$dataroot/valid.json \
	--test_file=$dataroot/test.json \
	--result_file=$path/result.json



