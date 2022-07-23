#!/bin/bash

dataroot=$(dirname "$PWD")/data
echo $dataroot

path=$PWD
echo $path

python inference.py --mode="test" \
	--tag_file=$path/data/tag.txt \
	--ner_pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
	--ner_save_model_path=$path/model/ner \
        --extractor_pretrain_model_path=$path/pretrain_model/chinese-roberta-base \
	--extractor_save_model_path=$path/model/intent \
	--kb_file=$dataroot/kg.json \
	--valid_file=$dataroot/valid.json \
	--test_file=$dataroot/test.json \
	--result_file=$path/result.json



