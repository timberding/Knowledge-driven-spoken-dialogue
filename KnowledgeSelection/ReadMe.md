# KnowledgeSelection
Knowledge Selection Module: Identifying entities from dialogue history ,and extracting attributes of entities that involved in current utterance; Selecting related triples from knowledge base files according to the entites and attributes.

## Environment
``` shell
$ pip install -r requirement.txt
```

## Pretrained Model
Download pretrained model from huggingface. We use chinese-roberta-base in this example.

## Run

``` shell
$ sh run.sh
```
script run.sh contains three part: train the ner model, train the intention recognition model, and do inference on these two models.
``` shell
#First we train the ner model.
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

#Next, we train the intent extraction model.
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

#Finally, we do inference on these two models.
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
```

## Inference
``` shell
$ sh inference.sh
```
