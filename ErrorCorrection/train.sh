# Step1. Data Preprocessing

## Download Structbert
if [ ! -f ./code/seq2edit-based-CGEC/plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./code/seq2edit-based-CGEC/plm/chinese-struct-bert-large/pytorch_model.bin
fi

## Tokenize
SRC_FILE=./data/train_data/lang8+hsk/train.src  # 每行一个病句
TGT_FILE=./data/train_data/lang8+hsk/train.tgt  # 每行一个正确句子，和病句一一对应
if [ ! -f $SRC_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $SRC_FILE > $SRC_FILE".char"
fi
if [ ! -f $TGT_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $TGT_FILE > $TGT_FILE".char"
fi


SRC_FILE_DEV=./data/valid_data/lang8+hsk/valid.src
TGT_FILE_DEV=./data/valid_data/lang8+hsk/valid.tgt
if [ ! -f $SRC_FILE_DEV".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $SRC_FILE_DEV > $SRC_FILE_DEV".char"
fi
if [ ! -f $TGT_FILE_DEV".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $TGT_FILE_DEV > $TGT_FILE_DEV".char"
fi


## Generate train label file
LABEL_FILE=./data/train_data/lang8+hsk/train.label
if [ ! -f $LABEL_FILE ]; then
    python3 ./code/seq2edit-based-CGEC/utils/preprocess_data.py -s $SRC_FILE".char" -t $TGT_FILE".char" -o $LABEL_FILE --worker_num 32
    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

## Generate valid label file
LABEL_FILE_DEV=./data/valid_data/lang8+hsk/dev.label
if [ ! -f $LABEL_FILE_DEV ]; then
    python3 ./code/seq2edit-based-CGEC/utils/preprocess_data.py -s $SRC_FILE_DEV".char" -t $TGT_FILE_DEV".char" -o $LABEL_FILE_DEV --worker_num 32
    shuf $LABEL_FILE_DEV > $LABEL_FILE_DEV".shuf"
fi

# Step2. Training
CUDA_DEVICE=0
SEED=1

DEV_SET=./data/valid_data/lang8+hsk/dev.label
MODEL_DIR=./code/seq2edit-based-CGEC/exps/seq2edit_lang8
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=./code/seq2edit-based-CGEC/plm/chinese-struct-bert-large/

VOCAB_PATH=./code/seq2edit-based-CGEC/data/output_vocabulary_chinese_char_hsk+lang8_5

## Freeze encoder (Cold Step)
COLD_LR=1e-3
COLD_BATCH_SIZE=128
COLD_MODEL_NAME=Best_Model_Stage_1
COLD_EPOCH=2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ./code/seq2edit-based-CGEC/train.py --tune_bert 0\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $COLD_MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $COLD_BATCH_SIZE\
                --n_epoch $COLD_EPOCH\
                --lr $COLD_LR\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --seed $SEED

## Unfreeze encoder
LR=1e-5
BATCH_SIZE=32
ACCUMULATION_SIZE=4
MODEL_NAME=Best_Model_Stage_2
EPOCH=20
PATIENCE=3

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ./code/seq2edit-based-CGEC/train.py --tune_bert 1\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "Temp_Model"\
                --seed $SEED

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
