# Step3. Inference
MODEL_DIR=./code/seq2edit-based-CGEC/exps/seq2edit_lang8
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"
PRETRAIN_WEIGHTS_DIR=./code/seq2edit-based-CGEC/plm/chinese-struct-bert-large/
VOCAB_PATH=./code/seq2edit-based-CGEC/data/output_vocabulary_chinese_char_hsk+lang8_5
VOCAB_FILE=./code/seq2edit-based-CGEC/vocab.txt
CUDA_DEVICE=0

## extract test and valid
TEST_INPUT_FILE=./data/input_data/test.json # 测试
TEST_OUTPUT_FILE=./code/seq2edit-based-CGEC/data/test.src
VALID_INPUT_FILE=./data/input_data/valid.json # 验证
VALID_OUTPUT_FILE=./code/seq2edit-based-CGEC/data/valid.src

if [ ! -f $INPUT_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/gector/test_data_reader.py --test_read_path $TEST_INPUT_FILE\
                                          --test_save_path $TEST_OUTPUT_FILE\
                                          --valid_read_path $VALID_INPUT_FILE\
                                          --valid_save_path $VALID_OUTPUT_FILE
fi



## inference test

if [ ! -f $TEST_OUTPUT_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $TEST_OUTPUT_FILE > $TEST_OUTPUT_FILE".char"
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi
OUTPUT_FILE=$RESULT_DIR"/test.output"

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ./code/seq2edit-based-CGEC/predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $TEST_OUTPUT_FILE".char"\
                  --vocab_file $VOCAB_FILE\
                  --output_file $OUTPUT_FILE --log



## inference valid

if [ ! -f $VALID_OUTPUT_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/tools/segment/segment_bert.py < $VALID_OUTPUT_FILE > $VALID_OUTPUT_FILE".char"
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi
OUTPUT_FILE=$RESULT_DIR"/valid.output"

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ./code/seq2edit-based-CGEC/predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $VALID_OUTPUT_FILE".char"\
                  --vocab_file $VOCAB_FILE\
                  --output_file $OUTPUT_FILE --log


## save test and valid

TEST_READ_FILE=$RESULT_DIR"/test.output.file"
VALID_READ_FILE=$RESULT_DIR"/valid.output.file"
TEST_SAVE_FILE=./data/input_data/test_fix.json
VALID_SAVE_FILE=./data/input_data/valid_fix.json


if [ ! -f $INPUT_FILE".char" ]; then
    python3 ./code/seq2edit-based-CGEC/gector/test_data_saver.py  --test_read_path $TEST_READ_FILE\
                                          --test_original_path $TEST_INPUT_FILE\
                                          --test_save_path $TEST_SAVE_FILE\
                                          --valid_read_path $VALID_READ_FILE\
                                          --valid_original_path $VALID_INPUT_FILE\
                                          --valid_save_path $VALID_SAVE_FILE
fi
