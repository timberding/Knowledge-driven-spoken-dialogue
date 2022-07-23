#Generation


## Model
+ We adapted pre-trained model GPT2[2] to our task, and fine-tuning GPT2 by task-related data. The train process reference to Kim S[1].

+ To change the pre-trained GPT2 model into "gpt2-Chinese-cluecorpussmall", competitors need to download the relevant model files from **huggingface** and store them in the `./model` path. Then, modified the value of "model_name_or_path" in file of `./baseline_ch/configs/generation/params.json`

## Data
+ **Dataset**：
  + **train data and validation data: `./data/train/data.json & ./data/val/data.json`**
  + **test data: `./data/test/data.json`**
  

### Run
+ **Execute script**：
  + **train script: `./train.sh`**：
  + **inference script: `./inference.sh`**：



### Environment

Environmental requirements are as follows：
```
apex==0.9.10dev
jsonschema==4.4.0
nltk==3.7
numpy==1.21.5
rouge==1.0.1
scikit_learn==1.1.1
tensorboardX==2.5.1
torch==1.7.0
tqdm==4.64.0
transformers==2.7.0
```

## Citation

Reference Note: On the Kim S' project[1] code, we modified the following points to adapt the task:

(1) The Knowledge-grounded Generation task-related code of Kim S' project[1] was used, and two other tasks were disassembled, we modified the code in baseline_ch/main.py.

(2) English pre-training GPT2 model was replaced by Chinese "gpt2-Chinese-cluecorpussmall" model, and we modified code in baseline_ch/main.py.

(3) Modified the data loading module: baseline_ ch/dataset. py, turn the training data to list of "dialog contex + knowledge graph + response".

```
[1]Kim S, Eric M, Gopalakrishnan K, et al. Beyond domain APIs: Task-oriented conversational modeling with unstructured knowledge access[J]. arXiv preprint arXiv:2006.03533, 2020.
[2]Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.
```
