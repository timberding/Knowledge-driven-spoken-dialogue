#ErrorCorrection 


## Model
+ **This baseline model directly follows the part of work in [MuCGEC](https://aclanthology.org/2022.naacl-main.227.pdf).**

+ **Chinese Grammatical Error Correction Baseline Model**：
  + **Seq2Edit Model`./code/seq2edit-based-CGEC`**: In [MuCGEC](https://aclanthology.org/2022.naacl-main.227.pdf), the author designed multiple action tags (such as replace, delete, insert, reorder, etc.), which treats the task of grammatical error correction as a sequence labeling task. 
  Meanwhile, the author of MuCGEC also modified [GECToR](https://github.com/grammarly/gector) to suit Chinese tasks.
    


## Data
+ **dataset**：
  + **train set :`./data/train_data/lang8+hsk/train.src & train.tgt` & validation set:`./data/valid_data/lang8+hsk/valid.src & valid.tgt`**：
  The file with suffix <b>'.src' </b> contains error sentences , when the file with suffix <b>'.tgt'</b> contains correct sentences. These two datasets use the lang8 and hsk data mentioned in [MuCGEC](https://github.com/HillZhang1999/MuCGEC).

  + **input data`./input_data/test.json & valid.json`**：Competition data need to be corrected.
  

### Run
+ **execute script**：
  + **train script`./train.sh`**：The model parameters can be adjusted or replaced
  + **inference script`./inference.sh`**：When executing the script, the corrected data <b>test_fix.json</b> and <b>valid_fix.json</b> will appear in `./input_data`. Finally, replace the original dataset with these new <b>.json</b> files.



### Environment


The operating environment is Python 3.8, in which the following packages should be installed:
```
allennlp==1.3.0
ltp==4.1.3.post1
OpenCC==1.1.2
pyarrow
python-Levenshtein==0.12.1
torch==1.7.1
transformers==4.0.1
numpy==1.21.1
```

## Citation

Thanks to [Yue Zhang](yzhang21@stu.suda.edu) and other authors for their excellent open-source work

1. We have deleted the irrelevant part of the code in [MuCGEC](https://github.com/HillZhang1999/MuCGEC)
2. We add the scripts `./code/seq2edit-based-CGEC/gector/test_data_reader.py & test_data_saver.py` to read and write the data

#### MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction (Accepted by NAACL2022 main conference) [[PDF]](https://aclanthology.org/2022.naacl-main.227.pdf)

```
@inproceedings{zhang-etal-2022-mucgec,
    title = "{M}u{CGEC}: a Multi-Reference Multi-Source Evaluation Dataset for {C}hinese Grammatical Error Correction",
    author = "Zhang, Yue  and
      Li, Zhenghua  and
      Bao, Zuyi  and
      Li, Jiacheng  and
      Zhang, Bo  and
      Li, Chen  and
      Huang, Fei  and
      Zhang, Min",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.227",
    pages = "3118--3130",
```
