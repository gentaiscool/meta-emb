## Learning Multilingual Meta-Embeddings for Code-Switching Named Entity Recognition 
### Genta Indra Winata, Zhaojiang Lin, Pascale Fung

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the implementation of our paper accepted in RepL4NLP, ACL 2019. You can find our paper [here](https://www.aclweb.org/anthology/W19-4320). 

This code has been written using PyTorch. If you use any source codes or datasets included in this toolkit in your work, please cite the following paper.
```
@inproceedings{winata-etal-2019-learning,
    title = "Learning Multilingual Meta-Embeddings for Code-Switching Named Entity Recognition",
    author = "Winata, Genta Indra  and
      Lin, Zhaojiang  and
      Fung, Pascale",
    booktitle = "Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-4320",
    pages = "181--186",
}
```

## Abstract
In this paper, we propose Multilingual Meta-Embeddings (MME), an effective method to learn multilingual representations by leveraging monolingual pre-trained embeddings. MME learns to utilize information from these embeddings via a self-attention mechanism without explicit language identification. We evaluate the proposed embedding method on the code-switching English-Spanish Named Entity Recognition dataset in a multilingual and cross-lingual setting. The experimental results show that our proposed method achieves state-of-the-art performance on the multilingual setting, and it has the ability to generalize to an unseen language task.

## Data
English-Spanish Twitter Dataset in CoNLL format. Due to copyright and privacy issue, we cannot share the dataset in this repository, but you can contact the Shared Task committee or crawl the data by following the instructions in the [Shared Task Website](https://code-switching.github.io/2018/). You can reuse this code and apply our method in other datasets. 

Please check the format [here](sample.txt)

## Model Architecture
<img src="img/model.jpg" width=40%/>

## Setup
- Install PyTorch (Tested in PyTorch 1.0 and Python 3.6)
- Install library dependencies:
```console
pip install tqdm numpy
```
- Download pre-trained word embeddings.
In this paper, we were using English, Spanish, Catalan, and Portuguese FastText and an English Twitter GloVe. We generated word embeddings for all words to remove out-of-vocabulary and let the model learns how to choose and combine embeddings.

## Train
* ```--emb_list```: list all pre-trained word embeddings
* ```--use_crf```: add an CRF layer
* ```--model_dir```: define the location of the saved model
* ```--lr```: tune the learning rate
* ```--batch_size```: number of samples in each batch
* ```--mode```: ``concat`` or ``linear`` or ``attn_sum``
* ```--no_projection```: to remove the projection layer (especially for CONCAT)
* ```--early_stop```: to early stop

### How to run
#### CONCAT
```console
python train.py --emb_list embedding/all_vocab_en_es_crawl-300d-2M-subword.vec embedding/all_vocab_en_es_cc.es.300.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir concat_eng_spa_trfs_crf_lr0.1_lossmse_en_es --lr 0.1 --batch_size 32 --mode concat --no_projection
```

#### LINEAR
```console
python train.py --emb_list embedding/all_vocab_en_es_crawl-300d-2M-subword.vec embedding/all_vocab_en_es_cc.es.300.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir concat_eng_spa_trfs_crf_lr0.1_lossmse_en_es --lr 0.1 --batch_size 32 --mode linear
```

#### Meta-Embeddings
```console
python train.py --emb_list embedding/all_vocab_en_es_crawl-300d-2M-subword.vec embedding/all_vocab_en_es_cc.es.300.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir concat_eng_spa_trfs_crf_lr0.1_lossmse_en_es --lr 0.1 --batch_size 32 --mode attn_sum
```

## Test
To evaluate the F1 score, generate attention scores and save them into a file.
```console
python test.py --emb_list embedding/all_vocab_en_es_crawl-300d-2M-subword.vec embedding/all_vocab_en_es_cc.es.300.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir concat_eng_spa_trfs_crf_lr0.1_lossmse_en_es --lr 0.1 --batch_size 32 --mode attn_sum
```

## Attention
<img src="img/heatmap.jpg" width=70%>

## Bug Report
Feel free to create an issue or send email to giwinata@connect.ust.hk
