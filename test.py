from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import sys
import os
import logging

from tqdm import tqdm

from models.transformer_tagger import TransformerTagger
from trainers.trainer import Trainer
from utils import constant
from utils.data import prepare_dataset
from utils.training_common import compute_num_params, lr_decay_map
from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score

import numpy as np

print("###### Initialize trainer  ######")
trainer = Trainer()

train_loader, valid_loader, test_loader, word2id, id2word, char2id, id2char, label2id, id2label, raw_test, bpe_embs = prepare_dataset(constant.params["train_file"], constant.params["valid_file"], constant.params["test_file"], constant.params["batch_size"], constant.params["batch_size"], bpe_lang_list=constant.params["bpe_lang_list"], bpe_vocab=constant.params["bpe_vocab"], bpe_emb_size=constant.params["bpe_emb_size"], bpe_cache=constant.params["bpe_cache"])

print("###### Prepare the dataset ######")

emb_size = constant.params["embedding_size_word"]
char_emb_size = constant.params["embedding_size_char"]
hidden_size = constant.params["hidden_size"]
char_hidden_size = constant.params["embedding_size_char_per_word"]
num_layers = constant.params["num_layers"]
num_heads = constant.params["num_heads"]
dim_key = constant.params["dim_key"]
dim_value = constant.params["dim_value"]
filter_size = constant.params["filter_size"]
max_length = constant.params["max_length"]
input_dropout = constant.params["input_dropout"]
dropout = constant.params["dropout"]
attn_dropout = constant.params["attn_dropout"]
relu_dropout = constant.params["relu_dropout"]
add_emb = constant.params["add_emb"]
no_word_emb = constant.params["no_word_emb"]
add_char_emb = constant.params["add_char_emb"]

emb_list = constant.params["emb_list"]

bpe_emb_size = constant.params["bpe_emb_size"]
bpe_hidden_size = constant.params["bpe_hidden_size"]
bpe_lang_list = [] if constant.params["bpe_lang_list"] is None else constant.params["bpe_lang_list"]
bpe_emb_size = constant.params["bpe_emb_size"]
bpe_vocab = constant.params["bpe_vocab"]

mode = constant.params["mode"]
no_projection = constant.params["no_projection"]

cuda = constant.USE_CUDA
pad_idx = 0
use_crf = constant.params["use_crf"]

print("######    Start training   ######")
model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, char2id, id2char, label2id, id2label, char_emb_size=char_emb_size, char_hidden_size=char_hidden_size, emb_list=emb_list, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, add_emb=add_emb, add_char_emb=add_char_emb, no_word_emb=no_word_emb, mode=mode, no_projection=no_projection,
bpe_emb_size=bpe_emb_size, bpe_hidden_size=bpe_hidden_size, bpe_lang_list=bpe_lang_list, bpe_vocab=bpe_vocab, bpe_embs=bpe_embs)

if cuda:
    model = model.cuda()

PATH = "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"])
model.load_state_dict(torch.load(PATH))
all_predictions, all_true_labels, all_pairs, valid_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, valid_loader, word2id, id2word, label2id, id2label)
fb1 = f1_score(all_true_labels, all_predictions)
all_flatten_true_labels = []
all_flatten_predictions = []
for x in all_true_labels:
    all_flatten_true_labels += list(x)
for x in all_predictions:
    all_flatten_predictions += list(x)
accu = accuracy_score(all_flatten_true_labels, all_flatten_predictions)

# fb1 = metrics["fb1"] / (100)
print("valid fb1:", fb1)
print("valid accu:", accu)

with open("attention_scores.txt", "w+") as file_out:
    for i in range(len(attn_scores)):
        arr = attn_scores[i]
        for j in range(len(arr)):
            file_out.write(str(arr[j]))
        file_out.write("\n")

with open("bpe_attention_scores.txt", "w+") as file_out:
    for i in range(len(bpe_attn_scores)):
        arr = bpe_attn_scores[i]
        for j in range(len(arr)):
            file_out.write(str(arr[j]))
        file_out.write("\n")

all_predictions, all_true_labels, all_pairs, test_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, test_loader, word2id, id2word, label2id, id2label)
fb1 = f1_score(all_true_labels, all_predictions)
all_flatten_true_labels = []
all_flatten_predictions = []
for x in all_true_labels:
    all_flatten_true_labels += list(x)
for x in all_predictions:
    all_flatten_predictions += list(x)
accu = accuracy_score(all_flatten_true_labels, all_flatten_predictions)
print("test fb1:", fb1)
print("test accu:", accu)