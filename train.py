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
from eval.metrics import measure

import numpy as np

print("###### Initialize trainer  ######")
trainer = Trainer()

eval_batch_size = 10
train_loader, valid_loader, test_loader, word2id, id2word, char2id, id2char, label2id, id2label, raw_test, bpe_embs = prepare_dataset(constant.params["train_file"], constant.params["valid_file"], constant.params["test_file"], constant.params["batch_size"], eval_batch_size, bpe_lang_list=constant.params["bpe_lang_list"], bpe_vocab=constant.params["bpe_vocab"], bpe_emb_size=constant.params["bpe_emb_size"], bpe_cache=constant.params["bpe_cache"])

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
use_crf = constant.params["use_crf"]

cuda = constant.USE_CUDA
pad_idx = 0

print("######    Start training   ######")
model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, char2id, id2char, label2id, id2label, char_emb_size=char_emb_size, char_hidden_size=char_hidden_size, emb_list=emb_list, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, add_emb=add_emb, 
add_char_emb=add_char_emb, no_word_emb=no_word_emb, mode=mode, no_projection=no_projection, bpe_emb_size=bpe_emb_size, bpe_hidden_size=bpe_hidden_size, bpe_lang_list=bpe_lang_list, bpe_vocab=bpe_vocab, bpe_embs=bpe_embs)
print(model)

best_valid_f1, best_valid_loss, best_epoch = trainer.train(model, constant.params["task_name"], train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, raw_test)

summary_path = "{}/summary.txt".format(constant.params["model_dir"])
with open(summary_path, "w+") as summary_file:
    summary_file.write("Best valid f1:{:3.5f} Best valid loss:{:3.5} at epoch:{:d}".format(best_valid_f1, best_valid_loss, best_epoch))