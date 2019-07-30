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

train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, raw_test = prepare_dataset(constant.params["train_file"], constant.params["valid_file"], constant.params["test_file"], constant.params["batch_size"], constant.params["batch_size"])

print("###### Prepare the dataset ######")
emb_size = constant.params["embedding_size_word"]
hidden_size = constant.params["hidden_size"]
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
emb_list = constant.params["emb_list"]

mode = constant.params["mode"]
no_projection = constant.params["no_projection"]

cuda = constant.USE_CUDA
pad_idx = 0
use_crf = constant.params["use_crf"]

print("######    Start training   ######")
if constant.params["model"] == "TRFS":
    model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, label2id, id2label, emb_list=emb_list, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, mode=mode, no_projection=no_projection)
else:
    print("No model is selected")

print(model)
best_valid_f1, best_valid_loss, best_epoch = trainer.train(model, constant.params["task_name"], train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, raw_test)

summary_path = "{}/summary.txt".format(constant.params["model_dir"])
with open(summary_path, "w+") as summary_file:
    summary_file.write("Best valid f1:{:3.5f} Best valid loss:{:3.5} at epoch:{:d}".format(best_valid_f1, best_valid_loss, best_epoch))