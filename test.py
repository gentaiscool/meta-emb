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
from models.lstm_tagger import LstmTagger
from trainers.trainer import Trainer
from utils import constant
from utils.data import prepare_dataset
from utils.training_common import compute_num_params, lr_decay_map
from eval.metrics import measure

import numpy as np

print("###### Initialize trainer  ######")
trainer = Trainer()

train_loader, valid_loader, test_loader, word2id, id2word, char2id, id2char, label2id, id2label, raw_test, bpe_embs = prepare_dataset(constant.params["train_file"], constant.params["valid_file"], constant.params["test_file"], constant.params["batch_size"], constant.params["batch_size"], train_valid=constant.params["train_valid"], stemming_arabic=constant.params["stemming_arabic"], new_preprocess=constant.params["new_preprocess"], bpe_lang_list=constant.params["bpe_lang_list"], bpe_vocab=constant.params["bpe_vocab"], bpe_emb_size=constant.params["bpe_emb_size"], bpe_cache=constant.params["bpe_cache"])

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
add_lstm = constant.params["add_lstm"]
add_char_emb = constant.params["add_char_emb"]

emb_list = constant.params["emb_list"]

bpe_emb_size = constant.params["bpe_emb_size"]
bpe_hidden_size = constant.params["bpe_hidden_size"]
bpe_lang_list = [] if constant.params["bpe_lang_list"] is None else constant.params["bpe_lang_list"]
bpe_emb_size = constant.params["bpe_emb_size"]
bpe_vocab = constant.params["bpe_vocab"]

lstm_num_layers = constant.params["lstm_num_layers"]
lstm_dropout = constant.params["lstm_dropout"]
lstm_bidirec = constant.params["lstm_bidirec"]
mode = constant.params["mode"]
no_projection = constant.params["no_projection"]
proj_beta = constant.params["proj_beta"]
orthogonalize = constant.params["orthogonalize"]

cuda = constant.USE_CUDA
pad_idx = 0
use_crf = constant.params["use_crf"]

print("######    Start training   ######")
if constant.params["model"] == "TRFS":
    # model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, label2id, id2label, fasttext_en_emb_file=fasttext_en_emb_file, fasttext_es_emb_file=fasttext_es_emb_file, fasttext_pt_emb_file=fasttext_pt_emb_file, fasttext_it_emb_file=fasttext_it_emb_file, glove_en_emb_file=glove_en_emb_file, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, add_emb=add_emb, add_glove_emb=add_glove_emb, add_fasttext_pt_emb=add_fasttext_pt_emb, add_fasttext_it_emb=add_fasttext_it_emb)
    model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, char2id, id2char, label2id, id2label, char_emb_size=char_emb_size, char_hidden_size=char_hidden_size, emb_list=emb_list, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, add_emb=add_emb, add_lstm=add_lstm, add_char_emb=add_char_emb, lstm_num_layers=lstm_num_layers, lstm_dropout=lstm_dropout, lstm_bidirec=lstm_bidirec, mode=mode, no_projection=no_projection, orthogonalize=orthogonalize, proj_beta=proj_beta,
    bpe_emb_size=bpe_emb_size, bpe_hidden_size=bpe_hidden_size, bpe_lang_list=bpe_lang_list, bpe_vocab=bpe_vocab, bpe_embs=bpe_embs)

# elif constant.params["model"] == "LSTM":
#     model = LstmTagger(cuda=constant.USE_CUDA, params=constant.params, pad_idx=word2id["<pad>"], word2id, id2word, label2id, id2label)
else:
    print("No model is selected")

# print(model)
# best_valid_f1, best_valid_loss, best_epoch = trainer.train(model, constant.params["task_name"], train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, raw_test)

if cuda:
    model = model.cuda()

PATH = "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"])
model.load_state_dict(torch.load(PATH))
all_predictions, all_true_labels, all_pairs, valid_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, valid_loader, word2id, id2word, label2id, id2label)
metrics = measure(all_pairs)
fb1 = metrics["fb1"] / (100)
print("valid fb1:", fb1)

with open("attention_scores.txt", "w+") as file_out:
    for i in range(len(attn_scores)):
        arr = attn_scores[i]
        for j in range(len(arr)):
            # print(arr[j].shape)
            file_out.write(str(arr[j]))
        file_out.write("\n")

with open("bpe_attention_scores.txt", "w+") as file_out:
    for i in range(len(bpe_attn_scores)):
        arr = bpe_attn_scores[i]
        for j in range(len(arr)):
            # print(arr[j].shape)
            file_out.write(str(arr[j]))
        file_out.write("\n")

# print(attn_scores)

all_predictions, all_true_labels, all_pairs, test_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, test_loader, word2id, id2word, label2id, id2label)
metrics = measure(all_pairs)
fb1 = metrics["fb1"] / (100)
print("test fb1:", fb1)