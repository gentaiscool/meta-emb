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
# from seqeval.metrics import accuracy_score
from sklearn.metrics import accuracy_score

import numpy as np

eval_task = constant.params["eval"]
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
metric = constant.params["metric"]

pred_file = constant.params["pred_file"]

cuda = constant.USE_CUDA
pad_idx = 0

print("###### Initialize trainer  ######")
trainer = Trainer()

eval_batch_size = 10
print("train_file:", constant.params["train_file"])
print("valid_file:", constant.params["valid_file"])
print("test_file:", constant.params["test_file"])
train_loader, valid_loader, test_loader, word2id, id2word, char2id, id2char, label2id, id2label, raw_test, bpe_embs = prepare_dataset(constant.params["train_file"], constant.params["valid_file"], constant.params["test_file"], constant.params["batch_size"], eval_batch_size, bpe_lang_list=constant.params["bpe_lang_list"], bpe_vocab=constant.params["bpe_vocab"], bpe_emb_size=constant.params["bpe_emb_size"], bpe_cache=constant.params["bpe_cache"], eval_type=constant.params["eval"], default_label=constant.params["default_label"])
print("###### Prepare the dataset ######")

metric = constant.params["metric"]

cuda = constant.USE_CUDA
pad_idx = 0
use_crf = constant.params["use_crf"]

print("######    Start training   ######")
model = TransformerTagger(emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, char2id, id2char, label2id, id2label, char_emb_size=char_emb_size, char_hidden_size=char_hidden_size, emb_list=emb_list, cuda=cuda, pad_idx=pad_idx, use_crf=use_crf, add_emb=add_emb, add_char_emb=add_char_emb, no_word_emb=no_word_emb, mode=mode, no_projection=no_projection, bpe_emb_size=bpe_emb_size, bpe_hidden_size=bpe_hidden_size, bpe_lang_list=bpe_lang_list, bpe_vocab=bpe_vocab, bpe_embs=bpe_embs)

if cuda:
    model = model.cuda()

def predict(model, test_loader, word2id, id2word, label2id, id2label, raw_test):
    """
    Predict the model on test dataset
    """
    model.eval()

    all_predictions = []
    all_pairs = []
    start_iter = 0

    pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
    for i, (data) in enumerate(pbar, start=start_iter):
        word_src, word_tgt, bpe_ids, char_src, src_lengths, src_raw = data

        if constant.USE_CUDA:
            word_src = word_src.cuda()
            word_tgt = word_tgt.cuda()
            char_src = char_src.cuda()
            if bpe_ids is not None:
                bpe_ids = bpe_ids.cuda()

        predictions, loss, _, _ = model.forward(word_src, word_tgt, bpe_ids, char_src, src_raw, src_lengths=src_lengths, print_loss=True)

        sample_id = 0
        for sample in predictions.cpu().numpy().tolist():
            token_id = 0
            # if classification
            if isinstance(sample, int):
                for token_id in range(len(word_src[sample_id])):
                    word = word_src[sample_id][token_id].item()
                    if id2word[word] != "" and id2word[word] != "<pad>":
                        all_predictions.append(id2word[word] + "\t" + id2label[sample] + "")
                        true_label = word_tgt[sample_id].item()
                        all_pairs.append(id2word[word] + "\t" + id2label[sample] + "\t" + id2label[true_label])
                    token_id += 1
                sample_id += 1
            else:
                for token in sample:
                    word = word_src[sample_id][token_id].item()
                    if id2word[word] != "" and id2word[word] != "<pad>":
                        all_predictions.append(id2word[word] + "\t" + id2label[token] + "")
                        true_label = word_tgt[sample_id][token_id].item()
                        all_pairs.append(id2word[word] + "\t" + id2label[token] + "\t" + id2label[true_label])
                    token_id += 1
                sample_id += 1
            all_predictions.append("")
            all_pairs.append("")

    return all_predictions

all_valid_ensemble_predictions, all_valid_ensemble_true_labels = [], []
all_test_ensemble_predictions = []
for i in range(1, 6):
    if i > 1:
        PATH = "{}/{}.pt".format(constant.params["model_dir"][:-1] + "_" + str(i), constant.params["save_path"])
    else:
        PATH = "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"])
    model.load_state_dict(torch.load(PATH))

    all_predictions, all_true_labels, all_pairs, valid_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, valid_loader, word2id, id2word, label2id, id2label)
    all_valid_ensemble_predictions.append(all_predictions)
    all_valid_ensemble_true_labels.append(all_true_labels)
    # print(all_predictions)
    all_valid_true_labels = all_true_labels

    # compute fb1 and accu
    fb1 = f1_score(all_true_labels, all_predictions)

    all_flatten_true_labels = []
    all_flatten_predictions = []
    for x in all_true_labels:
        all_flatten_true_labels += list(x)
    for x in all_predictions:
        all_flatten_predictions += list(x)
    accu = accuracy_score(all_flatten_true_labels, all_flatten_predictions)
    if metric == "fb1":
        best_valid = fb1
    elif metric == "accu":
        best_valid = accu
    print(len(all_predictions), len(all_true_labels))
    print("[Model {}] accu: {:3.5}, fb1: {:3.5f}, best-{}: {:3.5f}".format(i, accu, fb1, metric, best_valid))

    all_predictions, all_true_labels, all_pairs, test_log_loss, attn_scores, bpe_attn_scores = trainer.evaluate(model, test_loader, word2id, id2word, label2id, id2label)

    all_pairs = predict(model, test_loader, word2id, id2word, label2id, id2label, raw_test)
    all_test_ensemble_predictions.append(all_pairs)
    # print(len(all_predictions))
    # print([len(x) for x in])

# voting
final_valid_prediction = []
final_test_prediction = []
for i in range(len(all_valid_ensemble_predictions[0])):
    arr = []
    for l in range(len(all_valid_ensemble_predictions[0][i])):
        m = {}
        best_val = 0
        best_key = None
        for j in range(1, 6):
            key = all_valid_ensemble_predictions[j-1][i][l]
            if key not in m:
                m[key] = 1
            else:
                m[key] += 1
            if m[key] > best_val:
                best_val = m[key]
                best_key = key
        arr.append(best_key)
    final_valid_prediction.append(arr)

for i in range(len(all_test_ensemble_predictions[0])):
    if all_test_ensemble_predictions[0][i] == "":
        final_test_prediction.append("")
    else:
        m = {}
        best_val = 0
        best_key = None
        for j in range(1, 6):
            key = all_test_ensemble_predictions[j-1][i].split("\t")[-1]
            if key not in m:
                m[key] = 1
            else:
                m[key] += 1
            if m[key] > best_val:
                best_val = m[key]
                best_key = key
        final_test_prediction.append(best_key)
    
# evaluate on ensemble model
print(len(all_valid_true_labels), len(final_valid_prediction), len(all_valid_ensemble_predictions[0]))
fb1 = f1_score(all_valid_true_labels, final_valid_prediction)

all_flatten_true_labels = []
all_flatten_predictions = []
for x in all_valid_true_labels:
    all_flatten_true_labels += list(x)
for x in final_valid_prediction:
    all_flatten_predictions += list(x)
accu = accuracy_score(all_flatten_true_labels, all_flatten_predictions)
if metric == "fb1":
    best_valid = fb1
elif metric == "accu":
    best_valid = accu
print("Valid accu: {:3.5}, fb1: {:3.5f}, best-{}: {:3.5f}".format(accu, fb1, metric, best_valid))

with open(pred_file, "w+") as f:
    for i in range(len(final_test_prediction)):
        f.write(final_test_prediction[i] + "\n")

# with open("attention_scores.txt", "w+") as file_out:
#     for i in range(len(attn_scores)):
#         arr = attn_scores[i]
#         for j in range(len(arr)):
#             file_out.write(str(arr[j]))
#         file_out.write("\n")

# with open("bpe_attention_scores.txt", "w+") as file_out:
#     for i in range(len(bpe_attn_scores)):
#         arr = bpe_attn_scores[i]
#         for j in range(len(arr)):
#             file_out.write(str(arr[j]))
#         file_out.write("\n")