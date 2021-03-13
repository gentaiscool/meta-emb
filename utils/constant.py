from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import numpy as np
import random

parser = argparse.ArgumentParser(description='TransformerNER')
parser.add_argument('--name', type=str, default='', help='name')
parser.add_argument('--train_file', type=str, default='data/calcs_eng_spa/train.txt', help='LSTM / TRFS')
parser.add_argument('--valid_file', type=str, default='data/calcs_eng_spa/valid.txt', help='LSTM / TRFS')
parser.add_argument('--test_file', type=str, default='data/calcs_eng_spa/test.txt', help='LSTM / TRFS')

parser.add_argument('--eval', type=str, default='calcs', help='calcs / wnut / pos / connl2002')
parser.add_argument('--model', type=str, default='TRFS', help='TRFS')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--embedding_size_char', type=int,
                    default=300, help='embedding size char')
parser.add_argument('--embedding_size_char_per_word', type=int,
                    default=100, help='embedding size char per word')
parser.add_argument('--embedding_size_word', type=int,
                    default=300, help='embedding size word')

parser.add_argument('--num_epochs', type=int,
                    default=200, help='num epochs')
parser.add_argument('--num_layers', type=int, default=4, help='num layers')
parser.add_argument('--num_heads', type=int, default=4, help='num heads')
parser.add_argument('--dim_key', type=int,
                    default=0, help='attention key channels')
parser.add_argument('--dim_value', type=int,
                    default=0, help='attention value channels')
parser.add_argument('--filter_size', type=int, default=128, help='filter size')
parser.add_argument('--filter_size_char', type=int,
                    default=64, help='filter size char')
parser.add_argument('--input_dropout', type=float,
                    default=0.2, help='input dropout')
parser.add_argument('--attn_dropout', type=float,
                    default=0.2, help='attention dropout')
parser.add_argument('--relu_dropout', type=float,
                    default=0.2, help='relu dropout')
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--lr_decay', type=str, default='noam_step')
parser.add_argument('--use_crf', action="store_true",
                    help='add CRF in the end')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--model_dir', type=str, default='MODEL')
parser.add_argument('--save_path', type=str, default='model')

parser.add_argument('--emb_list', nargs='+', type=str)
parser.add_argument('--bpe_lang_list', nargs='+', type=str)
parser.add_argument('--bpe_vocab', type=int, default=5000)
parser.add_argument('--bpe_emb_size', type=int, default=300)
parser.add_argument('--bpe_hidden_size', type=int, default=100)
parser.add_argument('--bpe_cache', type=str, default='embedding/BPE')

parser.add_argument('--loss_weight', type=float,
                    default=0, help='mse weight')
parser.add_argument('--loss', type=str, default='mse', help='mse or l1')
parser.add_argument('--hidden_size', type=int,
                    default=200, help='hidden size')
parser.add_argument('--max_length', type=int,
                    default=256, help='maximum length')
parser.add_argument('--max_vocab_size', type=int,
                    default=5000000, help='max vocabulary size (embedding layer)')
parser.add_argument('--out', type=str, default='calcs_eng_spa_preds.conll')
parser.add_argument('--no_word_emb', action="store_true", 
                    help='no word embedding')
parser.add_argument('--add_emb', action="store_true", 
                    help='add trainable emb')
parser.add_argument('--add_char_emb', action="store_true", 
                    help='add trainable char emb')

parser.add_argument('--drop', type=float, default=0, help='dropout')
parser.add_argument('--early_stop', type=int,
                    default=5, help='early stop')

parser.add_argument('--mode', type=str, default='attn_sum', help='attn_sum or concat or linear')
parser.add_argument('--no_projection', action='store_true',
                    help='without projection matrix')

parser.add_argument('--default_label', type=str, default='O', help='O')
parser.add_argument('--metric', type=str, default='fb1', help='fb1/acc')

# for eval
parser.add_argument('--pred_file', type=str, default='predictions.txt')

args = parser.parse_args()

USE_CUDA = args.cuda

params = {
    "task_name": args.name,
    "model": args.model,
    "eval": args.eval,
    "train_file": args.train_file,
    "valid_file": args.valid_file,
    "test_file": args.test_file,
    "batch_size": args.batch_size,
    "embedding_size_char": args.embedding_size_char,
    "embedding_size_char_per_word": args.embedding_size_char_per_word,
    "num_layers": args.num_layers,
    "num_heads": args.num_heads,
    "dim_key": args.dim_key,
    "dim_value": args.dim_value,
    "filter_size": args.filter_size,
    "filter_size_char": args.filter_size_char,
    "input_dropout": args.input_dropout,
    "attn_dropout": args.attn_dropout,
    "relu_dropout": args.relu_dropout,
    "lr": args.lr,
    "lr_decay": args.lr_decay,
    "use_crf": args.use_crf,
    "seed": args.seed,
    "cuda": args.cuda,
    "num_epochs": args.num_epochs,
    "save_path": args.save_path,
    "hidden_size": args.hidden_size,
    "dropout": args.drop,
    "model_dir": "saved_models/" + args.model_dir + "/",
    "optimizer_adam_beta1": 0.9,
    "optimizer_adam_beta2": 0.98,
    "learning_rate_warmup_steps": 500,
    "embedding_size_word": args.embedding_size_word,
    "max_length": args.max_length,
    "max_vocab_size": args.max_vocab_size,
    "out": args.out,
    "emb_list": args.emb_list,
    "bpe_lang_list": args.bpe_lang_list,
    "bpe_vocab": args.bpe_vocab,
    "bpe_emb_size": args.bpe_emb_size,
    "bpe_hidden_size": args.bpe_hidden_size,
    "bpe_cache": args.bpe_cache,
    "loss_weight": args.loss_weight,
    "loss": args.loss,
    "no_word_emb": args.no_word_emb,
    "add_emb": args.add_emb,
    "add_char_emb": args.add_char_emb,
    "early_stop": args.early_stop,
    "mode": args.mode,
    "no_projection": args.no_projection,
    "default_label": args.default_label,
    "metric": args.metric,
    "pred_file": args.pred_file
}

print(params)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)