from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from utils.field import Field
from utils.vectors import get_all_vectors
from tqdm import tqdm
import unicodedata as ud
import numpy as np
import logging
import re
import string
import sys
import argparse
import math

logger = logging.getLogger(__name__)

import emoji

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

def text_has_punc(text):
    for character in text:
        if character in string.punctuation:
            return True
    return False

def run_preprocess_token(word, preprocess_token=False, stemming_arabic=False, new_preprocess=False):
    """ Word Tokenization """

    if not preprocess_token:
        return word

    # print(len(word))
    if word[0] == '@':
        return "<usr>"
    elif word[0] == '#':
        return "<usr>"
    else:
        urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', word)
        if len(urls) > 0:
            return "<url>"
        else:
            if word.isdigit():
                return "<num>"

            if text_has_emoji(word):
                return "<emo>"
    return word

class Dataset(data.Dataset):
    def __init__(self, inputs, targets, raw_inputs, word2id, id2word, label2id, id2label):
        self.inputs = inputs
        self.targets = targets
        self.raw_inputs = raw_inputs

        self.word2id = word2id
        self.id2word = id2word

        self.label2id = label2id
        self.id2label = id2label

    def __getitem__(self, index):
        word_input_id, word_target_id = self.vectorize(self.inputs[index], self.targets[index])
        raw_input_id = self.raw_inputs[index]
        return word_input_id, word_target_id, raw_input_id

    def __len__(self):
        return len(self.inputs)

    def vectorize(self, input, target):
        word_input_id = []
        word_target_id = []

        for i in range(len(input)):
            word_input_id.append(self.word2id[input[i]])

        for i in range(len(target)):
            word_target_id.append(self.label2id[target[i]])

        return word_input_id, word_target_id

def read_data(file_path, stemming_arabic=False, new_preprocess=False):
    inputs, targets, raw_inputs, raw_seqs = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as file_in:
        input_seq, target_seq, raw_seq = [], [], []
        for line in file_in:
            line = line.replace('\n', '')
            arr = line.split('\t')
            if len(arr) == 2:
                word, postag = arr[0], arr[1]
                if arr[1] == "":
                    if len(input_seq) > 0:
                        inputs.append(input_seq)
                        targets.append(target_seq)
                        raw_seqs.append(raw_seq)
                        input_seq, target_seq, raw_seq =  [], [], []
                else:
                    raw_seq.append(word)
                    word = run_preprocess_token(word, preprocess_token=True)
                    input_seq.append(word)
                    target_seq.append(postag)                    
                    raw_inputs.append(arr[0])
            else:
                if len(input_seq) > 0:
                    inputs.append(input_seq)
                    targets.append(target_seq)
                    raw_seqs.append(raw_seq)
                    input_seq, target_seq, raw_seq =  [], [], []

    if len(input_seq) > 0:
        inputs.append(input_seq)
        targets.append(target_seq)
        raw_seqs.append(raw_seq)
        input_seq, target_seq, raw_seq =  [], [], []
    return inputs, targets, raw_inputs, raw_seqs

def generate_vocab(train_file, validation_file, test_file):
    word2id, id2word = {}, {}

    label2id, id2label = {}, {}

    train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_data(train_file)
    valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_data(validation_file)
    test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file)

    # WORD-LEVEL
    word_list = ["<pad>", "<unk>", "<usr>", "<url>", "<num>"] # <emo> is found
    for i in range(len(word_list)):
        word2id[word_list[i]] = len(word2id)
        id2word[len(id2word)] = word_list[i]

    for i in range(len(train_inputs)):
        for word in train_inputs[i]:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word

    for i in range(len(valid_inputs)):
        for word in valid_inputs[i]:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word

    for i in range(len(test_inputs)):
        for word in test_inputs[i]:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(id2word)] = word

    # LABEL
    for i in range(len(train_targets)):
        for label in train_targets[i]:
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label

    for i in range(len(valid_targets)):
        for word in valid_targets[i]:
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label

    for i in range(len(test_targets)):
        for word in test_targets[i]:
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label

    return word2id, id2word, label2id, id2label

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        return padded_seqs, lengths 

    # data.sort(key=lambda x:len(x[0]), reverse=True)
    word_x, word_y, raw_x = zip(*data)
    word_x, x_len = merge(word_x)
    word_y, y_len = merge(word_y)
    word_x = torch.LongTensor(word_x)
    word_y = torch.LongTensor(word_y)

    return word_x, word_y, x_len, raw_x

def prepare_dataset(train_file, valid_file, test_file, batch_size, eval_batch_size):
    train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_data(train_file)
    valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_data(valid_file)
    test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file)

    print("train:", len(train_inputs), len(train_targets))
    print("valid:", len(valid_inputs), len(valid_targets))
    print("test:", len(test_inputs), len(test_targets))

    word2id, id2word, label2id, id2label = generate_vocab(train_file, valid_file, test_file)
    
    train_dataset = Dataset(train_inputs, train_targets, train_raw_seqs, word2id, id2word, label2id, id2label)
    valid_dataset = Dataset(valid_inputs, valid_targets, valid_raw_seqs, word2id, id2word, label2id, id2label)
    test_dataset = Dataset(test_inputs, test_targets, test_raw_seqs, word2id, id2word, label2id, id2label)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)
    
    return train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, test_raw_inputs