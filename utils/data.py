from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchtext import data
import torch
import numpy as np
import logging
import re
logger = logging.getLogger(__name__)

import re
import string
import sys
import argparse
import math
from tqdm import tqdm
import unicodedata as ud
from bpemb import BPEmb
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
    def __init__(self, inputs, targets, raw_inputs, word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs=None):
        self.inputs = inputs
        self.targets = targets
        self.raw_inputs = raw_inputs

        self.word2id = word2id
        self.id2word = id2word

        self.char2id = char2id
        self.id2char = id2char

        self.label2id = label2id
        self.id2label = id2label

        self.bpe_embs = bpe_embs

    def __getitem__(self, index):
        word_input_id, word_target_id, bpe_input_ids, char_input_id = self.vectorize(self.inputs[index], self.targets[index])
        raw_input_id = self.raw_inputs[index]
        return word_input_id, word_target_id, bpe_input_ids, char_input_id, raw_input_id

    def __len__(self):
        return len(self.inputs)

    def vectorize(self, input, target):
        word_input_id = []
        word_target_id = []
        bpe_input_ids = []
        char_input_id = []

        for i in range(len(input)):
            word_input_id.append(self.word2id[input[i]])
            char_arr_id = []
            for char in input[i]:
                char_arr_id.append(self.char2id[char])
            char_input_id.append(char_arr_id)

        for i in range(len(self.bpe_embs)):
            bpe_input_id = []
            for j in range(len(input)):
                word = input[j]
                subwords = self.bpe_embs[i].encode_ids(word)
                bpe_input_id.append(subwords)
            bpe_input_ids.append(bpe_input_id) # num_embs, num_word, num_subword
        #     print(">", bpe_input_id)
        # print(bpe_input_ids)

        # for classification
        if isinstance(target, str):
            word_target_id.append(self.label2id[target])
        else:
            for i in range(len(target)):
                word_target_id.append(self.label2id[target[i]])
            # char_arr_id = []
            # for char in target[i]:
            #     char_arr_id.append(self.char2id[char])
            # char_target_id.append(char_arr_id)

        return word_input_id, word_target_id, bpe_input_ids, char_input_id

def read_classification_data(file_path, stemming_arabic=False, new_preprocess=False, default_label="positive"):
    inputs, targets, raw_inputs, raw_seqs = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as file_in:
        input_seq, target_seq, raw_seq = [], [], []
        target = default_label
        m = {}
        for line in file_in:
            line = line.replace('\n', '')
            
            arr = line.split('\t')
            if len(arr) == 2:
                if "# sent_enum = " in line:
                    # print(">>", target, arr)
                    target = arr[1]
                    input_seq, raw_seq = [], []
                else:
                    tokens = emoji.demojize(arr[0]) 
                    tokens = tokens.replace("::",":\t:").replace(":","").replace("_","").split("\t")
                    # print(len(tokens))
                    # try:
                    #     print(tokens)
                    # except Exception:
                    #     pass

                    for token in tokens:
                        if len(token) == 0:
                            continue
                        word = token
                        raw_seq.append(word)
                        word = run_preprocess_token(word, preprocess_token=True)

                        input_seq.append(word)
                    raw_inputs.append(arr[0])
            else:
                if len(input_seq) > 0:
                    inputs.append(input_seq)
                    targets.append(target)
                    raw_seqs.append(raw_seq)
                    # print(raw_seq, target)
                    if not target in m:
                        m[target] = 0
                    m[target] += 1
                    input_seq, raw_seq =  [], []
    print(m)

    if len(input_seq) > 0:
        inputs.append(input_seq)
        targets.append(target)
        raw_seqs.append(raw_seq)
        input_seq, raw_seq =  [], []
    return inputs, targets, raw_inputs, raw_seqs

def read_data(file_path, stemming_arabic=False, new_preprocess=False, default_label="O"):
    inputs, targets, raw_inputs, raw_seqs = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as file_in:
        input_seq, target_seq, raw_seq = [], [], []
        for line in file_in:
            line = line.replace('\n', '')
            arr = line.split('\t')
            if len(arr) == 3:
                word, postag = arr[0], arr[2]
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
            elif len(arr) == 2: # for testset
                word, postag = arr[0], default_label
                if arr[0] == "":
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

def generate_vocab(train_file, validation_file, test_file, stemming_arabic=False, new_preprocess=False, bpe_lang_list=None, bpe_vocab=5000, bpe_emb_size=300, bpe_cache="", eval_type="", default_label="O"):
    word2id, id2word = {}, {}
    char2id, id2char = {}, {}

    label2id, id2label = {}, {}

    if eval_type == "classification":
        train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_classification_data(train_file)
    else:
        train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_data(train_file, default_label=default_label)

    if eval_type == "classification":
        valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_classification_data(validation_file)
    else:
        valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_data(validation_file, default_label=default_label)

    if eval_type == "classification":
        test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_classification_data(test_file)
    else:
        test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file, default_label=default_label)

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

    # BPE-LEVEL
    bpe_embs = []
    if bpe_lang_list is not None:
        print("Loading BPE:", bpe_lang_list)
        for i in range(len(bpe_lang_list)):
            bpemb = BPEmb(lang=bpe_lang_list[i], dim=bpe_emb_size, vs=bpe_vocab, cache_dir=bpe_cache)
            bpe_embs.append(bpemb)

    # CHAR-LEVEL
    for i in range(len(word_list)):
        for word in word_list[i]:
            for char in word:
                if char not in char2id:
                    char2id[char] = len(char2id)
                    id2char[len(id2char)] = char

    for i in range(len(train_inputs)):
        for word in train_inputs[i]:
            for char in word:
                if char not in char2id:
                    char2id[char] = len(char2id)
                    id2char[len(id2char)] = char

    for i in range(len(valid_inputs)):
        for word in valid_inputs[i]:
            for char in word:
                if char not in char2id:
                    char2id[char] = len(char2id)
                    id2char[len(id2char)] = char

    for i in range(len(test_inputs)):
        for word in test_inputs[i]:
            for char in word:
                if char not in char2id:
                    char2id[char] = len(char2id)
                    id2char[len(id2char)] = char

    # LABEL
    for i in range(len(train_targets)):
        if eval_type == "classification":
            label = train_targets[i]
            # if label != "positive":
            #     print(label)
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label
        else:
            for label in train_targets[i]:
                if label not in label2id:
                    label2id[label] = len(label2id)
                    id2label[len(id2label)] = label

    for i in range(len(valid_targets)):
        if eval_type == "classification":
            label = valid_targets[i]
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label
        else:
            for word in valid_targets[i]:
                if label not in label2id:
                    label2id[label] = len(label2id)
                    id2label[len(id2label)] = label

    for i in range(len(test_targets)):
        if eval_type == "classification":
            label = test_targets[i]
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label
        else:
            for word in test_targets[i]:
                if label not in label2id:
                    label2id[label] = len(label2id)
                    id2label[len(id2label)] = label

    return word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        return padded_seqs, lengths 

    def merge_char(sequences):
        lengths = [len(seq) for seq in sequences]
        char_lengths = []
        max_char_length = -1
        for i in range(len(sequences)):
            seq = sequences[i]
            word_arr = []
            for j in range(len(seq)):
                word_arr.append(len(seq[j]))
                max_char_length = max(max_char_length, len(seq[j]))
                # print(">>>>>>>>", seq[j], max_char_length)
            char_lengths.append(word_arr)
        # print(">max_char_length:", max_char_length)
        padded_seqs = torch.zeros(len(sequences), max(lengths), max_char_length).long()
        for i, seq in enumerate(sequences):
            for j, word in enumerate(sequences[i]):
                wordlen = len(word)
                padded_seqs[i, j, :wordlen] = torch.FloatTensor(sequences[i][j])
        return padded_seqs, char_lengths

    def merge_bpe(sequences): # num_seqs, num_embs, num_word, num_subword
        max_word_length, max_subword = -1, -1
        for i in range(len(sequences)):
            seq = sequences[i]
            for j in range(len(seq)): # num_embs x num_word x num_subword
                max_word_length = max(max_word_length, len(seq[j]))
                for k in range(len(seq[j])):
                    max_subword = max(max_subword, len(seq[j][k]))

        padded_seqs = torch.zeros(len(sequences), len(sequences[i]), max_word_length, max_subword).long()
        for i, seq in enumerate(sequences):
            for j, emb in enumerate(sequences[i]):
                for k, word in enumerate(sequences[i][j]):
                    subwordlen = len(word)
                    padded_seqs[i, j, k, :subwordlen] = torch.LongTensor(word)
        return padded_seqs

    # data.sort(key=lambda x:len(x[0]), reverse=True)
    word_x, word_y, bpe_ids, char_x, raw_x = zip(*data)
    word_x, x_len = merge(word_x)
    word_y, y_len = merge(word_y)
    word_x = torch.LongTensor(word_x)
    word_y = torch.LongTensor(word_y)

    # bpe_list = []
    # print("hoho")
    if len(bpe_ids[0]) > 0:
        bpe_ids = merge_bpe(bpe_ids)
        bpe_ids = torch.LongTensor(bpe_ids)
    else:
        bpe_ids = None

    # for i in range(len(bpe_ids)):
    #     bpe_id, _ = merge_char(bpe_ids[i])
    #     print(">>", bpe_id.size())
    #     bpe_list.append(bpe_id)
    # bpe_list = torch.LongTensor(bpe_list)

    char_x, char_x_len = merge_char(char_x)
    # char_y, char_y_len = merge_char(char_y)
    char_x = torch.LongTensor(char_x)
    # char_y = torch.LongTensor(char_y)

    return word_x, word_y, bpe_ids, char_x, x_len, raw_x

def prepare_dataset(train_file, valid_file, test_file, batch_size, eval_batch_size, train_valid=False, bpe_lang_list=None, bpe_vocab=5000, bpe_emb_size=300, bpe_cache="", eval_type="ner", default_label="O"):
    if train_valid:
        if eval_type == "classification":
            all_inputs, all_targets, all_raw_inputs, all_raw_seqs = read_classification_data(train_file)
        else:
            all_inputs, all_targets, all_raw_inputs, all_raw_seqs = read_data(train_file, default_label=default_label)
        
        num_train = int(len(all_inputs) * 0.8)
        num_valid = len(all_inputs) - num_train

        train_inputs, train_targets, train_raw_seqs = all_inputs[:num_train], all_targets[:num_train], all_raw_seqs[:num_train]
        train_raw_inputs = []

        for i in tqdm(range(len(train_raw_seqs))):
            train_raw_inputs += train_raw_seqs[i]

        valid_inputs, valid_targets, valid_raw_seqs = all_inputs[num_train:], all_targets[num_train:], all_raw_seqs[num_train:]
        valid_raw_inputs = []
        for i in tqdm(range(len(valid_raw_seqs))):
            valid_raw_inputs += valid_raw_seqs[i]

        if eval_type == "classification":
            test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_classification_data(test_file)
        else:
            test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file, default_label=default_label)
    else:
        if eval_type == "classification":
            train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_classification_data(train_file)
        else:
            train_inputs, train_targets, train_raw_inputs, train_raw_seqs = read_data(train_file, default_label=default_label)
        if eval_type == "classification":
            valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_classification_data(valid_file)
        else:
            valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_data(valid_file, default_label=default_label)
        if eval_type == "classification":
            test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_classification_data(test_file)
        else:
            test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file, default_label=default_label)

    print("train:", len(train_inputs), len(train_targets))
    print("valid:", len(valid_inputs), len(valid_targets))
    print("test:", len(test_inputs), len(test_targets))

    word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs = generate_vocab(train_file, valid_file, test_file,
    bpe_lang_list=bpe_lang_list, bpe_vocab=bpe_vocab, bpe_emb_size=bpe_emb_size, bpe_cache=bpe_cache, eval_type=eval_type)
    
    train_dataset = Dataset(train_inputs, train_targets, train_raw_seqs, word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs)
    valid_dataset = Dataset(valid_inputs, valid_targets, valid_raw_seqs, word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs)
    test_dataset = Dataset(test_inputs, test_targets, test_raw_seqs, word2id, id2word, char2id, id2char, label2id, id2label, bpe_embs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False)
    
    return train_loader, valid_loader, test_loader, word2id, id2word, char2id, id2char, label2id, id2label, test_raw_inputs, bpe_embs
