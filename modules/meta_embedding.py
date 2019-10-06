from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import numpy as np
import math

from tqdm import tqdm

from modules import transformer

def read_word_embeddings(word2id, id2word, emb_dim, emb_file):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.zeros((len(word2id), emb_dim))
    print('Embeddings: %d x %d' % (len(word2id), emb_dim))
    if emb_file is not None:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        num_line = 0
        for line in open(emb_file, encoding="utf-8").readlines():
            sp = line.split()
            if(len(sp) == emb_dim + 1): 
                if sp[0] in word2id:
                    pre_trained += 1
                    embeddings[word2id[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:",sp[0])
            num_line += 1
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / len(word2id)))
        print('Num line: %d' % (num_line))
    return embeddings

def gen_new_word_embedding(word2id, id2word, emb_size, file_path):
    emb = nn.Embedding(len(word2id), emb_size)
    vectors = read_word_embeddings(word2id, id2word, emb_size, file_path)
    emb.weight.data.copy_(torch.FloatTensor(vectors))
    emb.weight.requires_grad = False
    return emb

def gen_new_bpe_embedding(emb_vectors, num_vocab, emb_size):
    """
        Generate nn.Embedding object
    """
    # num_vocab, emb_size = emb.vectors.shape[0], emb.vectors.shape[1]
    emb = nn.Embedding(num_vocab, emb_size)
    emb.weight.data.copy_(torch.FloatTensor(emb_vectors))
    emb.weight.requires_grad = False
    return emb

class BPEMetaEmbedding(nn.Module):
    def __init__(self, embs, bpe_hidden_size, num_layers=1, num_heads=4, dim_key=32, dim_value=32, filter_size=32, max_length=100, input_dropout=0.1, layer_dropout=0.1, attn_dropout=0.1, relu_dropout=0.1,mode="attn_sum", no_projection=False, cuda=False):
        super(BPEMetaEmbedding, self).__init__()

        self.embs = embs
        self.bpe_embs = [gen_new_bpe_embedding(bpe_emb.vectors, bpe_emb.vectors.shape[0], bpe_emb.vectors.shape[1]) for bpe_emb in self.embs]
        self.bpe_embs = nn.ModuleList(self.bpe_embs)
        self.bpe_emb_sizes = [self.embs[i].vectors.shape[1] for i in range(len(self.embs))]
        self.mode = mode
        self.no_projection = no_projection
        self.bpe_hidden_size = bpe_hidden_size

        self.bpe_encoders = [
            transformer.Encoder(
                self.bpe_emb_sizes[i],
                hidden_size=bpe_hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_key=dim_key,
                dim_value=dim_value,
                filter_size=filter_size,
                max_length=max_length,
                input_dropout=input_dropout,
                layer_dropout=layer_dropout,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                use_mask=False
            ) for i in range(len(self.bpe_embs))]
        self.bpe_encoders = nn.ModuleList(self.bpe_encoders)

        if not self.no_projection:
            self.proj_matrix = nn.ModuleList([nn.Linear(bpe_hidden_size, bpe_hidden_size) for i in range(len(self.bpe_embs))])

        if cuda:
            self.bpe_embs = self.bpe_embs.cuda()
            self.bpe_encoders = self.bpe_encoders.cuda()
            if not self.no_projection:
                self.proj_matrix = self.proj_matrix.cuda()

        self.init_weights()

    def init_weights(self):
        if not self.no_projection:
            for i in range(len(self.proj_matrix)):
                self.init_layer(self.proj_matrix[i])

    def init_layer(self, m):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_bpes):
        """
            input_bpes: batch_size, num_of_emb, max_seq_len, uni_emb
        """
        attn_scores = None
        bpe_emb_vec = []

        input_bpes = input_bpes.transpose(0, 1) # num_of_emb, batch_size, max_word_len, max_bpe_len
        for i in range(len(input_bpes)):
            emb = self.bpe_embs[i](input_bpes[i])
            batch_size, max_seq_len, max_bpe_len, emb_size = emb.size(0), emb.size(1), emb.size(2), emb.size(3)

            emb = emb.reshape(batch_size * max_seq_len, max_bpe_len, emb_size)
            bpe_emb = self.bpe_encoders[i](emb) # batch_size * max_word_len, max_bpe_len, uni_emb
            bpe_emb = bpe_emb.reshape(batch_size, max_seq_len, max_bpe_len, self.bpe_hidden_size) # batch_size, max_word_len, max_bpe_len, uni_emb
            trained_bpe_emb = torch.sum(bpe_emb, dim=2).unsqueeze(-1) # batch_size, max_word_len, uni_emb, 1
            bpe_emb_vec.append(trained_bpe_emb)

        if len(self.bpe_embs) > 1:
            if len(bpe_emb_vec[0]) == 1:
                embedding = torch.stack(bpe_emb_vec, dim=-1).squeeze().unsqueeze(0) # 1 x word_len x bpe_seq_len x uni_emb_size x num_emb
            else:
                embedding = torch.stack(bpe_emb_vec, dim=-1).squeeze() # batch_size x word_len x bpe_seq_len x uni_emb_size x num_emb
            
            if self.mode == "attn_sum":
                attn = torch.tanh(embedding)
                attn_scores = F.softmax(attn, dim=-1)

                sum_embedding = None
                for i in range(embedding.size(-1)):
                    if i == 0:
                        sum_embedding = embedding[:,:,:,i] * attn_scores[:,:,:,i]
                    else:
                        sum_embedding = sum_embedding + embedding[:,:,:,i] * attn_scores[:,:,:,i]
            elif self.mode == "linear":
                sum_embedding = None
                for i in range(embedding.size(-1)):
                    if i == 0:
                        sum_embedding = embedding[:,:,:,i]
                    else:
                        sum_embedding = sum_embedding + embedding[:,:,:,i]
            elif self.mode == "concat":
                sum_embedding = torch.cat(bpe_emb_vec, dim=-1).squeeze() # batch_size x seq_len x (uni_emb_size x num_emb)

            final_embedding = sum_embedding
        else:
            final_embedding = bpe_emb_vec[0].squeeze(-1)
            embedding = bpe_emb_vec[0].squeeze(-1)

        return final_embedding, embedding, attn_scores

class WordMetaEmbedding(nn.Module):
    def __init__(self, emb_list, emb_size, word2id, id2word, label2id, id2label, mode="attn_sum", no_projection=False, cuda=False):
        super(WordMetaEmbedding, self).__init__()

        self.emb_list = emb_list
        self.emb_size = emb_size
        self.word_embs = [gen_new_word_embedding(word2id, id2word, emb_size, file_path) for file_path in self.emb_list]
        self.word_embs = nn.ModuleList(self.word_embs)

        self.word2id = word2id
        self.id2word = id2word
        self.label2id = label2id
        self.id2label = id2label

        self.mode = mode
        self.no_projection = no_projection

        if not self.no_projection:
            self.proj_matrix = nn.ModuleList([nn.Linear(emb_size, emb_size) for i in range(len(self.bpe_embs))])

        if cuda:
            self.word_embs = self.word_embs.cuda()
            if not self.no_projection:
                self.proj_matrix = self.proj_matrix.cuda()

        self.init_weights()

    def init_weights(self):
        if not self.no_projection:
            for i in range(len(self.proj_matrix)):
                self.init_layer(self.proj_matrix[i])

    def init_layer(self, m):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_words):
        attn_scores = None
        word_emb_vec = []
        for i in range(len(self.pretrained_emb)):
            if self.no_projection:
                new_emb = self.pretrained_emb[i](input_words).unsqueeze(-1) # batch_size x seq_len x uni_emb_size x 1
            else:
                new_emb = self.proj_matrix[i](self.pretrained_emb[i](input_words)).unsqueeze(-1) # batch_size x seq_len x uni_emb_size x 1
            word_emb_vec.append(new_emb)
        
        if self.mode == "attn_sum":
            if len(self.pretrained_emb) > 1:
                if len(word_emb_vec[0]) == 1:
                    embedding = torch.stack(word_emb_vec, dim=-1).squeeze().unsqueeze(0) # 1 x seq_len x uni_emb_size x num_emb
                else:  
                    embedding = torch.stack(word_emb_vec, dim=-1).squeeze() # batch_size x seq_len x uni_emb_size x num_emb
                attn = torch.tanh(embedding)
                attn_scores = F.softmax(attn, dim=-1)

                sum_embedding = None
                if len(embedding.size()) == 3:
                    embedding = embedding.unsqueeze(0)
                for i in range(embedding.size(-1)):
                    if i == 0:
                        sum_embedding = embedding[:,:,:,i] * attn_scores[:,:,:,i]
                    else:
                        sum_embedding = sum_embedding + embedding[:,:,:,i] * attn_scores[:,:,:,i]
                final_embedding = sum_embedding
            else:
                final_embedding = word_emb_vec[0].squeeze(-1)
                embedding = word_emb_vec[0].squeeze(-1)
        elif self.mode == "concat":
            if len(self.pretrained_emb) > 1:
                for i in range(len(word_emb_vec)):
                    word_emb_vec[i] = word_emb_vec[i].squeeze(-1)

                if len(word_emb_vec[0]) == 1:
                    embedding = torch.cat(word_emb_vec, dim=-1).squeeze().unsqueeze(0) # 1 x seq_len x (uni_emb_size x num_emb)
                else:  
                    embedding = torch.cat(word_emb_vec, dim=-1).squeeze() # batch_size x seq_len x (uni_emb_size x num_emb)
                
                final_embedding = embedding
            else:
                final_embedding = word_emb_vec[0].squeeze(-1)
                embedding = final_embedding
        elif self.mode == "linear":
            if len(self.pretrained_emb) > 1:
                final_embedding = None
                attn_scores = None
                for i in range(len(self.pretrained_emb)):
                    if final_embedding is None:
                        final_embedding = word_emb_vec[i]
                    else:
                        final_embedding = final_embedding + word_emb_vec[i]
                final_embedding = final_embedding.squeeze(-1)
                embedding = final_embedding
            else:
                final_embedding = word_emb_vec[0].squeeze(-1)
                embedding = word_emb_vec[0].squeeze(-1)
                attn_scores = None
        else:
            print("mode is not defined")

        return final_embedding, embedding, attn_scores