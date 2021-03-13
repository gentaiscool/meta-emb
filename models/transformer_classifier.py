from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys

from tqdm import tqdm

from modules import transformer
from modules.embedding import gen_new_bpe_embedding, BPEMetaEmbedding, gen_new_word_embedding
from modules.outputs import CRFOutputLayer, SoftmaxOutputLayer

from eval.metrics import measure

import numpy as np
import math

def gen_word_embeddings(word2id, id2word, emb_dim, emb_file):
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
            # else:
            #     print("Error:",sp[0])
            num_line += 1
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / len(word2id)))
        print('Num line: %d' % (num_line))
    return embeddings

def gen_new_embedding(word2id, id2word, emb_size, file_path):
    emb = nn.Embedding(len(word2id), emb_size)
    vectors = gen_word_embeddings(word2id, id2word, emb_size, file_path)
    emb.weight.data.copy_(torch.FloatTensor(vectors))
    emb.weight.requires_grad = False
    return emb

# def gen_new_bpe_embedding(bpe_emb):
#     num_vocab, emb_size = bpe_emb.vectors.shape[0], bpe_emb.vectors.shape[1]
#     emb = nn.Embedding(num_vocab, emb_size)
#     emb.weight.data.copy_(torch.FloatTensor(bpe_emb.vectors))
#     emb.weight.requires_grad = False
#     return emb

class TransformerClassifier(nn.Module):
    """
    Sequence tagger using the Transformer network (https://arxiv.org/pdf/1706.03762.pdf)
    Uses Transformer Encoder.
    For character embeddings (word-level) it uses the same Encoder module above which
    an additive (Bahdanau) self-attention layer
    """

    def __init__(self, emb_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length, input_dropout, dropout, attn_dropout, relu_dropout, word2id, id2word, char2id, id2char, label2id, id2label, 
                char_emb_size=0, char_hidden_size=100, 
                cuda=True, pad_idx=0, use_crf=False, add_emb=False, no_word_emb=False, emb_list=[], 
                bpe_emb_size=0, bpe_hidden_size=100, bpe_lang_list=[], bpe_dim=300, bpe_vocab=5000, bpe_embs=None, 
                add_char_emb=False, use_cuda=False, mode="attn_sum", no_projection=False):
        super(TransformerClassifier, self).__init__()

        self.iterations = 0
        self.epochs = 0
        self.pad_idx = pad_idx

        # WORD
        self.word2id = word2id
        self.id2word = id2word

        # BPE
        self.bpe_lang_list = bpe_lang_list
        self.bpe_dim = bpe_dim
        self.bpe_vocab = bpe_vocab
        self.bpe_hidden_size = bpe_hidden_size
        self.bpe_emb_size = bpe_emb_size
        self.bpe_embs = bpe_embs

        # CHAR
        self.char2id = char2id
        self.id2char = id2char

        self.label2id = label2id
        self.id2label = id2label

        self.emb_size = emb_size
        self.use_crf = use_crf

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.filter_size = filter_size
        self.max_length = max_length
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.add_emb = add_emb
        self.add_char_emb = add_char_emb
        self.no_projection = no_projection
        self.mode = mode

        self.use_cuda = cuda

        self.emb_list = emb_list
        uni_emb_size = 300

        self.no_word_emb = no_word_emb
        if no_word_emb:
            concat_emb_size = 0
        else:
            if mode == "attn_sum":
                concat_emb_size = 300
            elif mode == "concat":
                concat_emb_size = 300 * len(self.emb_list)
            else:
                concat_emb_size = 300

        if self.add_char_emb:
            concat_emb_size += char_hidden_size
            self.char_hidden_size = char_hidden_size

        self.pretrained_emb = []
        if self.emb_list is not None and not self.no_word_emb:
            self.pretrained_emb = [gen_new_word_embedding(word2id, id2word, emb_size, file_path) for file_path in self.emb_list]

        if self.add_emb:
            if len(self.pretrained_emb) > 0:
                concat_emb_size += 300

        if add_emb:
            self.trained_emb = nn.Embedding(len(word2id), uni_emb_size)
            self.context_matrix = nn.Linear(uni_emb_size, uni_emb_size)

        if add_char_emb:
            self.char_emb = nn.Embedding(len(char2id), char_emb_size)
            self.char_transformer_enc = transformer.Encoder(
                char_emb_size,
                hidden_size=char_hidden_size,
                num_layers=1,
                num_heads=4,
                dim_key=dim_key,
                dim_value=dim_value,
                filter_size=filter_size,
                max_length=max_length,
                input_dropout=input_dropout,
                layer_dropout=dropout,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                use_mask=False
            )
        
        if not self.no_projection and not self.no_word_emb:
            self.proj_matrix = nn.ModuleList([nn.Linear(emb_size, uni_emb_size) for i in range(len(self.pretrained_emb))])
        
        if not self.no_word_emb:
            self.pretrained_emb = nn.ModuleList(self.pretrained_emb)

        if cuda:
            if not self.no_word_emb:
                if not self.no_projection:
                    self.proj_matrix = self.proj_matrix.cuda()
                if add_emb:
                    self.trained_emb = self.trained_emb.cuda()
                self.pretrained_emb = self.pretrained_emb.cuda()

        self.output_layer = SoftmaxOutputLayer(hidden_size, len(label2id))

        self.pretrained_bpe_emb = []
        self.add_bpe_emb = len(self.bpe_lang_list) > 0

        if self.add_bpe_emb:
            self.bpe_meta_emb = BPEMetaEmbedding(self.bpe_embs, bpe_hidden_size, num_layers=1, num_heads=4, dim_key=dim_key, dim_value=dim_value, filter_size=filter_size, max_length=max_length, input_dropout=input_dropout, layer_dropout=dropout, attn_dropout=attn_dropout, relu_dropout=relu_dropout, mode="attn_sum", no_projection=False, cuda=True)
    
            concat_emb_size += bpe_hidden_size

            if cuda:
                self.bpe_meta_emb = self.bpe_meta_emb.cuda()

        ### ASSIGN PARAMS ###
        self.transformer_enc = transformer.Encoder(
            concat_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_key=dim_key,
            dim_value=dim_value,
            filter_size=filter_size,
            max_length=max_length,
            input_dropout=input_dropout,
            layer_dropout=dropout,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            use_mask=False
        )

        self.init_weights()

    def init_weights(self):
        print("init weights")
        if not self.no_word_emb:
            if not self.no_projection:
                for i in range(len(self.proj_matrix)):
                    self.init_layer(self.proj_matrix[i])
            if self.add_emb:
                self.init_layer(self.context_matrix)

    def init_layer(self, m):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

    def compute(self, inputs_word_emb, src_lengths=None):
        # Apply Transformer Encoder
        enc_out = self.transformer_enc(inputs_word_emb)
        return enc_out

    def get_word_meta_embedding(self, input_words):
        mode = self.mode

        attn_scores = None
        word_emb_vec = []
        ori_emb_vec = []
        for i in range(len(self.pretrained_emb)):
            ori_emb = self.pretrained_emb[i](input_words)
            if self.no_projection:
                new_emb = self.pretrained_emb[i](input_words).unsqueeze(-1) # batch_size x seq_len x uni_emb_size x 1
            else:
                new_emb = self.proj_matrix[i](self.pretrained_emb[i](input_words)).unsqueeze(-1) # batch_size x seq_len x uni_emb_size x 1
            ori_emb_vec.append(ori_emb)
            word_emb_vec.append(new_emb)
        
        if mode == "attn_sum":
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
        elif mode == "concat":
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
        elif mode == "linear":
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

    def get_bpe_meta_embedding(self, input_bpes):
        attn_scores = None
        bpe_emb_vec = []
        ori_emb_vec = []
        for i in range(len(self.pretrained_bpe_emb)):
            ori_emb = input_bpes[i] # batch_size, max_seq_len, uni_emb
            new_emb = input_bpes[i].unsqueeze(-1) # batch_size x word_len x seq_len x uni_emb_size x 1
            ori_emb_vec.append(ori_emb)
            bpe_emb_vec.append(new_emb)

        if len(self.pretrained_bpe_emb) > 1:
            if len(bpe_emb_vec[0]) == 1:
                embedding = torch.stack(bpe_emb_vec, dim=-1).squeeze().unsqueeze(0) # 1 x word_len x bpe_seq_len x uni_emb_size x num_emb
            else:  
                embedding = torch.stack(bpe_emb_vec, dim=-1).squeeze() # batch_size x word_len x bpe_seq_len x uni_emb_size x num_emb
            attn = torch.tanh(embedding)
            attn_scores = F.softmax(attn, dim=-1)

            sum_embedding = None
            for i in range(embedding.size(-1)):
                if i == 0:
                    sum_embedding = embedding[:,:,:,i] * attn_scores[:,:,:,i]
                else:
                    sum_embedding = sum_embedding + embedding[:,:,:,i] * attn_scores[:,:,:,i]
            final_embedding = sum_embedding
        else:
            final_embedding = bpe_emb_vec[0].squeeze(-1)
            embedding = bpe_emb_vec[0].squeeze(-1)

        return final_embedding, embedding, attn_scores

    def forward(self, word_src, word_tgt, bpe_srcs, char_src, raw_inputs, src_lengths=None, loss_weight=0, loss_type='mse', print_loss=True):
        """
        NOTE: batch must have the following attributes:
            inputs_word, labels
        """
        attn_scores = None
        bpe_attn_scores = None
        all_embs = []
        if not self.no_word_emb:
            if len(self.pretrained_emb) > 0:
                meta_emb, all_emb, attn_scores = self.get_word_meta_embedding(word_src) # batch_size x seq_len x uni_emb
                all_embs.append(meta_emb)

            if self.add_emb:
                trained_emb = self.trained_emb(word_src) # batch_size x seq_len x uni_emb
                all_embs.append(trained_emb)
        
        if self.add_char_emb:
            char_src = self.char_emb(char_src)
            batch_size, max_seq_len, max_word_len, uni_emb = char_src.size(0), char_src.size(1), char_src.size(2), char_src.size(3)
            
            char_src = char_src.reshape(batch_size * max_seq_len, max_word_len, uni_emb)
            char_emb = self.char_transformer_enc(char_src) # max_seq_len, max_word_len, uni_emb
            char_emb = char_emb.reshape(batch_size, max_seq_len, max_word_len, self.char_hidden_size)
            trained_char_emb = torch.sum(char_emb, dim=2) # batch_size, uni_emb, max_seq_len
            
            if self.use_cuda:
                trained_char_emb = trained_char_emb.cuda()
            all_embs.append(trained_char_emb)

        # BPE
        if self.add_bpe_emb:
            bpe_meta_emb, _, bpe_attn_scores = self.bpe_meta_emb(bpe_srcs)

            if self.use_cuda:
                bpe_meta_emb = bpe_meta_emb.cuda()
            all_embs.append(bpe_meta_emb)

        if len(all_embs) == 1 and len(self.pretrained_emb) == 1:
            all_embs = meta_emb
        else:
            all_embs = torch.cat(all_embs, dim=-1)

        hidden = self.compute(all_embs, src_lengths=src_lengths)

        # hid_attn = torch.tanh(hidden)
        hid_attn_scores = F.softmax(hidden, dim=1)

        sum_embedding = None
        for i in range(hidden.size(1)):
            if i == 0:
                sum_embedding = hidden[:,i,:] * hid_attn_scores[:,i,:]
            else:
                sum_embedding = sum_embedding + hidden[:,i,:] * hid_attn_scores[:,i,:]

        # print(hidden[:,:,:].mean(1).squeeze().size())
        # print(hidden.size(), sum_embedding.size())
        output = self.output_layer(sum_embedding)

        if print_loss:
            loss_val = self.output_layer.loss(sum_embedding, word_tgt)
            return output, loss_val, attn_scores, bpe_attn_scores
        else:
            return output, None, attn_scores, bpe_attn_scores