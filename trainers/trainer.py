from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import logging

from tqdm import tqdm

from models.transformer_tagger import TransformerTagger
from utils import constant
from utils.training_common import compute_num_params, lr_decay_map
# from eval.metrics import measure
from seqeval.metrics import f1_score
# from seqeval.metrics import accuracy_score
from sklearn.metrics import accuracy_score

import numpy as np

class Trainer():

    def __init__(self):
        super(Trainer, self).__init__()

    def predict(self, model, test_loader, word2id, id2word, label2id, id2label, raw_test):
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

        prediction_path = "{}/{}".format(constant.params["model_dir"], constant.params["out"])
        with open(prediction_path, "w+", encoding="utf-8") as file_out:
            id = 0
            for i in range(len(all_predictions)):
                if all_predictions[i] != "":
                    file_out.write(raw_test[id] + "\t" + all_predictions[i].split("\t")[1])
                    id += 1
                file_out.write("\n")

        words_path = "{}/{}.words".format(constant.params["model_dir"], constant.params["out"])
        with open(words_path, "w+", encoding="utf-8") as file_out:
            for i in range(len(all_pairs)):
                file_out.write(all_pairs[i] + "\n")

    def evaluate(self, model, valid_loader, word2id, id2word, label2id, id2label):
        model.eval()

        all_predictions, all_true_labels, all_pairs, valid_log_loss, all_attn_scores, all_bpe_attn_scores, all_src_raws = [], [], [], [], [], [], []
        start_iter = 0

        pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
        for i, (data) in enumerate(pbar, start=start_iter):
            word_src, word_tgt, bpe_ids, char_src, src_lengths, src_raw = data

            if constant.USE_CUDA:
                word_src = word_src.cuda()
                word_tgt = word_tgt.cuda()
                char_src = char_src.cuda()
                if bpe_ids is not None:
                    bpe_ids = bpe_ids.cuda()

            predictions, loss, attn_scores, bpe_attn_scores = model.forward(word_src, word_tgt, bpe_ids, char_src, src_raw, src_lengths=src_lengths, print_loss=True)
            
            sample_predictions, sample_true_labels = [], []

            sample_id = 0
            for sample in predictions.cpu().numpy().tolist():
                token_id = 0
                # if classification
                if isinstance(sample, int):
                    true_label = word_tgt[sample_id].item()
                    sample_predictions.append(id2label[sample])
                    sample_true_labels.append(id2label[true_label])
                    sample_id += 1
                else:
                    for token in sample:
                        word = word_src[sample_id][token_id].item()
                        true_label = word_tgt[sample_id][token_id].item()
                        sample_predictions.append(id2label[token])
                        sample_true_labels.append(id2label[true_label])
                        if id2word[word] != "<pad>" and id2word[word] != "<bos>" and id2word[word] != "<eos>" and id2word[word] != "<unk>":
                            all_pairs.append(
                                id2word[word] + "\t" + id2label[token] + "\t" + id2label[true_label])
                        token_id += 1
                    sample_id += 1

            all_predictions.append(sample_predictions)
            all_true_labels.append(sample_true_labels)
            all_src_raws.append(src_raw)

            if attn_scores is not None:
                attn_energies = nn.Softmax(dim=-1)(torch.sum(attn_scores, dim=2))

                for j in range(len(attn_scores)):
                    for k in range(src_lengths[j]):
                        all_attn_scores.append(src_raw[j][k] + " " + str(attn_energies[j,k,:].detach().cpu().numpy()).replace("\n","").replace("  "," "))
                    all_attn_scores.append("")

            if bpe_attn_scores is not None:
                attn_energies = nn.Softmax(dim=-1)(torch.sum(bpe_attn_scores, dim=2))

                for j in range(len(bpe_attn_scores)):
                    for k in range(src_lengths[j]):
                        all_bpe_attn_scores.append(src_raw[j][k] + " " + str(attn_energies[j,k,:].detach().cpu().numpy()).replace("\n","").replace("  "," "))
                    all_bpe_attn_scores.append("")
            valid_log_loss.append(loss.item())

        return all_predictions, all_true_labels, all_pairs, valid_log_loss, all_attn_scores, all_bpe_attn_scores

    def train(self, model, task_name, train_loader, valid_loader, test_loader, word2id, id2word, label2id, id2label, raw_test, metric):
        """
        Train a model on a dataset
        """
        print("<unk>",word2id["<unk>"],"<pad>",word2id["<pad>"],"<usr>",word2id["<usr>"],"<url>",word2id["<url>"])

        print("label:", id2label)
        print("len word vocab:", len(word2id))

        if constant.USE_CUDA:
            model = model.cuda()

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(
            model_params,
            betas=(constant.params["optimizer_adam_beta1"],
                constant.params["optimizer_adam_beta2"]),
            lr=constant.params["lr"])

        lr_scheduler_step, lr_scheduler_epoch = None, None  # lr schedulers
        lrd_scheme, lrd_range = constant.params["lr_decay"].split('_')
        lrd_func = lr_decay_map()[lrd_scheme]

        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lrd_func(constant.params),
            last_epoch=int(model.iterations) or -1
        )

        lr_scheduler_epoch, lr_scheduler_step = None, None
        if lrd_range == 'epoch':
            lr_scheduler_epoch = lr_scheduler
        elif lrd_range == 'step':
            lr_scheduler_step = lr_scheduler
        else:
            raise ValueError("Unknown lr decay range {}".format(lrd_range))

        print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))
        print("Training scheme: {} {}".format(lrd_scheme, lrd_range))

        iterations, epochs, best_valid, best_valid_loss, best_epoch = 0, 0, 0, 100, 0
        cnt = 0
        start_iter = 0
        for epoch in range(constant.params["num_epochs"]):
            model.train()
            sys.stdout.flush()
            log_loss = []

            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                word_src, word_tgt, bpe_ids, char_src, src_lengths, src_raw = data
                if constant.USE_CUDA:
                    word_src = word_src.cuda()
                    word_tgt = word_tgt.cuda()
                    char_src = char_src.cuda()
                    if bpe_ids is not None:
                        bpe_ids = bpe_ids.cuda()

                output, loss, _, _ = model(word_src, word_tgt, bpe_ids, char_src, src_raw, src_lengths=src_lengths, loss_weight=constant.params["loss_weight"], loss_type=constant.params["loss"],print_loss=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if lr_scheduler_step:
                    lr_scheduler_step.step()

                log_loss.append(loss.item())
                iterations += 1
                pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format(
                    (epoch+1), np.mean(log_loss)))

            if lr_scheduler_epoch:
                lr_scheduler_epoch.step()

            print("Train loss: {:3.5f}".format(np.mean(log_loss)))

            # evaluation
            all_predictions, all_true_labels, all_pairs, valid_log_loss, _, _ = self.evaluate(model, valid_loader, word2id, id2word, label2id, id2label)

            # metrics = measure(all_pairs)

            # fb1 = metrics["fb1"] / (100)
            # print(all_true_labels)
            # print(">", all_predictions)
            fb1 = f1_score(all_true_labels, all_predictions)
            # accu = accuracy_score(all_true_labels, all_predictions)

            all_flatten_true_labels = []
            all_flatten_predictions = []
            for x in all_true_labels:
                all_flatten_true_labels += list(x)
            for x in all_predictions:
                all_flatten_predictions += list(x)
            # print(all_flatten_true_labels)
            # print(all_flatten_predictions)
            accu = accuracy_score(all_flatten_true_labels, all_flatten_predictions)

            if metric == "fb1":
                if best_valid < fb1:
                    print("(Epoch {:d}) save model:".format(epoch+1))
                    best_valid = fb1
                    best_valid_loss =  np.mean(valid_log_loss)
                    best_epoch = epoch
                    cnt = 0

                    if not os.path.isdir(constant.params["model_dir"]):
                        os.makedirs(constant.params["model_dir"])

                    # save model
                    torch.save(model.state_dict(), "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"]))

                    print("######    Run Prediction   ######")
                    # load the best model
                    # model_path = constant.params["model_dir"] + "/" + constant.params["save_path"] + ".pt"
                    # model.load_state_dict(torch.load(model_path))
                    self.predict(model, test_loader, word2id, id2word, label2id, id2label, raw_test)
                else:
                    cnt+=1
            elif metric == "accu":
                if best_valid < accu:
                    print("(Epoch {:d}) save model:".format(epoch+1))
                    best_valid = accu
                    best_valid_loss =  np.mean(valid_log_loss)
                    best_epoch = epoch
                    cnt = 0

                    if not os.path.isdir(constant.params["model_dir"]):
                        os.makedirs(constant.params["model_dir"])

                    # save model
                    torch.save(model.state_dict(), "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"]))

                    print("######    Run Prediction   ######")
                    # load the best model
                    # model_path = constant.params["model_dir"] + "/" + constant.params["save_path"] + ".pt"
                    # model.load_state_dict(torch.load(model_path))
                    self.predict(model, test_loader, word2id, id2word, label2id, id2label, raw_test)
                else:
                    cnt+=1
            
            print("(Epoch {:d}) Val loss: {:3.5f}, accu: {:3.5}, fb1: {:3.5f}, best-{}: {:3.5f}".format((epoch+1), np.mean(valid_log_loss), accu, fb1, metric, best_valid))
            if cnt >= constant.params["early_stop"]: # early stopping
                break
        return best_valid, best_valid_loss, best_epoch