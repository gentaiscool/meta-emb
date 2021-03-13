import itertools
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# from .conlleval import conll_evaluation
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

def ner_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp)
    metrics["REC"] = recall_score(list_label, list_hyp)
    metrics["PRE"] = precision_score(list_label, list_hyp)
    return metrics

def pos_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp)
    metrics["REC"] = recall_score(list_label, list_hyp)
    metrics["PRE"] = precision_score(list_label, list_hyp)
    return metrics