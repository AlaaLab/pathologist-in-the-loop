from torch import optim
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  

def configure_optimizers(model, lr, weight_decay, gamma, lr_decay_every_x_epochs):
    #optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    print(lr_decay_every_x_epochs)
    #scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_decay_every_x_epochs, gamma=gamma)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=lr_decay_every_x_epochs)
    return optimizer, scheduler
    

def soft_iou_loss(pred, label):
    b = pred.size()[0]
    pred = pred.view(b, -1)
    label = label.view(b, -1)
    inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
    unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
    return torch.mean(1 - inter / unit)


#######################################################################
#########################Graoh Loss
##########################
######################################################################
    #
def graph_loss():
    Graph = np.ones((23, 23)) * 1
    _index = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 7]
    for i in range(23):
        for j in range(23):
            if i == j:
                Graph[i, j] = 5
            elif _index[i] != _index[j]:
                if i != 22 and j != 22:
                    Graph[i, j] = 5
                else:
                    Graph[i, j] = 5
    return torch.from_numpy(Graph).float()

def apply_along_axis(function, x, axis: int = 0):
    graph = graph_loss()
    return torch.stack([
        function(x_i, graph) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)

def find_scale(stacks, torch_Graph):
    return torch_Graph[stacks[0],stacks[1]]

class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2

def binary_cross_entropy(
    self,
    target,
    weight_harry= None,
    reduction: int = Reduction.MEAN.value,

):
    # We cannot currently model this without introducing data-dependent control flow
    # TORCH_CHECK(
    #     (input_val >= 0) && (input_val <= 1),
    #     "all elements of input should be between 0 and 1"
    # )
    input = torch.clamp(self,min=1e-7,max=1-1e-7)
    loss = (target - 1) * torch.log(1 - input) - target * torch.log(input)
    if weight_harry is not None:

        duplicates = loss.shape[2]

        scaler = weight_harry.expand(duplicates, 1,  weight_harry.shape[0]).reshape(
            weight_harry.shape[0],
            1,
            duplicates)

        loss = loss * scaler

    return apply_loss_reduction(loss, reduction)

def apply_loss_reduction(loss, reduction: int):
    if reduction == Reduction.MEAN.value:
        return torch.mean(loss)
    elif reduction == Reduction.SUM.value:
        return torch.sum(loss)
    else:
        return loss
#######################################################################
#########################Evaluation metrics############################
#######################################################################
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score,\
  precision_score, f1_score, balanced_accuracy_score
from itertools import product
from itertools import combinations
import pandas as pd


def one_vs_rest_metrics(y_true, y_pred, y_score):
    """
    Computes various one-vs-rest classification metrics.

    Parameters
    ----------
    y_true: array-like, shape (n_samples, )
      The true class labels.

    y_pred: array-like, shape (n_samples, )
      The predicted class labels.

    y_score: array-like, shape (n_samples, n_classes)
      The predicted scores for each class e.g. the class probabilities.

    Output
    ------
    ovr_metrics: pd.DataFrame, shape (n_classes, n_metrics)
        The one-vs-rest metrics for each class.
    """

    n_classes = y_score.shape[1]

    ovr_metrics = []
    for class_idx in range(n_classes):
        y_true_this_class = (y_true == class_idx).astype(int)
        y_pred_this_class = (y_pred == class_idx).astype(int)
        y_score_this_class = y_score[:, class_idx]

        metrics_this_class = binary_clf_metrics(y_true=y_true_this_class,
                                                y_pred=y_pred_this_class,
                                                y_score=y_score_this_class,
                                                pos_label=1)

        # helpful to have
        metrics_this_class['n_true'] = sum(y_true_this_class)
        metrics_this_class['n_pred'] = sum(y_pred_this_class)

        ovr_metrics.append(metrics_this_class)

    ovr_metrics = pd.DataFrame(ovr_metrics)
    ovr_metrics.index.name = 'class_idx'
    return ovr_metrics


def binary_clf_metrics(y_true, y_pred, y_score, pos_label=1):
    """
    Computes various binary classification metrics.

    Parameters
    ----------
    y_true: array-like, shape (n_samples, )
        The true class labels. Should be

    y_pred: array-like, shape (n_samples, )
        The predicted class labels.

    y_score: array-like, shape (n_samples, )
        The predicted scores e.g. class probabilities.

    pos_label: str or int
        See sklearn precision_score, recall_score, f1_score, etc

    Output
    ------
    metrics: dict
        The metrics.
    """

    return {'auc': roc_auc_score(y_true=y_true, y_score=y_score),
            'f1': f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label),
            'Recall': recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label),
  'Precision':precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
            # TODO: add others: e.g. precision/recall/f1, etc
            }


def one_vs_one_metrics(y_true, y_pred, y_score):
    """
    Computes various one-vs-one classification metrics.

    Parameters
    ----------
    y_true: array-like, shape (n_samples, )
      The true class labels.

    y_pred: array-like, shape (n_samples, )
      The predicted class labels.

    y_score: array-like, shape (n_samples, n_classes)
      The predicted scores for each class e.g. the class probabilities.

    Output
    ------
    ovo_metrics: list of list of dicts
        The one-vs-ove metrics for each pair of classes.
    """

    n_classes = y_score.shape[1]

    # ovo_metrics = [[None for _ in range(n_classes)] for _ in range(n_classes)]
    ovo_metrics = []
    for class_idx_a, class_idx_b in combinations(range(n_classes), 2):

        # TODO: is this the right way to come up with a score
        # for class a?
        y_a = y_score[:, class_idx_a] - y_score[:, class_idx_b]

        metrics = binary_clf_metrics(y_true=y_true == class_idx_a,
                                     y_pred=y_true == class_idx_a,
                                     y_score=y_a,
                                     pos_label=True)
        metrics['pos_class'] = class_idx_a
        metrics['neg_class'] = class_idx_b

        ovo_metrics.append(metrics)

    return pd.DataFrame(ovo_metrics).set_index(['pos_class', 'neg_class'])


def get_overall_multiclass_metrics(y_true, y_pred, y_score):
    """
    Computes various overall metrics for multiclass-classification.

    Parameters
    ----------
    y_true: array-like, shape (n_samples, )
      The true class labels.

    y_pred: array-like, shape (n_samples, )
      The predicted class labels.

    y_score: array-like, shape (n_samples, n_classes)
      The predicted scores e.g. class probabilities.

    Output
    ------
    metrics: dict
        The metrics.
    """

    return {'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),

            'balanced_accuracy': balanced_accuracy_score(y_true=y_true,
                                                         y_pred=y_pred),

            'auc_ovr_macro': roc_auc_score(y_true=y_true, y_score=y_score,
                                           average='macro',
                                           multi_class='ovr'),


            'auc_ovr_weighted': roc_auc_score(y_true=y_true, y_score=y_score,
                                           average='weighted',
                                           multi_class='ovr')}