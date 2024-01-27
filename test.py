## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator, Discriminator


def gLossDiff(pred: torch.Tensor, real: torch.Tensor):
    temp = torch.where(real == 0, pred, 20 * (1 - pred)) / 21
    loss = torch.mean(torch.sum(temp, dim=(1, 2, 3)))
    return loss


def recallAndPrecision(pred: torch.Tensor, real: torch.Tensor):
    pred = pred > 0.5
    real = real > 0.5
    total_overlap = torch.sum(pred & real, dim=(1, 2, 3))
    total_pred = torch.sum(pred, dim=(1, 2, 3))
    total_real = torch.sum(real, dim=(1, 2, 3))
    return (total_overlap / total_real, total_overlap / total_pred)


def test():
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.   
    return