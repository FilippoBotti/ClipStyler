#!/usr/bin/env python3

"""Fits a linear aesthetic model to precomputed CLIP embeddings."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)
        self.linear = self.linear.half()

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)
