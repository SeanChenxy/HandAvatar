# coding: UTF-8
"""
    @date:  2022.07.22  week30  星期五
    @func:  SingleVarianceNetwork迁移.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)