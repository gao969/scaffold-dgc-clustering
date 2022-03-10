#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    # w_avg == w[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        # torch.div(input, other, out=None) ,返回--input/other   -----求均值
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
