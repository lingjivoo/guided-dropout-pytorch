import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def DR(x,strength,drop_rate):
    if drop_rate <=0:
        return x
    node_num = strength.size(0)
    strength = torch.abs(strength)
    max_strength = strength.max()
    norm_strength = ((strength/max_strength) * 100).int()
    bins = torch.bincount(norm_strength)

    max_count, max_count_index = bins.max(dim=0)
    inactive_region_mask = (norm_strength == max_count_index).int()
    probs = torch.ones(node_num).float()*(1-drop_rate)
    probs = probs.to(x.get_device())
    random_mask = torch.bernoulli(probs)
    dropout_mask = random_mask * (1-inactive_region_mask) + inactive_region_mask
    x = x * dropout_mask.detach().expand_as(x)
    return x


def DR2D(x,strength,drop_rate):
    if drop_rate <=0:
        return x
    node_num = strength.size(0)
    strength = torch.abs(strength)
    max_strength = strength.max()
    norm_strength = ((strength/max_strength) * 100).int()
    bins = torch.bincount(norm_strength)

    max_count, max_count_index = bins.max(dim=0)
    inactive_region_mask = (norm_strength == max_count_index).int()
    probs = torch.ones(node_num).float()*(1-drop_rate)
    probs = probs.to(x.get_device())
    random_mask = torch.bernoulli(probs)

    dropout_mask = random_mask * (1-inactive_region_mask) + inactive_region_mask
    x = x * dropout_mask.view(1,-1,1,1).detach().expand_as(x)
    return x


def top_k(x,strength,drop_rate):
    if drop_rate <=0:
        return x
    node_num = strength.size(0)
    strength = torch.abs(strength)
    K = int(drop_rate*node_num)
    th = strength.topk(k=K, dim=-1, largest=True)[0][-1]
    dropout_mask = (strength <= th).int()
    x = x * dropout_mask.detach().view(1,-1)
    return x
#
def top_k2D(x,strength,drop_rate):
    if drop_rate <=0:
        return x
    node_num = strength.size(0)
    strength = torch.abs(strength)
    K = int(drop_rate*node_num)
    th = strength.topk(k=K, dim=-1, largest=True)[0][-1]
    dropout_mask = (strength <= th).int()
    x = x * dropout_mask.detach().expand_as(x)
    return x


class GuidedDropout(nn.Module):
    def __init__(self, dim, drop_rate=0.2,drop_type = 'DR'):
        super(GuidedDropout, self).__init__()

        # dim: num of nodes of the hidden layer
        # drop_type: two types of guided dropout : 1. top_k  2.  DR (select the nodes from the activate region)
        self.dim = dim
        self.drop_type = drop_type
        # print(self.drop_type)
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("{} must in range 0-1".format(drop_rate))
        self.drop_rate = drop_rate
        self.begin_flag = False
        self.t = nn.Parameter(torch.FloatTensor(self.dim).uniform_(0,1))

    def forward(self,x):
        if self.train():
            if self.begin_flag:
                x = eval(self.drop_type)(x,self.t,self.drop_rate)
        x = x * self.t.expand_as(x)
        return x


class GuidedDropout2D(nn.Module):
    def __init__(self, dim, drop_rate=0.2,drop_type = 'DR2D'):
        super(GuidedDropout2D, self).__init__()

        # dim: num of nodes of the hidden layer
        # drop_type: two types of guided dropout : 1. top_k_2D  2.  DR2D (select the nodes from the activate region)
        self.dim = dim
        self.drop_type = drop_type
        # print(self.drop_type)
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("{} must in range 0-1".format(drop_rate))
        self.drop_rate = drop_rate
        self.begin_flag = False
        self.t = nn.Parameter(torch.FloatTensor(self.dim).uniform_(0,1))

    def forward(self,x):
        if self.train():
            if self.begin_flag:
                x = eval(self.drop_type)(x,self.t,self.drop_rate)
        x = x * self.t.view(1,-1,1,1).expand_as(x)
        return x
