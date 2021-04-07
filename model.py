# Author: Pang Haoyu
# Date: 19 December, 2020
# Description: A PyTorch implementation of MetaCDR

# Pytorch imports
import torch
import torch.nn as nn  
from torch.nn import functional as F
from torch.autograd import Variable

# Python imports
import numpy as np
from collections import OrderedDict
from copy import deepcopy

# Workspace imports
from embeddings import item, user
from argslist import parse_args

      
class Linear(nn.Linear):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__(in_dim, out_dim)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear, self).forward(x)
        
        return out

class BN(nn.BatchNorm1d):
    def __init__(self, dim):
        super(BN, self).__init__(dim)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, torch.mean(x.detach(), dim=0), torch.var(x.detach(), dim=0), weight = self.weight.fast, bias = self.bias.fast)
        else:
            out = super(BN, self).forward(x)
        
        return out

class CrossOrigin(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossOrigin, self).__init__()
        self.param = nn.ParameterDict({
            'wa': nn.Parameter(torch.randn(in_dim, out_dim)),  # weight for A domain
            'wb': nn.Parameter(torch.randn(in_dim, out_dim)),
            'h': nn.Parameter(torch.randn(in_dim, out_dim)),  # H matrix
            'ba': nn.Parameter(torch.randn(out_dim)),  # bias for A domain
            'bb': nn.Parameter(torch.randn(out_dim))
        })

    def forward(self, xa, xb):
        xa_out = torch.mm(xa, self.param['wa']) + torch.mm(xb, self.param['h']) + self.param['ba']
        xb_out = torch.mm(xb, self.param['wb']) + torch.mm(xa, self.param['h']) + self.param['bb']

        return xa_out, xb_out

class Cross(CrossOrigin):
    def __init__(self, in_dim, out_dim):
        super(Cross, self).__init__(in_dim, out_dim)
        self.param['wa'].fast = None 
        self.param['wb'].fast = None
        self.param['h'].fast = None
        self.param['ba'].fast = None
        self.param['bb'].fast = None

    def forward(self, xa, xb):
        if self.param['wa'].fast is not None and self.param['wb'].fast is not None and self.param['h'].fast is not None:
            xa_out = torch.mm(xa, self.param['wa'].fast) + torch.mm(xb, self.param['h'].fast) + self.param['ba'].fast
            xb_out = torch.mm(xb, self.param['wb'].fast) + torch.mm(xa, self.param['h'].fast) + self.param['bb'].fast
        else:
            xa_out, xb_out = super(Cross, self).forward(xa, xb)

        return xa_out, xb_out

class CrossDNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossDNN, self).__init__()
        self.fc1 = Cross(in_dim, 64)
        self.bn1s = BN(64)
        self.bn1t = BN(64)
        self.fc2 = Cross(64, 64)
        self.bn2s = BN(64)
        self.bn2t = BN(64)
        self.source_out = Linear(64, out_dim)
        self.target_out = Linear(64, out_dim)

    def forward(self, xa, xb):
        xa, xb = self.fc1(xa, xb)
        xa = self.bn1s(xa)
        xb = self.bn1t(xb)
        xa = F.relu(xa)
        xb = F.relu(xb)
        xa, xb = self.fc2(xa, xb)
        xa = self.bn2s(xa)
        xb = self.bn2t(xb)
        xa = F.relu(xa)
        xb = F.relu(xb)
        xa = self.source_out(xa)
        xb = self.target_out(xb)

        return xa, xb

class RecNet(nn.Module):
    def __init__(self, config):
        super(RecNet, self).__init__()
        self.embedding_dim = config.embedding_dim  # embedding dim for every feature
        self.domain_in_dim = self.embedding_dim * 8  # 4 movie feature, 4 user feature
        
        self.item_emb_source = item(config)  # embeder for item in source domain
        self.item_emb_target = item(config)  # embeder for item in target domain
        self.user_emb = user(config)  # embeder for user 

        self.final_part = CrossDNN(self.domain_in_dim, 1)

    def forward(self, xa, xb):
        gender_idx_source = xa[:, 10242]
        age_idx_source = xa[:, 10243]
        occupation_idx_source = xa[:, 10244]
        area_idx_source = xa[:, 10245]

        user_emb = self.user_emb(gender_idx_source, age_idx_source, occupation_idx_source, area_idx_source) 

        rate_idx_source = xa[:, 0]  # [[5],[5],[5],[4]]
        genre_idx_source = xa[:, 1:26]
        director_idx_source = xa[:, 26:2212]
        actor_idx_source = xa[:, 2212:10242]

        item_emb_source = self.item_emb_source(rate_idx_source, genre_idx_source, director_idx_source, actor_idx_source)  
        
        x_source = torch.cat((item_emb_source, user_emb), 1)
        
        rate_idx_target = xb[:, 0] 
        genre_idx_target = xb[:, 1:26]
        director_idx_target = xb[:, 26:2212]
        actor_idx_target = xb[:, 2212:10242]

        item_emb_target = self.item_emb_target(rate_idx_target, genre_idx_target, director_idx_target, actor_idx_target)  

        x_target = torch.cat((item_emb_target, user_emb), 1)

        x_source, x_target = self.final_part(x_source, x_target)

        return x_source, x_target

