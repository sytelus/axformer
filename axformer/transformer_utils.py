import copy
import math

import numpy as np

import torch
from torch import nn
from torch.functional import F



def clones(module, n):
    "Produce n identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None): #query/key/value shape [batch_size, heads, sentence_size, d_k]
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) # shape: [batch, heads, sentence_size, sentence_size]
    if mask is not None:
        # mask tensor is filled with True or False values, mask==0 negates it
        scores = scores.masked_fill(mask == 0, -1e9) # set some large -ve value where mask is to be applied, this will become zero after softmax
    p_attn = F.softmax(scores, dim = -1) # shape: [batch, heads, sentence_size, sentence_size]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # value product (shape [batch_size, heads, sentence_size, d_k]), attention_scores

def average_models(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))