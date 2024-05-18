
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, positional_encoding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, embed, freq, dropout=0.):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.embed = embed
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.freq = freq
    
    
    def get_patch_sizes(self, seq_len, exclude_zero=True):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:] if exclude_zero else 1 / torch.fft.rfftfreq(seq_len)
        patch_sizes = peroid_list.ceil().int().unique()
        return patch_sizes
    
    def fft(self, x):
        x = x.permute(0, 2, 1)
        x = torch.fft.fft(x)
        x = torch.norm(x, dim=-1)
        return x

    def forward(self, x, mark):
        # x [B, L, C] mark [B, L, M]
        B, L, C = x.shape
        _, _, M = mark.shape

        x = x.unsqueeze(-1) # [B, L, C, 1]
        x = rearrange(x, 'B L C 1 -> (B C) L 1') # [B*C, L, 1]
        mark = repeat(mark, 'B L M -> (B C) L M', C=C) # [B*C, L, M]

        x = torch.cat((x, mark), dim=-1) # [B*C, L, C+M]
        x = x.view(-1, self.embed) # [B*C*L, C+M]
        x = self.dropout(x)
        return x



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.mask_ratio = configs.mask_ratio
        self.individual = configs.individual

        self.pretrain = configs.pretrain

        self.revin = RevIN(configs.enc_in)

        if self.individual:
            self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

