
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_inverted, PositionalEmbedding, PositionalEmbedding2D
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

import re
import matplotlib.pyplot as plt
import seaborn as sns


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore

def random_masking_4D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, C, L, D = xb.shape
    x = xb.clone()

    # x = rearrange(x, 'B C L D -> B (C L) D') # [B, C, L, D]
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, C, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=2)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, C, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=2)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, C, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=2, index=ids_restore)                                  # [bs x num_patch]

    return x_masked, x_kept, mask, ids_restore

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size=2, d_model=512, dropout=0., sparsity_threshold=0.01, hidden_size_factor=4):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        # Patch Embedding parameters
        self.value_embedding = nn.Linear(patch_size, d_model, bias=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len))
        self.position_embedding = PositionalEmbedding2D(d_model, 11, ceil(seq_len / patch_size))
        self.dropout = nn.Dropout(dropout)
    
    def patch_embedding(self, x, patch_size):
        B, L, C = x.shape
        # do patching
        x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        x = self.padding_patch_layer(x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, L//patch_size, patch_size]
        # print(self.value_embeddings[index_of_patch](x).shape, self.position_embedding(x).shape)
        x = self.value_embedding(x) + self.position_embedding(x) # [B, C, L, D]
        return self.dropout(x) # [B, C, L, D]

    def forward(self, x, mask_ratio=0.4):
        B, L, C = x.shape
        x = self.patch_embedding(x, self.patch_size)
        x_masked, x_kept, mask, ids_restore = random_masking_4D(x, mask_ratio)
        return x_masked, mask


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


class CrossDimensionalPeriodicEncoderLayer(nn.Module):
    def __init__(self, cross_dimensional_attention, inter_periodic_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(CrossDimensionalPeriodicEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_dimensional_attention = cross_dimensional_attention
        self.inter_periodic_attention = inter_periodic_attention
        self.cross_dimensional_mlp = MLP(d_model, d_ff, dropout, activation)
        self.inter_periodic_mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        x = rearrange(x, 'B C L D -> (B L) C D')
        res = x
        x, attn = self.cross_dimensional_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res + self.dropout(x))

        res = x
        x = self.cross_dimensional_mlp(x)
        x = self.norm1(res + self.dropout(x))

        x = rearrange(x, '(B L) C D -> (B C) L D', L=L)
        res = x
        x, attn = self.inter_periodic_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm3(res + self.dropout(x))

        res = x
        x = self.inter_periodic_mlp(x)
        x = self.norm4(res + self.dropout(x))

        x = rearrange(x, '(B C) L D -> B C L D', C=C)

        return x, attn


class CrossDimensionalPeriodicEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(CrossDimensionalPeriodicEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, C, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class LinearPretrainHead(nn.Module):
    def __init__(self, patch_size, seq_len, d_model, dropout=0., n_vars=11):
        super(LinearPretrainHead, self).__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_size)
    
    def forward(self, xs):
        # xs Ps*[B, C, L, D] ids_stores [Ps, B, C*L]
        x = self.linear(self.dropout(xs))
        x = rearrange(x, 'B C L P -> B (L P) C', P=self.patch_size)[:,:self.seq_len,:] # [B, L, C]
        return x # [B, L, C]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len

        self.mask_ratio = configs.mask_ratio
        self.pretrain = configs.pretrain
        self.individual = configs.individual


        self.patch_size = configs.patch_size_ssl
        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, patch_size=self.patch_size, d_model=configs.d_model, dropout=configs.dropout)

        self.encoder = CrossDimensionalPeriodicEncoder(
            [
                CrossDimensionalPeriodicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )


        if self.pretrain:
            self.head = LinearPretrainHead(self.patch_size, self.seq_len, configs.d_model, dropout=configs.dropout, n_vars=configs.enc_in)
        # else:    
        #     self.head = LinearPredictionHead(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev


        B, L, C = x_enc.shape

        # Multi-scale periodic patch embedding 
        x_enc, masks = self.msppe(x_enc) # Ps*[B, C, L, D], [B, Ps]
        # Encoder and Decoder
        enc_out, attns = self.encoder(x_enc) # [B, C, VT, D]

        # Head
        dec_out = self.head(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out, masks
