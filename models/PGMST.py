import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_wo_pos, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import seaborn as sns

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
    
class PeriodGuidedMultiScaleRouter(nn.Module):
    def __init__(self, top_k, num_features, seq_len, mlp_ratio=4.):
        super(PeriodGuidedMultiScaleRouter, self).__init__()
        self.top_k = top_k
        # GET Patch sizes corresponding to the period
        self.patch_sizes = self.get_patch_sizes(seq_len)
        # AFNO1D
        self.start_fc = nn.Linear(num_features, 1)
        self.num_freqs = seq_len // 2
        self.mlp_ratio = mlp_ratio
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs, int(self.num_freqs * self.mlp_ratio)))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, int(self.num_freqs * self.mlp_ratio)))
        # self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor, len(self.patch_sizes)))
        # self.b2 = nn.Parameter(self.scale * torch.randn(2, len(self.patch_sizes)))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, int(self.num_freqs * self.mlp_ratio), self.num_freqs))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs))
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.patch_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.patch_sizes)))

    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy()[::-1]
        # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()
        return patch_sizes
    
    def forward(self, x, training, noise_epsilon=1e-2):
        # x = self.start_fc(x).squeeze(-1)
        x = self.start_fc(x.permute(0, 2, 3, 1)).squeeze(-1).mean(-1)

        B, L = x.shape
        xf = torch.fft.rfft(x, dim=-1, norm='ortho') # B, C, L//2+1

        xf_ac = xf[:, 1:]

        o1_real = torch.zeros([B, int(self.num_freqs * self.mlp_ratio)], device=x.device)
        o1_imag = torch.zeros([B, int(self.num_freqs * self.mlp_ratio)], device=x.device)
        o2_real = torch.zeros([B, self.num_freqs], device=x.device)
        o2_imag = torch.zeros([B, self.num_freqs], device=x.device)

        o1_real = F.relu(xf_ac.real @ self.w1[0] - xf_ac.imag @ self.w1[1] + self.b1[0])
        o1_imag = F.relu(xf_ac.imag @ self.w1[0] + xf_ac.real @ self.w1[1] + self.b1[1])
        o2_real = o1_real @ self.w2[0] - o1_imag @ self.w2[1] + self.b2[0]
        o2_imag = o1_imag @ self.w2[0] + o1_real @ self.w2[1] + self.b2[1]
        
        xf_ac = torch.stack([o2_real, o2_imag], dim=-1) # B, L-1, 2
        xf_ac = torch.view_as_complex(xf_ac)
        xf_ac = torch.abs(xf_ac) # B, L-1
        
        clean_logits = xf_ac @ self.w_gate

        # visual gates
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # sns.heatmap(clean_logits.detach().cpu().numpy(), ax=ax[0, 0], cmap='jet')

        if training:
            raw_noise_stddev = xf_ac @ self.w_noise
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            logits = noisy_logits # [B, L-1]
        else:
            logits = clean_logits # [B, L-1]
       
        weights = logits # B, L-1

        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1) # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1) # [B, top_k]

        zeros = torch.zeros_like(weights) # [B, Ps]
        gates = zeros.scatter_(-1, top_indices, top_weights) # [B, Ps]
        
        # sns.heatmap(noise.detach().cpu().numpy(), ax=ax[0, 1], cmap='jet')
        # sns.heatmap(weights.detach().cpu().numpy(), ax=ax[1, 0], cmap='jet')
        # sns.heatmap(gates.detach().cpu().numpy(), ax=ax[1, 1], cmap='jet')

        # plt.savefig('/root/MSPT/test_visuals/sets.png')

        return gates
    
def dispatcher(x, gates):
    # sort experts
    sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
    # get according batch index for each expert
    _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
    _part_sizes = (gates > 0).sum(0).tolist()
    # assigns samples to experts whose gate is nonzero
    # expand according to batch index so we can just split by _part_sizes
    xs = x[_batch_index].squeeze(1)
    return list(torch.split(xs, _part_sizes, dim=0))

def aggregate(xs, gates, multiply_by_gates=True):
    # sort experts
    sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0) 
    _, _expert_index = sorted_experts.split(1, dim=1)
    # get according batch index for each expert
    _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
    gates_exp = gates[_batch_index.flatten()]
    _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)
    # apply exp to expert outputs, so we are not longer in log space
    stitched = torch.cat(xs, 0).exp() # [BN L D]
    # stitched = torch.cat(xs, 0)
    if multiply_by_gates:
        stitched = torch.einsum("ijkh,ik -> ijkh", stitched, _nonzero_gates) # [BN L D] [BN L] -> [BN L D]
    zeros = torch.zeros(gates.size(0), xs[-1].size(1), xs[-1].size(2), xs[-1].size(3),
                        requires_grad=True, device=stitched.device)
    # combine samples that have been processed by the same k experts
    combined = zeros.index_add(0, _batch_index, stitched.float())
    # add eps to all zero values in order to avoid nans when going back to log space
    combined[combined == 0] = np.finfo(float).eps
    # go back to log space
    combined = combined.log()
    return combined # [B, L, D]
    
class PeriodicEncoderLayer(nn.Module):
    def __init__(self, attention, seq_len, patch_size, d_embed, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(PeriodicEncoderLayer, self).__init__()

        padding_len = ceil(seq_len / patch_size) * patch_size - seq_len
        self.patch_size = patch_size
        self.attention = attention
        self.start_fc = nn.Linear(seq_len, seq_len+padding_len)
        self.pos_embed = PositionalEmbedding(d_model, max_len=ceil(seq_len / patch_size))
        self.pos_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(d_model, d_ff, dropout, activation)
        self.down_fc = nn.Linear(patch_size * d_embed, d_model)
        self.up_fc = nn.Linear(d_model, patch_size * d_embed)


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        res_outer = x
        x = self.start_fc(x.transpose(-1, -2)).transpose(-1, -2) # B, C, L, D
        # Patch Division
        x = rearrange(x, 'B C (F P) D -> (B C) F (P D)', P=self.patch_size) # BC, F, PD
        x = self.down_fc(x) # BC, F, D
        x = self.pos_drop(x + self.pos_embed(x)) # BC, F, D
        res_inner = x # BC, F, D
        x, attn = self.attention( # BC, F, D
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res_inner + self.dropout(x)) # BC, F, D

        res_inner = x # BC, F, D
        x = self.mlp(x) # BC, F, D
        x = self.norm2(res_inner + self.dropout(x)) # BC, F, D

        x = self.up_fc(x)
        x = rearrange(x, '(B C) F (P D) -> B C (F P) D', C=C, P=self.patch_size)[:, :, :L, :] # B, C, L, D
        x = self.norm3(res_outer + self.dropout(x)) # B, C, L, D

        return x

class TwoStageEncoderLayer(nn.Module):
    def __init__(self, attention, num_features, seq_len, d_embed, n_heads, d_model, d_ff=None, dropout=0.1, activation="relu", individual=False):
        super(TwoStageEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)
        self.down_fc = nn.Linear(seq_len * d_embed, d_model)
        self.up_fc = nn.Linear(d_model, seq_len * d_embed)
        self.pos_embed = PositionalEmbedding(d_model, max_len=num_features)
        self.pos_drop = nn.Dropout(dropout)
        self.router = PeriodGuidedMultiScaleRouter(5, num_features, seq_len)
        self.patch_sizes = self.router.patch_sizes
        self.multi_scale_periodic_attention_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.multi_scale_periodic_attention_layers.append(
                PeriodicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 3, attention_dropout=dropout), 
                        d_model, n_heads),
                    seq_len,
                    patch_size,
                    d_embed,
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        res_outer = x
        x = self.down_fc(x.flatten(start_dim=2)) # B, C, D
        x = self.pos_drop(x + self.pos_embed(x)) # B, C, D
        res_inner = x
        x, attn = self.attention( # B, C, D
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res_inner + self.dropout(x))

        res_inner = x
        x = self.mlp(x)
        x = self.norm2(res_inner + self.dropout(x))

        x = self.up_fc(x).reshape(B, C, L, D) # B, C, L, D
        x = self.norm3(res_outer + self.dropout(x)) # B, C, L, D

        gates = self.router(x, self.training)

        xs = dispatcher(x, gates)
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.multi_scale_periodic_attention_layers[i](xs[i], attn_mask=attn_mask, tau=tau, delta=delta)
        x = aggregate(xs, gates)

        return x, attn


class TwoStageEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(TwoStageEncoder, self).__init__()
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
    

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.mask_ratio = configs.mask_ratio
        self.pretrain = configs.pretrain
        self.individual = configs.individual

        self.enc_embedding = DataEmbedding_wo_pos(1, 8, configs.embed, configs.freq, configs.dropout) if configs.individual else DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.enc_in*8, configs.freq, configs.dropout)

        # self.patch_sizes = self.router.patch_sizes

        self.encoder = TwoStageEncoder(
            [
                TwoStageEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    # self.patch_sizes,
                    configs.enc_in,
                    configs.seq_len,
                    8,
                    configs.n_heads,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    individual=configs.individual
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(8)
        )
        
        self.projection = nn.Linear(configs.seq_len*8, configs.pred_len, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        B, L, C = x_enc.shape
        x_enc = x_enc.transpose(-1, -2).unsqueeze(-1) if self.individual else x_enc
        x_mark_enc = x_mark_enc.repeat(C, 1, 1).reshape(B, C, L, -1) if self.individual else x_mark_enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out = self.encoder(enc_out)[0]
        # Head
        dec_out = self.projection(enc_out.flatten(start_dim=2)).transpose(-1, -2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
