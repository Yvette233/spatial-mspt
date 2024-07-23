
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, PositionalEmbedding2D
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


class AdaptiveFourierTransformGateLayer(nn.Module):
    def __init__(self, seq_len, top_k, num_features, d_model, d_ff, dropout=0.1, activation="relu", sparsity_threshold=0.01, multi_factor=4):
        super(AdaptiveFourierTransformGateLayer, self).__init__()
        self.top_k = top_k

        self.start_fc = nn.Linear(num_features, 1) # 11->512
        self.mlp = MLP(d_model, d_ff, dropout, activation) # 512->2048
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.patch_sizes = self.get_patch_sizes(seq_len)
        
        self.num_freqs = seq_len // 2
        self.scale = 0.02
        self.multi_factor = multi_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs, self.num_freqs * self.multi_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor))
        # self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor, len(self.patch_sizes)))
        # self.b2 = nn.Parameter(self.scale * torch.randn(2, len(self.patch_sizes)))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor, self.num_freqs))
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
        n_vars = x.size(-1)

        x = self.start_fc(x) # B, L, D
        # x = self.norm(x + self.dropout(self.mlp(x))) # B, L, D

        B, L, D = x.shape
        x = rearrange(x, 'B L D -> B D L') # B, D, L
        xf = torch.fft.rfft(x, dim=-1, norm='ortho') # B, D, L//2+1

        xf_ac = xf[:, :, 1:]
        
        o1_real = torch.zeros([B, D, self.num_freqs * self.multi_factor], device=x.device)
        o1_imag = torch.zeros([B, D, self.num_freqs * self.multi_factor], device=x.device)
        o2_real = torch.zeros([B, D, self.num_freqs], device=x.device)
        o2_imag = torch.zeros([B, D, self.num_freqs], device=x.device)

        o1_real = F.relu(
            torch.einsum('...i,io->...o', xf_ac.real, self.w1[0]) - \
            torch.einsum('...i,io->...o', xf_ac.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...i,io->...o', xf_ac.imag, self.w1[0]) + \
            torch.einsum('...i,io->...o', xf_ac.real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real = (
            torch.einsum('...i,io->...o', o1_real, self.w2[0]) - \
            torch.einsum('...i,io->...o', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = (
            torch.einsum('...i,io->...o', o1_imag, self.w2[0]) + \
            torch.einsum('...i,io->...o', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        xf_ac = torch.stack([o2_real, o2_imag], dim=-1) # B, D, L-1, 2
        xf_ac = torch.view_as_complex(xf_ac)
        xf_ac = torch.abs(xf_ac) # B, D, L-1

        xf_ac = xf_ac.squeeze(dim=1)
        
        clean_logits = xf_ac @ self.w_gate

        # visual gates
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        sns.heatmap(clean_logits.detach().cpu().numpy(), ax=ax[0, 0], cmap='jet')

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
        
        sns.heatmap(noise.detach().cpu().numpy(), ax=ax[0, 1], cmap='jet')
        sns.heatmap(weights.detach().cpu().numpy(), ax=ax[1, 0], cmap='jet')
        sns.heatmap(gates.detach().cpu().numpy(), ax=ax[1, 1], cmap='jet')

        plt.savefig('/root/MSPT/test_visuals/sets.png')

        return gates

# class AdaptiveFourierTransformGateLayer(nn.Module):
#     def __init__(self, seq_len, top_k, num_features, d_model, d_ff, dropout=0.1, activation="relu", sparsity_threshold=0.01, multi_factor=4):
#         super(AdaptiveFourierTransformGateLayer, self).__init__()
#         self.top_k = top_k
#         self.patch_sizes = self.get_patch_sizes(seq_len)

#         self.start_fc = nn.Linear(num_features, 1) # 11->512

#         self.num_freqs = seq_len // 2 # 182
#         self.num_low_freqs = int(sqrt(self.num_freqs)) # 13
#         self.num_high_freqs = self.num_freqs - self.num_low_freqs # 169
#         self.num_large_patches = self.num_low_freqs
#         self.num_small_patches = len(self.patch_sizes) - self.num_large_patches
#         # print(self.num_low_freqs, self.num_high_freqs)
#         self.scale = 0.02
#         self.multi_factor = multi_factor
#         self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs, self.num_freqs * self.multi_factor))
#         self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor))
#         self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * self.multi_factor, len(self.patch_sizes)))
#         self.b2 = nn.Parameter(self.scale * torch.randn(2, len(self.patch_sizes)))
#         # self.w1_low = nn.Parameter(self.scale * torch.randn(2, self.num_low_freqs, self.num_low_freqs * self.multi_factor))
#         # self.b1_low = nn.Parameter(self.scale * torch.randn(2, self.num_low_freqs * self.multi_factor))
#         # self.w2_low = nn.Parameter(self.scale * torch.randn(2, self.num_low_freqs * self.multi_factor, self.num_large_patches))
#         # self.b2_low = nn.Parameter(self.scale * torch.randn(2, self.num_large_patches))

#         # self.w1_high = nn.Parameter(self.scale * torch.randn(2, self.num_high_freqs, self.num_high_freqs * self.multi_factor))
#         # self.b1_high = nn.Parameter(self.scale * torch.randn(2, self.num_high_freqs * self.multi_factor))
#         # self.w2_high = nn.Parameter(self.scale * torch.randn(2, self.num_high_freqs * self.multi_factor, self.num_small_patches))
#         # self.b2_high = nn.Parameter(self.scale * torch.randn(2, self.num_small_patches))

#         self.squeeze_fc = nn.Linear(num_features, 1)
#         self.w_gate = nn.Parameter(torch.zeros(len(self.patch_sizes), len(self.patch_sizes)))
#         self.w_noise = nn.Parameter(torch.zeros(len(self.patch_sizes), len(self.patch_sizes)))

#     def get_patch_sizes(self, seq_len):
#         # get the period list, first element is inf if exclude_zero is False
#         peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
#         patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy()[::-1]
#         # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()
#         return patch_sizes

#     def forward(self, x, training, noise_epsilon=1e-2):
#         n_vars = x.size(-1)

#         x = self.start_fc(x) # B, L, D

#         B, L, D = x.shape
#         x = rearrange(x, 'B L D -> B D L') # B, D, L
#         xf = torch.fft.rfft(x, dim=-1) # B, D, L//2+1

#         xf_ac = xf[:, :, 1:]

#         xf_low = xf[:, :, 1:1+self.num_low_freqs]
#         xf_high = xf[:, :, 1+self.num_low_freqs:]

#         o1_real = torch.zeros([B, D, self.num_freqs * self.multi_factor], device=x.device)
#         o1_imag = torch.zeros([B, D, self.num_freqs * self.multi_factor], device=x.device)
#         o2_real = torch.zeros([B, D, len(self.patch_sizes)], device=x.device)
#         o2_imag = torch.zeros([B, D, len(self.patch_sizes)], device=x.device)
        
#         # o1_real = torch.zeros([B, D, self.num_low_freqs * self.multi_factor], device=x.device)
#         # o1_imag = torch.zeros([B, D, self.num_low_freqs * self.multi_factor], device=x.device)
#         # o2_real = torch.zeros([B, D, self.num_large_patches], device=x.device)
#         # o2_imag = torch.zeros([B, D, self.num_large_patches], device=x.device)

#         # o3_real = torch.zeros([B, D, self.num_high_freqs * self.multi_factor], device=x.device)
#         # o3_imag = torch.zeros([B, D, self.num_high_freqs * self.multi_factor], device=x.device)
#         # o4_real = torch.zeros([B, D, self.num_small_patches], device=x.device)
#         # o4_imag = torch.zeros([B, D, self.num_small_patches], device=x.device)

#         o1_real = F.relu(
#             torch.einsum('...i,io->...o', xf_ac.real, self.w1[0]) - \
#             torch.einsum('...i,io->...o', xf_ac.imag, self.w1[1]) + \
#             self.b1[0]
#         )

#         o1_imag = F.relu(
#             torch.einsum('...i,io->...o', xf_ac.imag, self.w1[0]) + \
#             torch.einsum('...i,io->...o', xf_ac.real, self.w1[1]) + \
#             self.b1[1]
#         )

#         o2_real = (
#             torch.einsum('...i,io->...o', o1_real, self.w2[0]) - \
#             torch.einsum('...i,io->...o', o1_imag, self.w2[1]) + \
#             self.b2[0]
#         )

#         o2_imag = (
#             torch.einsum('...i,io->...o', o1_imag, self.w2[0]) + \
#             torch.einsum('...i,io->...o', o1_real, self.w2[1]) + \
#             self.b2[1]
#         )

#         clean_logits = torch.stack([o2_real, o2_imag], dim=-1) # B, D, L-1, 2
#         clean_logits = torch.view_as_complex(clean_logits)
#         clean_logits = torch.abs(clean_logits).mean(1) # B, D, L-1

#         # o1_real = F.gelu(
#         #     torch.einsum('...i,io->...o', xf_low.real, self.w1_low[0]) - \
#         #     torch.einsum('...i,io->...o', xf_low.imag, self.w1_low[1]) + \
#         #     self.b1_low[0]
#         # )

#         # o1_imag = F.gelu(
#         #     torch.einsum('...i,io->...o', xf_low.imag, self.w1_low[0]) + \
#         #     torch.einsum('...i,io->...o', xf_low.real, self.w1_low[1]) + \
#         #     self.b1_low[1]
#         # )

#         # o2_real = (
#         #     torch.einsum('...i,io->...o', o1_real, self.w2_low[0]) - \
#         #     torch.einsum('...i,io->...o', o1_imag, self.w2_low[1]) + \
#         #     self.b2_low[0]
#         # )

#         # o2_imag = (
#         #     torch.einsum('...i,io->...o', o1_imag, self.w2_low[0]) + \
#         #     torch.einsum('...i,io->...o', o1_real, self.w2_low[1]) + \
#         #     self.b2_low[1]
#         # )

#         # clean_logits_low = torch.stack([o2_real, o2_imag], dim=-1) # B, D, L-1, 2
#         # clean_logits_low = torch.view_as_complex(clean_logits_low)
#         # clean_logits_low = torch.abs(clean_logits_low).mean(1) # B, D, L-1

#         # o3_real = F.gelu(
#         #     torch.einsum('...i,io->...o', xf_high.real, self.w1_high[0]) - \
#         #     torch.einsum('...i,io->...o', xf_high.imag, self.w1_high[1]) + \
#         #     self.b1_high[0]
#         # )

#         # o3_imag = F.gelu(
#         #     torch.einsum('...i,io->...o', xf_high.imag, self.w1_high[0]) + \
#         #     torch.einsum('...i,io->...o', xf_high.real, self.w1_high[1]) + \
#         #     self.b1_high[1]
#         # )

#         # o4_real = (
#         #     torch.einsum('...i,io->...o', o3_real, self.w2_high[0]) - \
#         #     torch.einsum('...i,io->...o', o3_imag, self.w2_high[1]) + \
#         #     self.b2_high[0]
#         # )

#         # o4_imag = (
#         #     torch.einsum('...i,io->...o', o3_imag, self.w2_high[0]) + \
#         #     torch.einsum('...i,io->...o', o3_real, self.w2_high[1]) + \
#         #     self.b2_high[1]
#         # )

#         # clean_logits_high = torch.stack([o4_real, o4_imag], dim=-1) # B, D, L-1, 2
#         # clean_logits_high = torch.view_as_complex(clean_logits_high)
#         # clean_logits_high = torch.abs(clean_logits_high).mean(1) # B, D, L-1

#         clean_logits_low = clean_logits[:, :self.num_low_freqs]
#         clean_logits_high = clean_logits[:, self.num_low_freqs:]

#         top_weights_low, top_indices_low = torch.topk(clean_logits_low, 2, dim=-1) # [B, top_k]
#         top_weights_high, top_indices_high = torch.topk(clean_logits_high, 3, dim=-1) # [B, top_k]
#         top_weights = F.softmax(torch.cat([top_weights_low, top_weights_high], dim=-1), dim=-1)

#         top_weights_low = top_weights[:, :2]
#         # top_weights_low = F.softmax(top_weights_low, dim=-1) # [B, top_k]
#         zeros_low = torch.zeros_like(clean_logits_low) # [B, Ps]
#         gates_low = zeros_low.scatter_(-1, top_indices_low, top_weights_low) # [B, Ps]

#         top_weights_high = top_weights[:, 2:]
#         # print(top_weights_low.shape, top_weights_high.shape)
#         # top_weights_high = F.softmax(top_weights_high, dim=-1) # [B, top_k]
#         zeros_high = torch.zeros_like(clean_logits_high) # [B, Ps]
#         gates_high = zeros_high.scatter_(-1, top_indices_high, top_weights_high) # [B, Ps]

#         gates = torch.cat([gates_low, gates_high], dim=-1)

#         # visual gates
#         fig, ax = plt.subplots(2, 2, figsize=(10, 10))

#         # sns.heatmap(clean_logits_low.detach().cpu().numpy(), ax=ax[0, 0])

#         # sns.heatmap(clean_logits_high.detach().cpu().numpy(), ax=ax[0, 1])
#         # # sns.heatmap(weights.detach().cpu().numpy(), ax=ax[1, 0])
#         # sns.heatmap(gates.detach().cpu().numpy(), ax=ax[1, 1])

#         # plt.savefig('/root/MSPT/test_visuals/sets.png')

#         return gates

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model=512, dropout=0., num_variates=11, individual=True):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        # get the patch sizes
        self.patch_sizes = self.get_patch_sizes(seq_len)        
        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        # self.position_embeddings = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_size, d_model, bias=False) if individual else nn.Linear(patch_size*num_variates, d_model, bias=False))
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len)))
            # self.position_embeddings.append(PositionalEmbedding2D(patch_size*8, num_variates, 256))
        self.position_embedding = PositionalEmbedding2D(d_model, num_variates, 256)
        self.dropout = nn.Dropout(dropout)
    
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy()[::-1]
        # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()
        return patch_sizes

    def dispatcher(self, x, gates):
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # get according batch index for each expert
        _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        _part_sizes = (gates > 0).sum(0).tolist()
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        xs = x[_batch_index].squeeze(1)
        return list(torch.split(xs, _part_sizes, dim=0))
    
    def patch_embedding(self, x, patch_size, index_of_patch):
        B, L, C = x.shape
        # do patching
        x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, L//patch_size, patch_size]
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x) # [B, C, L, D]
        return self.dropout(x)

    def forward(self, x, gates):
        B, L, C = x.shape
        xs = self.dispatcher(x, gates)
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.patch_embedding(xs[i], patch_size, i)
        return xs


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
        # x = rearrange(x, 'B C L D -> (B L) C D')
        # res = x
        # x, attn = self.cross_dimensional_attention(
        #     x, x, x,
        #     attn_mask=attn_mask,
        #     tau=tau, delta=delta
        # )
        # x = self.norm1(res + self.dropout(x))

        # res = x
        # x = self.cross_dimensional_mlp(x)
        # x = self.norm2(res + self.dropout(x))

        x = rearrange(x, 'B C L D -> (B C) L D', L=L)
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
    

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = rearrange(x, 'B C L D -> (B C) L D')
        cross = rearrange(cross, 'B C L D -> (B C) L D')

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)
    

class Decoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x [B, C, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class CrossScalePeriodicFeatureAggregator(nn.Module):
    def __init__(self, patch_sizes, seq_len, d_model):
        super(CrossScalePeriodicFeatureAggregator, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.d_model = d_model
       

    def forward(self, xs, gates, multiply_by_gates=True):
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
            stitched = torch.einsum("ikh,ik -> ikh", stitched, _nonzero_gates) # [BN L D] [BN L] -> [BN L D]
        zeros = torch.zeros(gates.size(0), xs[-1].size(1), xs[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, _batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # go back to log space
        combined = combined.log()
        return combined # [B, L, D]
    

class DecoderPredictionHead(nn.Module):
    def __init__(self, patch_sizes, configs):
        super(DecoderPredictionHead, self).__init__()
        self.patch_sizes = patch_sizes

        self.data_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.decoders = nn.ModuleList()
        for patch_size in patch_sizes:
            self.decoders.append(
                Decoder(
                    [
                        DecoderLayer(
                            AttentionLayer(
                                FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.d_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )
        
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, configs.seq_len, configs.d_model)

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # Ps*[B, C, L, D]
        x_dec = rearrange(x_dec, 'B L D -> B D L ()')
        dec_out = self.data_embedding(x_dec, x_mark_dec)
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i], _ = self.decoders[i](dec_out, xs[i])
        dec_out = self.cspfa(xs, gates)
        dec_out = self.projection(dec_out)

        return dec_out


class LinearPredictionHead(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(ceil(seq_len / patch_size)*d_model, pred_len))
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)
        
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # Ps*[B, C, L, D]

        # visual gates
        # fig, ax = plt.subplots(9, 4, figsize=(80, 40))

        # sns.heatmap(clean_logits_low.detach().cpu().numpy(), ax=ax[0, 0])

        # sns.heatmap(clean_logits_high.detach().cpu().numpy(), ax=ax[0, 1])
        # # sns.heatmap(weights.detach().cpu().numpy(), ax=ax[1, 0])
        # sns.heatmap(gates.detach().cpu().numpy(), ax=ax[1, 1])

        # plt.savefig('/root/MSPT/test_visuals/sets.png')
        for i, patch_size in enumerate(self.patch_sizes):
            # pos_x = i // 4
            # pos_y = i % 4
            xs[i] = self.linears[i](self.dropout(xs[i].flatten(start_dim=2))) # [B, C, L, D] -> [B, C, P]
            xs[i] = rearrange(xs[i], 'B C P -> B P C') # [B, P, C]
        #     sns.lineplot(data=xs[i].mean(0)[:, -1].detach().cpu().numpy(), ax=ax[pos_x, pos_y])
        # plt.savefig('/root/MSPT/test_visuals/xs.png')
        xs = self.cspfa(xs, gates)
        return xs # [bs, P, C]
    

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


        self.aft_gate_layer = AdaptiveFourierTransformGateLayer(self.seq_len, top_k=configs.top_k, num_features=configs.enc_in, d_model=configs.d_model, d_ff=configs.d_ff, dropout=configs.dropout, activation=configs.activation)

        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, d_model=configs.d_model, dropout=configs.dropout, num_variates=configs.enc_in, individual=self.individual)
        self.patch_sizes = self.msppe.patch_sizes

        self.encoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                CrossDimensionalPeriodicEncoder(
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
                    norm_layer=nn.LayerNorm(patch_size*8)
                )
            )

        
        self.head = LinearPredictionHead(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout)
        # self.head = DecoderPredictionHead(self.patch_sizes, configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, L, C = x_enc.shape

        gates = self.aft_gate_layer(x_enc, training=self.training)

        # Multi-scale periodic patch embedding 
        xs_enc = self.msppe(x_enc, gates) # Ps*[B, C, L, D], [B, Ps]
        # Encoder and Decoder
        enc_outs = []
        for i, x_enc in enumerate(xs_enc):
            enc_out, attns = self.encoders[i](x_enc) # [B, C, VT, D]
            enc_outs.append(enc_out)

        # Head
        dec_out = self.head(enc_outs, gates, x_dec, x_mark_dec)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
