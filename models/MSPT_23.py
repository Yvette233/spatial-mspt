
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import seaborn as sns

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, top_k=5, d_model=512, dropout=0., sparsity_threshold=0.01, hidden_size_factor=4):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        # get the patch sizes
        self.patch_sizes = self.get_patch_sizes(seq_len)
        # AFNO1D parameters
        self.freq_seq_len = seq_len // 2
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len, self.freq_seq_len * hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * hidden_size_factor, len(self.patch_sizes)))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, len(self.patch_sizes)))
        # Noise parameters
        self.w_noise = nn.Parameter(torch.zeros(seq_len, len(self.patch_sizes)))
        # Patch Embedding parameters
        self.start_fc = nn.Linear(1, d_model)
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_size*d_model, d_model, bias=False))
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len)))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique()
        return patch_sizes
    
    def afno1d_for_peroid_weights(self, x, training=True, noise_epsilon=1e-2):
        # x [B, L, C, D]
        B, L, C, D = x.shape

        x = rearrange(x, 'B L C D -> B C D L') # [B, C, L] 
        # xf = torch.fft.rfft(x, dim=-1, norm='ortho') # [B, C, L//2+1]
        xf = torch.fft.rfft(x, dim=-1) # [B, C, L//2+1]
        xf_no_zero = xf[:, :, 1:] # [B, C, L//2]

        o1_real = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros([B, C, len(self.patch_sizes)], device=x.device)
        o2_imag = torch.zeros([B, C, len(self.patch_sizes)], device=x.device)

        o1_real = F.relu(
            torch.einsum('...i,io->...o', xf_no_zero.real, self.w1[0]) - \
            torch.einsum('...i,io->...o', xf_no_zero.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...i,io->...o', xf_no_zero.imag, self.w1[0]) + \
            torch.einsum('...i,io->...o', xf_no_zero.real, self.w1[1]) + \
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

        xf_no_zero = torch.stack([o2_real, o2_imag], dim=-1) # [B, C, L-1, 2]
        # xf_no_zero = F.softshrink(xf_no_zero, lambd=self.sparsity_threshold) # [B, C, L-1, 2]
        xf_no_zero = torch.view_as_complex(xf_no_zero) # [B, C, L-1]


        weights = torch.abs(xf_no_zero) # [B, C, L-1]
        if training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_weights = weights + (torch.randn_like(weights) * noise_stddev)
            weights = noisy_weights.mean(dim=-2) # [B, L-1]
        else:
            weights = weights.mean(dim=-2) # [B, L-1]
        
        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(weights.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/weights.png')

        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1) # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1) # [B, top_k]

        zeros = torch.zeros_like(weights) # [B, Ps]
        gates = zeros.scatter_(-1, top_indices, top_weights) # [B, Ps]

        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(gates.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/gates.png')

        return gates # [B, Ps]

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
        # start fc
        x = self.start_fc(x.unsqueeze(-1)) # [B, L, C, D]
        # do patching
        x = rearrange(x, 'B L C D -> B C D L') # [B, C, D, L]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, D, L//patch_size, patch_size]
        x = rearrange(x, 'B C D L P -> B C L (P D)') # [B, C, L, P*D]
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x) # [B, C, L, D]
        return self.dropout(x) # [B, C, L, D]

    def forward(self, x):
        B, L, C = x.shape
        gates = self.afno1d_for_peroid_weights(x, self.training) # [B, Ps]
        xs = self.dispatcher(x, gates) # Ps*[B, C, L, D]
        for i, patch_size in enumerate(self.patch_sizes): 
            xs[i] = self.patch_embedding(xs[i], patch_size, i)
        return xs, gates # Ps*[B, C, L, D], [bs, Ps]


class CrossDimensionalPeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, n_vars, d_keys=None,
            d_values=None):
        super(CrossDimensionalPeriodicAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection1 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection1 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection1 = nn.Linear(d_model, d_values * n_heads)
        self.out_projection1 = nn.Linear(d_values * n_heads, d_model)
        self.query_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection2 = nn.Linear(d_model, d_values * n_heads)
        self.out_projection2 = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.n_vars = n_vars

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, C, L, _ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads
        N = self.n_vars

        queries = self.query_projection1(queries)
        keys = self.key_projection1(keys)
        values = self.value_projection1(values)

        # queries = rearrange(queries, '(B H) L D -> B H L D', H=H)
        # keys = rearrange(keys, '(B H) S D -> B H S D', H=H)
        # values = rearrange(values, '(B H) S D -> B H S D', H=H)

        # cross dimensional attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = self.out_projection1(out)

        # inter-periodic attention
        queries = self.query_projection2(out)
        keys = self.key_projection2(out)
        values = self.value_projection2(out)

        queries = rearrange(queries, 'B C L (H D) -> (B C) L H D', H=H)
        keys = rearrange(keys, 'B C S (H D) -> (B C) S H D', H=H)
        values = rearrange(values, 'B C S (H D) -> (B C) S H D', H=H)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = rearrange(out, '(B C) L H D -> B C L (H D)', C=N)
        out = self.out_projection2(out)

        return out, attn


class CrossDimensionalPeriodicEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(CrossDimensionalPeriodicEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        res = x
        x = self.norm1(x)
        x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = res + self.dropout(x)

        res = x
        x = self.norm2(x)
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, -2))))
        x = self.dropout(self.conv2(x).transpose(-1, -2))
        x = res + x

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


class CrossScalePeriodicFeatureAggregator(nn.Module):
    def __init__(self, patch_sizes, seq_len, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(CrossScalePeriodicFeatureAggregator, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(d_model, patch_size*d_model))
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, xs, gates, multiply_by_gates=True):
   
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](xs[i])
            xs[i] = rearrange(xs[i], 'B C L (P D) -> B C (L P) D', P=patch_size)[:,:self.seq_len,:]
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0) 
        _, _expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        gates_exp = gates[_batch_index.flatten()]
        _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)
        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(xs, 0).exp() # [BN L D]
        stitched = torch.cat(xs, 0)
        stitched = self.projection(stitched)
        if multiply_by_gates:
            stitched = torch.einsum("ikh,ik -> ikh", stitched, _nonzero_gates) # [BN L D] [BN L] -> [BN L D]
        zeros = torch.zeros(gates.size(0), xs[-1].size(1), xs[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, _batch_index, stitched.float())
        # # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # # go back to log space
        # combined = combined.log()
        return combined # [B, L, D]
    

class LinearRegressionHead(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0., n_vars=11):
        super(LinearRegressionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(ceil(seq_len / patch_size)*d_model, pred_len))
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)

        
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # L-1*[bs, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(xs[i].flatten(1))) # [bs, L, D] -> [bs, P*D]
            xs[i] = rearrange(xs[i], '(B C) P -> B P C', C=self.n_vars)
        xs = self.cspfa(xs, gates)
        return xs # [bs, P, C]


class DecoderRegressionHead(nn.Module):
    def __init__(self, configs, patch_sizes):
        super(DecoderRegressionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.individual = configs.individual
        self.dropout = nn.Dropout(configs.dropout)
        self.data_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout) if self.individual else DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.decoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
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
                    norm_layer=torch.nn.LayerNorm(configs.d_model),
                )
            )
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, configs.seq_len, configs.d_model)
        if self.individual:
            self.projections = nn.ModuleList()
            for i in range(configs.dec_in):
                self.projections.append(nn.Linear(configs.d_model, configs.c_out))
        else:
            self.projection = nn.Linear(configs.d_model, configs.dec_in)

    
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # L-1*[bs, L, D]
        B, S, C = x_dec.shape
        x_dec = rearrange(x_dec, 'B S (C D) -> (B C) S D', D=1) if self.individual else x_dec
        x_dec = self.data_embedding(x_dec, None) # [bs, S, D]
        dec_outs = []
        for i, patch_size in enumerate(self.patch_sizes):
            dec_out = self.decoders[i](x_dec, xs[i]) # [bs, S, D]
            dec_outs.append(dec_out)
        dec_outs = self.cspfa(dec_outs, gates) # [bs, S, D]
        if self.individual:
            dec_outs = [self.projections[i](dec_out) for i, dec_out in enumerate(dec_outs)]
            dec_outs = torch.cat(dec_outs, dim=-1) # [B, S, C]
        else:
            dec_outs = self.projection(dec_outs) # [B, S, C]
        return dec_outs # [B, S, C]
    

class LinearPretrainHead(nn.Module):
    def __init__(self, patch_sizes, seq_len, d_model, dropout=0., n_vars=11):
        super(LinearPretrainHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(d_model, patch_size))
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)
    
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # L-1*[bs, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(xs[i]))
            xs[i] = rearrange(xs[i], 'B L (P C) -> B (L P) C', P=patch_size)[:,:self.seq_len,:] # [bs, L, C]
        xs = self.cspfa(xs, gates) # [bs, L, C]
        xs = rearrange(xs, '(B C) L D -> B L (C D)', C=self.n_vars)
        return xs # [bs, L, C]


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


        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, top_k=configs.top_k, d_model=configs.d_model, dropout=configs.dropout)
        self.patch_sizes = self.msppe.patch_sizes

        self.encoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                CrossDimensionalPeriodicEncoder(
                   [
                        CrossDimensionalPeriodicEncoderLayer(
                            CrossDimensionalPeriodicAttentionLayer(
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
            )

        if self.pretrain:
            self.head = LinearPretrainHead(self.patch_sizes, self.seq_len, configs.d_model, dropout=configs.dropout, n_vars=configs.enc_in)
        else:    
            self.head = LinearRegressionHead(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout, n_vars=configs.enc_in)

        self.revin = RevIN(configs.enc_in)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        x_enc, _ = self.revin(x_enc, mode='forward')

        B, L, C = x_enc.shape

        # Multi-scale periodic patch embedding 
        xs_enc, gates_enc = self.msppe(x_enc) # Ps*[B, C, L, D], [B, Ps]
        # Encoder and Decoder
        enc_outs = []
        for i, x_enc in enumerate(xs_enc):
            enc_out, attns = self.encoders[i](x_enc) # [B, C, VT, D]
            enc_outs.append(enc_out)

        # Head
        dec_out = self.head(enc_outs, gates_enc, x_dec)

        dec_out = self.revin(dec_out, mode='inverse')

        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
