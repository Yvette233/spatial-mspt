import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, PositionalEmbedding2D, TemporalEmbedding, TimeFeatureEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention

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

class PeriodGuidedMultiScaleEmbeding(nn.Module):
    def __init__(self, num_features, seq_len, d_embed, embed_type, freq, dropout=0.1, activation="relu", mlp_ratio=4., individual=False):
        super(PeriodGuidedMultiScaleEmbeding, self).__init__()
        d_ff = d_embed * mlp_ratio
        self.individual = individual
        # GET Patch sizes corresponding to the period
        self.patch_sizes = self.get_patch_sizes(seq_len)
        # GET padding length for each patch size
        self.padding_lens = [ceil(seq_len / patch_size) * patch_size - seq_len for patch_size in self.patch_sizes]
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
        # Embedding
        self.value_embedding = nn.Linear(1, d_embed, bias=False) if individual else nn.Linear(num_features, d_embed, bias=False)
        self.temporal_embedding = TemporalEmbedding(d_model=d_embed, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_embed, embed_type=embed_type, freq=freq)
        self.embed_dropout = nn.Dropout(dropout)
        # self.position_embedding = PositionalEmbedding2D(d_model, num_features, seq_len + max(self.padding_lens)) if individual else PositionalEmbedding(d_model, seq_len + max(self.padding_lens))
        self.position_embedding = PositionalEmbedding2D(d_embed, num_features, ceil(seq_len / min(self.patch_sizes)) * min(self.patch_sizes)) if individual else PositionalEmbedding(d_embed, seq_len + ceil(seq_len / min(self.patch_sizes)) * min(self.patch_sizes))
        self.pos_dropout = nn.Dropout(dropout)
        # Start projections
        self.projections = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.projections.append(nn.Linear(seq_len, ceil(seq_len / patch_size) * patch_size))
        # MLPs
        self.mlps = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.mlps.append(MLP(patch_size*d_embed, patch_size*d_ff, dropout, activation))
        
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy()[::-1]
        # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()
        return patch_sizes
    
    def afno1d_for_peroid_gates(self, x, training, noise_epsilon=1e-2):
        x = self.start_fc(x).squeeze(-1) # B, L, 1

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

    def forward(self, x, x_mark=None):
        B, L, C = x.shape
        # Gate
        gates = self.afno1d_for_peroid_gates(x, self.training)
        
        if self.individual:
            x = rearrange(x, 'B L C -> B C L').unsqueeze(-1)
        # embedding + pos
        if x_mark is None:
            x_embed = self.embed_dropout(self.value_embedding(x))
        else:
            x_embed = self.embed_dropout(self.value_embedding(x) + self.temporal_embedding(x_mark))
        # Dispatcher
        xs = self.dispatcher(x_embed, gates)
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.projections[i](xs[i].transpose(1, 2)).transpose(1, 2)
            xs[i] = rearrange(xs[i], 'B C (F P) D -> B C F (P D)', P=patch_size) if self.individual else rearrange(xs[i], 'B (F P) D -> B F (P D)', P=patch_size)
            xs[i] = self.pos_dropout(xs[i] + self.position_embedding(xs[i]))
            xs[i] = self.mlps[i](xs[i])
        return xs, gates
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_in, d_hid, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_out = d_in

        d_keys = d_keys or (d_hid // n_heads)
        d_values = d_values or (d_hid // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_in, d_keys * n_heads)
        self.key_projection = nn.Linear(d_in, d_keys * n_heads)
        self.value_projection = nn.Linear(d_in, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_out)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = rearrange(queries, 'B L (H D) -> B L H D', H=H)
        keys = rearrange(keys, 'B S (H D) -> B S H D', H=H)
        values = rearrange(values, 'B S (H D) -> B S H D', H=H)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = rearrange(out, 'B L H D -> B L (H D)')

        return self.out_projection(out), attn

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
        B, C, F, D = x.shape
        x = rearrange(x, 'B C F D -> B C (F D)')
        res = x
        x, attn1 = self.cross_dimensional_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res + self.dropout(x))

        res = x
        x = self.cross_dimensional_mlp(x)
        x = self.norm2(res + self.dropout(x))

        x = rearrange(x, 'B C (F D) -> (B C) F D', F=F)
        res = x
        x, attn2 = self.inter_periodic_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm3(res + self.dropout(x))

        res = x
        x = self.inter_periodic_mlp(x)
        x = self.norm4(res + self.dropout(x))

        x = rearrange(x, '(B C) F D -> B C F D', C=C)

        return x, attn2


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
    def __init__(self):
        super(CrossScalePeriodicFeatureAggregator, self).__init__()

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
            xs[i] = self.linears[i](self.dropout(xs[i].flatten(start_dim=2))) # [B, C, F, D] -> [B, C, P]
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

        self.pgmse = PeriodGuidedMultiScaleEmbeding(configs.enc_in, self.seq_len, d_embed=16, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout, activation=configs.activation, mlp_ratio=configs.mlp_ratio, individual=self.individual)

        self.patch_sizes = self.pgmse.patch_sizes

        self.encoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                CrossDimensionalPeriodicEncoder(
                   [
                        CrossDimensionalPeriodicEncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), patch_size*16, configs.d_model, configs.n_heads),
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), patch_size*16, configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )
        

        
        self.head = LinearPredictionHead(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-scale periodic patch embedding 
        xs_enc, gates = self.pgmse(x_enc) # Ps*[B, C, F, D], [B, Ps]
        # Encoder and Decoder
        enc_outs = []
        for i, x_enc in enumerate(xs_enc):
            enc_out, attns = self.encoders[i](x_enc) # [B, C, varF, D]
            enc_outs.append(enc_out)
        # Head
        dec_out = self.head(enc_outs, gates, x_dec, x_mark_dec)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
