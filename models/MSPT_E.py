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

import matplotlib.pyplot as plt
import seaborn as sns
                            
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        # stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("ikh,ik -> ikh", stitched, self._nonzero_gates) # [BN L D] [BN L] -> [BN L D]
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()
        # return combined


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None,
                 attention_dropout=0.1, res_attention=False):
        super(Attention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.res_attention = res_attention

    def forward(self, x, prev=None):
        # [B*C, N, D] or [B, N, D]
        B, N, D = x.shape
        H = self.n_heads

        # project the queries, keys and values
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # split the keys, queries and values in multiple heads
        queries = rearrange(queries, 'B N (H D) -> B H N D', H=H)
        keys = rearrange(keys, 'B N (H D) -> B H D N', H=H)
        values = rearrange(values, 'B N (H D) -> B H N D', H=H)

        # compute the unnormalized attention scores
        scale = 1. / sqrt(D // H)
        # print(queries.shape, keys.shape)
        attn_scores = torch.matmul(queries, keys) * scale # [B*C, H, N, N] or [B, H, N, N]

        # Add pre-softmax attention scores from the previous layer (optional)
        if self.res_attention and prev is not None:
            attn_scores = attn_scores + prev

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*C, H, N, N] or [B, H, N, N]
        attn_weights = self.attention_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, values)  # [B*C, H, N, D//H] or [B, H, N, D//H]

        # concatenate the heads
        output = rearrange(output, 'B H N D -> B N (H D)') # [B*C, N, D] or [B, N, D]

        # project the output 
        output = self.out_projection(output) # [B*C, N, D] or [B, N, D]

        return output, attn_weights


class MLP(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(MLP, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, fft, d_model, dropout=0.1, pre_norm=False):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.ffn = fft
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(self, x, prev=None):
        # [B*C, N, D] or [B, N, D]
        if self.pre_norm:
            res = x 
            x = self.norm1(x)
        new_x, attn = self.attention(x, prev) # [B*C, N, D] or [B, N, D]
        if self.pre_norm:
            x = res + self.dropout(new_x)
        else:
            x = self.norm1(x + self.dropout(new_x))

        res = x
        if self.pre_norm:
            x = self.norm2(x)
        x = self.ffn(x) # [B*C, N, D] or [B, N, D]
        if self.pre_norm:
            x = res + x
        else:
            x = self.norm2(res + x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer, patch_size, num_patchs, n_vars, d_model, pos_embed_dropout=0.1, learned_pos_embed=True, individual=False):
        super(Encoder, self).__init__()
        self.patch_size = patch_size
        self.token_embed = nn.Linear(patch_size, d_model) if individual else nn.Linear(patch_size*n_vars, d_model)
        self.pos_embed = positional_encoding(pe='zeros', learn_pe=True, q_len=num_patchs, d_model=d_model) if learned_pos_embed else positional_encoding(pe='sincos', learn_pe=False, q_len=num_patchs, d_model=d_model)
        self.pos_embed_dropout = nn.Dropout(pos_embed_dropout)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer
        self.out_proj = nn.Linear(d_model, patch_size*d_model)
        
    def patchify_embedding(self, x):
        # [B*C, L, 1] or [B, L, C]
        # switch to [B*C, 1, L] or [B, C, L]
        L = x.shape[1]
        x = rearrange(x, 'B L C -> B C L')
        # padding
        x = F.pad(x, (0, self.patch_size - L % self.patch_size), mode='replicate') # [B*C, 1, L] or [B, C, L]
        # unfold
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size) # [B*C, 1, N, P] or [B, C, N, P]
        # switch to [B*C, N, P] or [B, N, P*C]
        x = rearrange(x, 'B C N P -> B N (P C)')
        # embed 
        x = self.token_embed(x) # [B*C, N, D] or [B, N, D]
        # add position embedding
        x = self.pos_embed_dropout(x + self.pos_embed)
        return x
    
    def unpacthify(self, x):
        # [B*C, N, D] or [B, N, D]
        # project to [B*C, N, P*D] or [B, N, P*D]
        x = self.out_proj(x)
        # switch to [B*C, N, P, D] or [B, N, P, D]
        x = rearrange(x, 'B N (P D) -> B (N P) D', P=self.patch_size)
        return x

    def forward(self, x):
        # [B*C, L, 1] or [B, L, C]
        # patchify
        L = x.shape[1]
        x = self.patchify_embedding(x) # [B*C, N, D] or [B, N, D]

        attns = []
        for encoder_layer in self.encoder_layers:
            prev = attns[-1] if len(attns) > 0 else None
            x, attn = encoder_layer(x, prev) # [B*C, N, D] or [B, N, D]
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x) # [B*C, N, D] or [B, N, D]
        
        x = self.unpacthify(x)[:, :L, :] # [B*C, L, D] or [B, L, D]

        return x, attns
        

class EncoderStack(nn.Module):
    def __init__(self, configs):
        super(EncoderStack, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.e_layer = configs.e_layers

        patch_sizes = self.get_patch_sizes(configs.seq_len, exclude_zero=True)
        self.num_patch_sizes = len(patch_sizes)

        self.start_linear = nn.Linear(in_features=14, out_features=1)
        self.w_noise = nn.Parameter(torch.zeros(configs.seq_len, self.num_patch_sizes), requires_grad=True)

        self.encoders = nn.ModuleList()
        for patch_size in patch_sizes:
            num_patchs = int(self.seq_len / patch_size) + 1
            self.encoders.append(
                Encoder(
                    [
                        EncoderLayer(
                            Attention(
                                configs.d_model,
                                configs.n_heads,        
                                attention_dropout=configs.dropout,
                                res_attention=False
                            ),
                            MLP(configs.d_model, configs.d_ff, dropout=configs.dropout),
                            configs.d_model,
                            dropout=configs.dropout,
                            pre_norm=False
                        ) for l in range(self.e_layer)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model),
                    patch_size=patch_size,
                    num_patchs=num_patchs,
                    n_vars=configs.enc_in,
                    d_model=configs.d_model,
                    pos_embed_dropout=configs.dropout,
                    learned_pos_embed=True,
                    individual=configs.individual
                )
            )

    def get_patch_sizes(self, seq_len, exclude_zero=True):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:] if exclude_zero else 1 / torch.fft.rfftfreq(seq_len)
        patch_sizes = peroid_list.ceil().int().unique()
        return patch_sizes

    def fft_for_peroid(self, x, exclude_zero=True):
        # [bs, L, D]
        # transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=1)
        # compute the amplitude
        amplitude_list = abs(x_freq).mean(-1)[:, 1:] if exclude_zero else abs(x_freq).mean(-1)
        # get the frequency list
        frequency_list = torch.fft.rfftfreq(x.shape[1], 1)[1:] if exclude_zero else torch.fft.rfftfreq(x.shape[1], 1)
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / frequency_list
        return peroid_list, amplitude_list

    def groups_by_period(self, peroid_list, amplitude_list):
        # peroid_list [L] amplitude_list [bs, L]
        int_peroid_list = peroid_list.ceil().int()
        groups_period, indices = torch.unique(int_peroid_list, return_inverse=True)
        indices = indices.unsqueeze(0).expand(amplitude_list.shape[0], -1).to(amplitude_list.device)
        groups_amplitude = torch.zeros(amplitude_list.shape[0], groups_period.shape[0], device=amplitude_list.device)
        groups_amplitude = groups_amplitude.scatter_add(1, indices, amplitude_list)
        # groups_amplitude = torch.bincount(indices, weights=amplitude_list)
        return groups_period, groups_amplitude

    def top_k_gating(self, x, groups_amplitude, train, noise_epsilon=1e-2):
        # x = self.start_linear(x).squeeze(-1)
        if x.shape[-1] != 1:
            x = self.start_linear(x)
        x = x.squeeze(-1)

        clean_logits = groups_amplitude
        # if train:
        #     raw_noise_stddev = x @ self.w_noise
        #     noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
        #     noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        #     logits = noisy_logits
        # else:
        logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(self.k, dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits.softmax(1)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates

    def forward(self, x):
        # [B, L, C] or [B*C, L, 1]
        period_list, amplitude_list = self.fft_for_peroid(x) # [L], [B, L] or [B*C, L]
        groups_period, groups_amplitude = self.groups_by_period(period_list, amplitude_list) # [num_period], [B, num_period] or [B*C, num_period]
        groups_amplitude = groups_amplitude.softmax(dim=-1) # [B, num_period] or [B*C, num_period]
        gates = self.top_k_gating(x, groups_amplitude, train=self.training) # [B, num_period] or [B*C, num_period]
        
        # visual gates
        plt.figure(figsize=(10, 10))
        sns.heatmap(gates.cpu().detach().numpy(), cmap='viridis')
        plt.savefig('/root/MSPT/test_visuals/gates.png')

        dispatcher = SparseDispatcher(self.num_patch_sizes, gates)
        encoders_input = dispatcher.dispatch(x) # [B*C, L, 1]*num_period or [B, L, C]*num_period
        encoders_output = [self.encoders[i](encoders_input[i])[0] for i in range(self.num_patch_sizes)] # [B*C, L, D]*num_period or [B, L, D]*num_period
        output = dispatcher.combine(encoders_output) # [B*C, L, D] or [B, L, D]
        return output # [B*C, L, D] or [B, L, D]
    

class PredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.individual = configs.individual
        self.n_vars = configs.enc_in
        self.channels_fusion = nn.Linear(configs.d_model * configs.enc_in, configs.d_model)
        if self.individual:
            self.decoder = nn.ModuleList()
            for i in range(self.n_vars):
                self.decoder.append(
                    Decoder(
                        [
                            DecoderLayer(
                                AttentionLayer(
                                    FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                                    configs.d_model, configs.n_heads),
                                AttentionLayer(
                                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                                    configs.d_model, configs.n_heads),
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout,
                                activation=configs.activation,
                            ) for l in range(configs.d_layers)
                        ],
                        norm_layer=nn.LayerNorm(configs.d_model),
                        projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
                    )
                )
        else:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for l in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.enc_in, bias=True)
            )
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x: [B*C, L, D] or [B, L, D] cross: [B*C, S, D] or [B, S, D]
        # dec_out = self.decoder(x, cross, x_mask, cross_mask, tau, delta) # [B*C, L, D] or [B, L, D]
        # dec_out = self.projection(dec_out) # [B*C, L, 1] or [B, L, C]
        if self.individual:
            x = rearrange(x, '(B C) L D -> B C L D', C=self.n_vars) # [B, C, L, D]
            cross = rearrange(cross, '(B C) S D -> B C S D', C=self.n_vars) # [B, C, S, D]
            dec_outs = []
            for i in range(self.n_vars):
                temp = self.decoder[i](x[:,i,:,:], cross[:,i,:,:], x_mask, cross_mask, tau, delta) # [B, L, 1]
                dec_outs.append(temp)
            dec_out = torch.cat(dec_outs, dim=-1) # [B, L, C]
        else:
            dec_out = self.decoder(x, cross, x_mask, cross_mask, tau, delta)
        return dec_out
    
class LinearProbeHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.individual = configs.individual
        self.n_vars = configs.enc_in
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(configs.seq_len * configs.d_model, configs.pred_len))
                self.dropouts.append(nn.Dropout(configs.dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(configs.seq_len * configs.d_model, configs.pred_len)
            self.dropout = nn.Dropout(configs.dropout)
            
    def forward(self, x):
        # [B*C, L, D] or [B, L, D]                           
        if self.individual:
            x = rearrange(x, '(B C) N D -> B C N D', C=self.n_vars) # [B, C, L, D]
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])  # [B, L*D]
                z = self.linears[i](z)  # [B, S]
                z = self.dropouts[i](z) # [B, S]
                x_out.append(z)
            x = torch.stack(x_out, dim=-1) # x: [B, S, C]
        else:
            x = self.flatten(x)
            x = self.linear(x).unsqueeze(-1).expand(-1, -1, self.n_vars) # [B, S, C]
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

        self.revin = RevIN(configs.enc_in, affine=False, subtract_last=False)

        if self.individual:
            self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder_stack = EncoderStack(configs) 

        # self.head = PredictionHead(configs)
        self.head = LinearProbeHead(configs)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc [B, L, C] x_mark_enc [B, L, M] x_dec [B, S, C] x_mark_dec [B, S, M]

        # revin
        x_enc, x_dec = self.revin(x_enc, 'forward', x_dec)

        
        # individual
        if self.individual:
            n_vars = x_enc.shape[-1]
            x_enc = x_enc.unsqueeze(-1) # [B, L, C, 1]
            x_enc = rearrange(x_enc, 'B L C 1 -> (B C) L 1') # [B*C, L, 1]
            x_mark_enc = repeat(x_mark_enc, 'B L M -> (B C) L M', C=n_vars) # [B*C, L, M]

            x_dec = x_dec.unsqueeze(-1)
            x_dec = rearrange(x_dec, 'B S C 1 -> (B C) S 1')
            x_mark_dec = repeat(x_mark_dec, 'B S M -> (B C) S M', C=n_vars)
        
        enc_out = self.encoder_stack(x_enc) # [B*C, L, D] or [B, L, D]

        dec_out = self.dec_embedding(x_dec, None)
    
        # dec_out = self.head(dec_out, enc_out)
        dec_out = self.head(enc_out)

        # print(dec_out.shape)
        dec_out = self.revin(dec_out, 'inverse')

        return dec_out[:, -self.pred_len:, :]