import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding, positional_encoding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np
from math import sqrt
from einops import rearrange, repeat

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
                                

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    

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
        # stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("ikh,ik -> ikh", stitched, self._nonzero_gates) # [BN L D] [BN L] -> [BN L D]
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        # return combined.log()
        return combined


class DualAttention(nn.Module):
    def __init__(self, n_vars, num_patchs, patch_size, d_model, n_heads, d_keys=None, d_values=None,
                  attention_dropout=0.1, pos_embed_dropout=0.1, learned_pos_embed=False, cascaded=True, 
                  res_attention=False, cross=True):
        super(DualAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_vars = n_vars
        self.num_patchs = num_patchs
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.cross = cross
        
        ## intra_periodic_attention
        if self.cross:
            self.intra_periodic_tokens = nn.Parameter(torch.rand(1, n_vars, num_patchs, 1, 16))
            self.intra_periodic_embedding_weights = nn.Parameter(torch.rand(num_patchs, 16, d_model))
            self.intra_periodic_embedding_biases = nn.Parameter(torch.rand(num_patchs, d_model))

            self.intra_periodic_pos_embed_q = positional_encoding(pe='zeros', learn_pe=True, q_len=1, d_model=d_model) if learned_pos_embed else positional_encoding(pe='sincos', learn_pe=False, q_len=1, d_model=d_model) # [1, 1, 1, d_model]

            self.intra_periodic_pos_embed_kv = positional_encoding(pe='zeros', learn_pe=True, q_len=patch_size, d_model=d_model) if learned_pos_embed else positional_encoding(pe='sincos', learn_pe=False, q_len=patch_size, d_model=d_model)

            self.intra_periodic_pos_embed_dropout = nn.Dropout(pos_embed_dropout)

            self.intra_periodic_out_proj = nn.Linear(num_patchs, num_patchs * patch_size)

        else:
            self.intra_periodic_pos_embed = positional_encoding(pe='zeros', learn_pe=True, q_len=patch_size, d_model=d_model) if learned_pos_embed else positional_encoding(pe='sincos', learn_pe=False, q_len=patch_size, d_model=d_model)

            self.intra_periodic_pos_embed_dropout = nn.Dropout(pos_embed_dropout)

            self.intra_periodic_out_proj = nn.Linear(d_model, d_model)

        self.intra_periodic_q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.intra_periodic_k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.intra_periodic_v_proj = nn.Linear(d_model, d_values * n_heads)

        

        ## inter_periodic_attention
        self.inter_periodic_embed_linear = nn.Linear(patch_size * d_model, patch_size * d_model)
        self.inter_periodic_pos_embed = positional_encoding(pe='zeros', learn_pe=True, q_len=num_patchs, d_model=d_model) if learned_pos_embed else positional_encoding(pe='sincos', learn_pe=False, q_len=num_patchs, d_model=d_model)
        self.inter_periodic_pos_embed_dropout = nn.Dropout(pos_embed_dropout)

        self.inter_periodic_q_proj = nn.Linear(patch_size * d_model, patch_size * d_keys * n_heads)
        self.inter_periodic_k_proj = nn.Linear(patch_size * d_model, patch_size * d_keys * n_heads)
        self.inter_periodic_v_proj = nn.Linear(patch_size * d_model, patch_size * d_values * n_heads)

        self.inter_periodic_out_proj = nn.Linear(patch_size * d_model, patch_size * d_model)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.res_attention = res_attention


    def forward(self, x, prev=None):
        B, N, P, D = x.shape
        H = self.n_heads

        # intra_out_concat = None
        
        n_vars = self.intra_periodic_tokens.shape[1]

        ####intra Attention#####
        if self.cross:
            intra_periodic_tokens = self.intra_periodic_tokens.expand(B//n_vars, -1, -1, -1, -1) # [B, n_vars, num_patchs, patch_size, 16]
            # embed to d_model for each patch
            intra_periodic_tokens = rearrange(intra_periodic_tokens, 'B C N 1 D -> (B C) 1 N D')
            intra_periodic_tokens = torch.bmm(intra_periodic_tokens, self.intra_periodic_embedding_weights) + self.intra_periodic_embedding_biases
            intra_periodic_tokens = rearrange(intra_periodic_tokens, 'BC 1 N D -> BC N 1 D')

            # add position embedding
            intra_periodic_pos_embed_q = repeat(self.intra_periodic_pos_embed_q, '1 D -> BC N 1 D', BC=B, N=N)
            intra_periodic_pos_embed_kv = repeat(self.intra_periodic_pos_embed_kv, 'P D -> BC N P D', BC=B, N=N)
            intra_periodic_tokens = self.intra_periodic_pos_embed_dropout(intra_periodic_tokens + intra_periodic_pos_embed_q)
            cross_x = self.intra_periodic_pos_embed_dropout(x + intra_periodic_pos_embed_kv)

            # project the queries, keys and values
            intra_periodic_queries = self.intra_periodic_q_proj(intra_periodic_tokens)
            intra_periodic_keys = self.intra_periodic_k_proj(cross_x)
            intra_periodic_values = self.intra_periodic_v_proj(cross_x)

            # split the keys, queries and values in multiple heads
            intra_periodic_queries = rearrange(intra_periodic_queries, 'BC N 1 (H D) -> (BC N) H 1 D', H=self.n_heads)
            intra_periodic_keys = rearrange(intra_periodic_keys, 'BC N P (H D) -> (BC N) H D P', H=self.n_heads)
            intra_periodic_values = rearrange(intra_periodic_values, 'BC N P (H D) -> (BC N) H P D', H=self.n_heads)
        
        else:
            # add position embedding
            intra_periodic_pos_embed = repeat(self.intra_periodic_pos_embed, 'P D -> BC N P D', BC=B, N=N)
            cross_x = self.intra_periodic_pos_embed_dropout(x + intra_periodic_pos_embed)

            # project the queries, keys and values
            intra_periodic_queries = self.intra_periodic_q_proj(cross_x)
            intra_periodic_keys = self.intra_periodic_k_proj(cross_x)
            intra_periodic_values = self.intra_periodic_v_proj(cross_x)

            # split the keys, queries and values in multiple heads 
            intra_periodic_queries = rearrange(intra_periodic_queries, 'BC N P (H D) -> (BC N) H P D', H=H)
            intra_periodic_keys = rearrange(intra_periodic_keys, 'BC N P (H D) -> (BC N) H D P', H=H)
            intra_periodic_values = rearrange(intra_periodic_values, 'BC N P (H D) -> (BC N) H P D', H=H)
            

        # compute the unnormalized attention scores
        scale = 1. / sqrt(D // H)
        intra_periodic_attn_scores = torch.matmul(intra_periodic_queries, intra_periodic_keys) * scale # [bs x n_heads x q_len x k_len]

        # # Add pre-softmax attention scores from the previous layer (optional)
        # if self.res_attention and prev is not None:
        #     intra_periodic_attn_scores = intra_periodic_attn_scores + prev

        # normalize the attention weights
        intra_periodic_attn_weights = F.softmax(intra_periodic_attn_scores, dim=-1)  # [bs x n_heads x q_len x k_len]
        intra_periodic_attn_weights = self.attention_dropout(intra_periodic_attn_weights)

        # compute the new values given the attention weights
        intra_periodic_output = torch.matmul(intra_periodic_attn_weights, intra_periodic_values)  # output: [bs x n_heads x q_len x dim]

        # concatenate the heads and project the output back to the patch_size*d_model dimensions
        intra_periodic_output = rearrange(intra_periodic_output, '(BC N) H 1 D -> BC 1 (H D) N', N=N)
        intra_periodic_output = self.intra_periodic_out_proj(intra_periodic_output)
        intra_periodic_output = rearrange(intra_periodic_output, 'BC 1 D (N P) -> BC N P D', N=N)

        ####inter Attention######
        x_reshape = rearrange(x, 'BC N P D -> BC N (P D)') # [b*nvar, patch_num, dim*patch_len]

        x_reshape = self.inter_periodic_embed_linear(x_reshape)
        x_reshape = self.inter_periodic_pos_embed_dropout(x_reshape + self.inter_periodic_pos_embed)

        # project the queries, keys and values
        inter_periodic_queries = self.inter_periodic_q_proj(x_reshape)
        inter_periodic_keys = self.inter_periodic_k_proj(x_reshape)
        inter_periodic_values = self.inter_periodic_v_proj(x_reshape)

        # split the keys, queries and values in multiple heads
        inter_periodic_queries = rearrange(inter_periodic_queries, 'BC N (H D) -> BC H N D', H=H)
        inter_periodic_keys = rearrange(inter_periodic_keys, 'BC N (H D) -> BC H D N', H=H)
        inter_periodic_values = rearrange(inter_periodic_values, 'BC N (H D) -> BC H N D', H=H)

        # compute the unnormalized attention scores
        scale = 1. / sqrt(D // H)
        inter_periodic_attn_scores = torch.matmul(inter_periodic_queries, inter_periodic_keys) * scale # [bs x n_heads x q_len x k_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if self.res_attention and prev is not None:
            inter_periodic_attn_scores = inter_periodic_attn_scores + prev

        # normalize the attention weights
        inter_periodic_attn_weights = F.softmax(inter_periodic_attn_scores, dim=-1)  # [bs x n_heads x q_len x k_len]
        inter_periodic_attn_weights = self.attention_dropout(inter_periodic_attn_weights)

        # compute the new values given the attention weights
        inter_periodic_output = torch.matmul(inter_periodic_attn_weights, inter_periodic_values)  # output: [bs x n_heads x q_len x dim]

        # concatenate the heads and project the output back to the patch_size*d_model dimensions
        inter_periodic_output = rearrange(inter_periodic_output, 'BC H N D -> BC N (H D)')
        inter_periodic_output = self.inter_periodic_out_proj(inter_periodic_output)
        inter_periodic_output = rearrange(inter_periodic_output, 'BC N (P D) -> BC N P D', P=P, D=D)

        out = intra_periodic_output + inter_periodic_output
        return out, inter_periodic_attn_weights


class MLP(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu", flatten=False):
        super(MLP, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.flatten = flatten

    def forward(self, x):
        B, N, P, D = x.shape
        if self.flatten:
            x = rearrange(x, 'B N P D -> B N (P D)')
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.flatten:
            x = rearrange(x, 'B N (P D) -> B N P D', P=P, D=D)
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
        # [bs, num_patch, patch_size, D]
        if self.pre_norm:
            res = x 
            x = self.norm1(x)
        new_x, attn = self.attention(x, prev) # [bs, num_patch, patch_size, D]
        if self.pre_norm:
            x = res + self.dropout(new_x)
        else:
            x = self.norm1(x + self.dropout(new_x))

        res = x
        if self.pre_norm:
            x = self.norm2(x)
        x = self.ffn(x) # [bs, num_patch, patch_size, D]
        if self.pre_norm:
            x = res + x
        else:
            x = self.norm2(res + x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None, patch_size=4):
        super(Encoder, self).__init__()
        self.patch_size = patch_size
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer
        
    def patchify(self, x):
        # [bs, L, D]
        # switch to [bs, D, L]
        x = rearrange(x, 'B L D -> B D L')
        # padding
        x = F.pad(x, (0, self.patch_size - x.shape[-1] % self.patch_size), mode='replicate') # [bs, D, L]
        # unfold
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size) # [bs D num_patch patch_size]
        # switch to [bs, num_patch, patch_size, D]
        x = rearrange(x, 'B D N P -> B N P D')
        return x
    
    def unpacthify(self, x):
        # [bs, num_patch, patch_size, D]
        x = rearrange(x, 'B N P D -> B (N P) D')
        return x

    def forward(self, x):
        # [bs, L, D]
        # patchify
        L = x.shape[1]
        x = self.patchify(x) # [bs, num_patch, patch_size, D]

        attns = []
        for encoder_layer in self.encoder_layers:
            prev = attns[-1] if len(attns) > 0 else None
            x, attn = encoder_layer(x, prev) # [bs, num_patch, patch_size, D]
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x) # [bs, num_patch, patch_size, D]
        
        x = self.unpacthify(x)[:, :L, :] # [bs, L, D]

        return x, attns
        

    
class EncoderStack(nn.Module):
    def __init__(self, configs):
        super(EncoderStack, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.e_layer = configs.e_layers

        patch_sizes = self.get_patch_sizes(configs.seq_len, exclude_zero=True)
        self.num_patch_sizes = len(patch_sizes)

        self.start_linear = nn.Linear(in_features=configs.d_model, out_features=1)
        self.w_noise = nn.Parameter(torch.zeros(configs.seq_len, self.num_patch_sizes), requires_grad=True)

        self.encoders = nn.ModuleList()
        for patch_size in patch_sizes:
            num_patchs = int(self.seq_len / patch_size) + 1
            self.encoders.append(
                Encoder(
                    [
                        EncoderLayer(
                            DualAttention(
                                configs.enc_in, num_patchs, patch_size, configs.d_model, configs.n_heads, attention_dropout=configs.dropout,
                                pos_embed_dropout=configs.dropout, learned_pos_embed=True, res_attention=True, cross=True),
                            MLP(patch_size*configs.d_model, patch_size*configs.d_ff, dropout=configs.dropout),
                            configs.d_model,
                            dropout=configs.dropout,
                            pre_norm=False
                        ) for l in range(self.e_layer)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model),
                    patch_size=patch_size
                )
            )

    def get_patch_sizes(self, seq_len, exclude_zero=True):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:] if exclude_zero else 1 / torch.fft.rfftfreq(seq_len)
        patch_sizes = peroid_list.int().unique()
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
        int_peroid_list = peroid_list.int()
        groups_period, indices = torch.unique(int_peroid_list, return_inverse=True)
        indices = indices.unsqueeze(0).expand(amplitude_list.shape[0], -1).to(amplitude_list.device)
        groups_amplitude = torch.zeros(amplitude_list.shape[0], groups_period.shape[0], device=amplitude_list.device)
        groups_amplitude = groups_amplitude.scatter_add(1, indices, amplitude_list)
        # groups_amplitude = torch.bincount(indices, weights=amplitude_list)
        return groups_period, groups_amplitude

    def top_k_gating(self, x, groups_amplitude, train, noise_epsilon=1e-2):
        x = self.start_linear(x).squeeze(-1)

        clean_logits = groups_amplitude
        if train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(self.k + 1, dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits.softmax(1)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates

    def forward(self, x):
        # [bs, L, D]
        period_list, amplitude_list = self.fft_for_peroid(x) # [T], [bs, T]
        groups_period, groups_amplitude = self.groups_by_period(period_list, amplitude_list) # [num_period], [bs, num_period]
        groups_amplitude = groups_amplitude.softmax(dim=-1) # [bs, num_period]
        gates = self.top_k_gating(x, groups_amplitude, train=self.training) # [bs, num_period]

        dispatcher = SparseDispatcher(self.num_patch_sizes, gates)
        encoders_input = dispatcher.dispatch(x) # list[[bs, L, D]*num_patch_sizes] bs may be equal to zero
        encoders_output = [self.encoders[i](encoders_input[i])[0] for i in range(self.num_patch_sizes)] # list[[bs, L, D]*num_patch_sizes]
        output = dispatcher.combine(encoders_output) # [bs, L, D]
        # output = output + x
        return output # [bs, L, D]


class PretrainHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.top_k = configs.top_k
        self.dec_embedding = DataEmbedding(configs.d_model, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        self.decoder = Encoder(
            [
                EncoderLayer(
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
        )

        self.projections = nn.Linear(configs.d_model, configs.enc_in, bias=True)
    
    def forward(self, x, ids_restore, x_mask=None):
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = self.dec_embedding(x_, x_mask)
        dec_out, _ = self.decoder(x_)
        dec_out = self.projections(dec_out)
        return dec_out
    

class PredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_vars = configs.enc_in
        self.channels_fusion = nn.Linear(configs.d_model * configs.enc_in, configs.d_model)
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
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x [B, S, C1] cross [B, L, C2]
        cross = rearrange(cross, '(B N) L D -> B L (N D)', N=self.n_vars)
        cross = self.channels_fusion(cross)
        dec_out = self.decoder(x, cross, x_mask, cross_mask, tau, delta)
        return dec_out
    
class LinearProbeHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
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
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        dec_out = self.decoder(x, cross, x_mask, cross_mask, tau, delta)
        return dec_out 


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
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
        

        self.encoder_stack = EncoderStack(configs)
        
        if self.pretrain:
            self.head = PretrainHead(configs) # custom head passed as a partial func with all its kwargs
        else:
            self.head = PredictionHead(configs)
    
    def random_masking(self, x, len_keep, ids_shuffle, ids_restore):
        # xb: [bs x num_patch x dim]
        bs, L, D = x.shape

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]                                        
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  
    
        # removed x
        x_removed = torch.zeros(bs, L-len_keep, D, device=x.device)         
        x_ = torch.cat([x_kept, x_removed], dim=1)   

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, x_kept
    
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
        
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [bs, L, D]

        if self.pretrain:
            # mask
            enc_out = self.encoder_stack(enc_out, n_vars) # [bs, L, D]
            # dec_out = self.head(enc_outs, ids_restore, x_mask=None)

            dec_out = self.revin(dec_out, 'inverse')

            return dec_out, mask

        else:
            dec_out = self.dec_embedding(x_dec, None)
            enc_out = self.encoder_stack(enc_out) # [bs, L, D]
            dec_out = self.head(dec_out, enc_out)

            dec_out = self.revin(dec_out, 'inverse')

            return dec_out