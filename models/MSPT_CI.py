
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, top_k=5, d_model=512, dropout=0., sparsity_threshold=0.01, hidden_size_factor=4, num_variates=14, individual=False):
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
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * hidden_size_factor, self.freq_seq_len * 2))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * 2))
        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        for patch_len in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_len, d_model, bias=False) if individual else nn.Linear(patch_len*num_variates, d_model, bias=False))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        patch_sizes = [i for i in range(2, seq_len+1)] # [2, 3, ..., seq_len]
        return patch_sizes
    
    def afno1d_for_peroid_weights(self, x):
        # x [B, L, C]
        B, L, C = x.shape

        x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        xf = torch.fft.rfft(x, dim=-1) # [B, C, L//2+1]
        xf_no_zero = xf[:, :, 1:] # [B, C, L//2]

        o1_real = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros([B, C, self.freq_seq_len * 2], device=x.device)
        o2_imag = torch.zeros([B, C, self.freq_seq_len * 2], device=x.device)

        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', xf_no_zero.real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', xf_no_zero.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...bi,bio->...bo', xf_no_zero.imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', xf_no_zero.real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real = (
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = (
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        xf_no_zero = torch.stack([o2_real, o2_imag], dim=-1) # [B, C, L-1, 2]
        xf_no_zero = F.softshrink(xf_no_zero, lambd=self.sparsity_threshold) # [B, C, L-1, 2]
        xf_no_zero = torch.view_as_complex(xf_no_zero) # [B, C, L-1]

        weights = torch.abs(xf_no_zero).mean(dim=1) # [B, L-1]
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1) # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1) # [B, top_k]

        zeros = torch.zeros_like(weights) # [B, L-1]
        gates = zeros.scatter_(-1, top_indices, top_weights) # [B, L-1]

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
        return torch.split(xs, _part_sizes, dim=0)
    
    def patch_embedding(self, x, patch_size, index_of_patch):
        B, L, C = x.shape
        # do patching
        padding_len = ceil(L / patch_size) * patch_size - L
        x = F.pad(x, (0, 0, 0, padding_len), 'replicate') # [B, L+padding_len, C]
        x = x.unfold(1, patch_size, patch_size) # [B, L//patch_size, patch_size, C]
        x = rearrange(x, 'B L P C -> B L (P C)')
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x)
        return self.dropout(x)

    def forward(self, x):
        B, L, C = x.shape
        gates = self.afno1d_for_peroid_weights(x)
        xs = self.dispatcher(x, gates)
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.patch_embedding(xs[i], patch_size, i)
        return xs, gates


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
        stitched = torch.cat(xs, 0).exp() 
        if multiply_by_gates:
            stitched = torch.einsum("ikh,ik -> ikh", stitched, _nonzero_gates) # [BN L D] [BN L] -> [BN L D]
        zeros = torch.zeros(gates.size(0), xs[-1].size(1), xs[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, _batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()
    

class PredictionHead(nn.Module):
    def __init__(self, enc_in, head_nf, pred_len, head_dropout=0.):
        super(PredictionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(enc_in, head_nf),
            nn.ReLU(), 
            nn.Dropout(head_dropout),
            nn.Linear(head_nf, pred_len)
        )
    
    def forward(self, x):
        return self.head(x)
    

class PretrainHead(nn.Module):
    def __init__(self, enc_in, head_nf, pred_len, head_dropout=0.):
        super(PretrainHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(enc_in, head_nf),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_nf, pred_len)
        )
    
    def forward(self, x):
        return self.head(x)


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

        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, top_k=configs.top_k, d_model=configs.d_model, dropout=configs.dropout, num_variates=configs.enc_in, individual=self.individual)
        self.patch_sizes = self.msppe.patch_sizes
        self.dec_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout) if self.individual else DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                Encoder(
                   [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(configs.d_model)
                )
            )
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
                    # projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
                )
            )
        
        self.cspfa = CrossScalePeriodicFeatureAggregator()

        self.iTransformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.pred_len*configs.d_model, configs.n_heads),
                    configs.pred_len*configs.d_model,
                    configs.pred_len*configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(configs.pred_len*configs.d_model)
        )

        self.projection = nn.Linear(configs.pred_len*configs.d_model, configs.pred_len, bias=True) if self.individual else nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Channel independence
        B, L, C = x_enc.shape
        if self.individual:
            x_enc = rearrange(x_enc, 'B L C -> (B C) L 1')
            x_dec = rearrange(x_dec, 'B L C -> (B C) L 1')
            B, L, C = x_enc.shape

        # Multi-scale periodic patch embedding  
        xs_enc, gates_enc = self.msppe(x_enc) # N*[B, L, D], [B, L-1]
        # Decoder embedding
        x_dec = self.dec_embedding(x_dec, None) # [B, S, D]

        # Encoder and Decoder
        dec_outs = []
        for i, x_enc in enumerate(xs_enc):
            enc_out, attns = self.encoders[i](x_enc) # [B, L, D]
            dec_out = self.decoders[i](x_dec, enc_out) # [B, S, D]
            dec_outs.append(dec_out[:, -self.pred_len:, :]) # [B, P, D]
        
        # Cross-scale periodic feature aggregator
        dec_outs = self.cspfa(dec_outs, gates_enc) # [B, P, D]
        
        # iTransformer
        if self.individual:
            dec_outs = rearrange(dec_outs, '(B C) L D -> B C (L D)', B=B, C=C)
            dec_outs, attns = self.iTransformer(dec_outs)
            dec_outs = self.projection(dec_outs)
            dec_out = rearrange(dec_outs, 'B C L -> B L C')
        else:
            dec_out = self.projection(dec_outs)
            dec_out = dec_out.repeat(1, 1, C)

        return dec_out
