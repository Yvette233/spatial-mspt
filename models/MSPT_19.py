
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

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, top_k=5, d_model=512, dropout=0., sparsity_threshold=0.01, hidden_size_factor=4, c_in=14, individual=False):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.individual = individual
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
        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_size, d_model, bias=False) if individual else nn.Linear(c_in*patch_size, d_model, bias=False))
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len)))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique()
        return patch_sizes
    
    def afno1d_for_peroid_weights(self, x):
        # x [B, L, C]
        B, L, C = x.shape

        x = rearrange(x, 'B L C -> B C L') # [B, C, L] 
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
        xf_no_zero = F.softshrink(xf_no_zero, lambd=self.sparsity_threshold) # [B, C, L-1, 2]
        xf_no_zero = torch.view_as_complex(xf_no_zero) # [B, C, L-1]

        weights = torch.abs(xf_no_zero).mean(dim=-2) # [B, L-1]
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1) # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1) # [B, top_k]

        zeros = torch.zeros_like(weights) # [B, L-1]
        gates = zeros.scatter_(-1, top_indices, top_weights) # [B, L-1]

        return gates # [B, L-1]

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
        x = rearrange(x, 'B L (C D) -> (B C) D L', D=1) if self.individual else rearrange(x, 'B L C -> B C L')
        # padding_len = ceil(L / patch_size) * patch_size - L
        # x = F.pad(x, (0, padding_len), 'replicate') # [B, L+padding_len, C]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, L//patch_size, patch_size]
        x = rearrange(x, 'B C L P -> B L (P C)')
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x) # [B, L, D]
        return self.dropout(x)

    def forward(self, x):
        B, L, C = x.shape
        gates = self.afno1d_for_peroid_weights(x) # [bs, L-1]
        xs = self.dispatcher(x, gates) # L-1*[bs, L, D]
        for i, patch_size in enumerate(self.patch_sizes): 
            xs[i] = self.patch_embedding(xs[i], patch_size, i)
        return xs, gates # L-1*[bs, L, D], [bs, L-1]


class CrossScalePeriodicFeatureAggregator(nn.Module):
    def __init__(self, patch_sizes, seq_len, d_model):
        super(CrossScalePeriodicFeatureAggregator, self).__init__()
        # self.patch_sizes = patch_sizes
        # self.seq_len = seq_len
        # self.projections = nn.ModuleList()
        # for patch_size in self.patch_sizes:
        #     self.projections.append(nn.Linear(d_model, patch_size*d_model//8))

    def forward(self, xs, gates, multiply_by_gates=True):
        # # back to original length
        # for i, patch_size in enumerate(self.patch_sizes):
        #     xs[i] = self.projections[i](xs[i])
        #     xs[i] = rearrange(xs[i], 'B L (P D) -> B (L P) D', P=patch_size)[:,:self.seq_len,:]
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
        # go back to log space
        combined = combined.log()
        return combined
    

class LinearRegressionHead(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0., c_in=14, individual=False):
        super(LinearRegressionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.c_in = c_in
        self.individual = individual
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(ceil(seq_len / patch_size)*d_model, pred_len) if individual else nn.Linear(ceil(seq_len / patch_size)*d_model, pred_len*c_in)) 
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)

        
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # L-1*[bs, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(xs[i].flatten(1))) # [bs, L, D] -> [bs, P*D]
            xs[i] = rearrange(xs[i], '(B C) P -> B P C', C=self.c_in) if self.individual else rearrange(xs[i], 'B (P C) -> B P C', C=self.c_in) # [bs, P, C]
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
        x_dec = rearrange(x_dec, 'B S C -> B S C D', D=1) if self.individual else x_dec
        x_dec = self.data_embedding(x_dec) # [bs, S, D]
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
    def __init__(self, patch_sizes, seq_len, d_model, dropout=0., c_in=14, individual=False):
        super(LinearPretrainHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.c_in = c_in
        self.individual = individual
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(d_model, patch_size)) if individual else nn.Linear(d_model, patch_size*c_in)
        self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)
    
    def forward(self, xs, gates, x_dec, x_mark_dec=None):
        # L-1*[bs, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(xs[i]))
            xs[i] = rearrange(xs[i], 'B L (P C) -> B (L P) C', P=patch_size)[:,:self.seq_len,:] # [bs, L, C]
        xs = self.cspfa(xs, gates) # [bs, L, C]
        xs = rearrange(xs, '(B C) L D -> B L (C D)', C=self.c_in) if self.individual else xs
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

        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, top_k=configs.top_k, d_model=configs.d_model, dropout=configs.dropout, c_in=configs.enc_in, individual=configs.individual)
        self.patch_sizes = self.msppe.patch_sizes

        self.periodicencoders = nn.ModuleList()
        self.variationalencoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.periodicencoders.append(
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
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )

            self.variationalencoders.append(
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
                        )
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )

        if self.pretrain:
            self.head = LinearPretrainHead(self.patch_sizes, self.seq_len, configs.d_model, dropout=configs.dropout, c_in=configs.enc_in, individual=self.individual)
        else:    
            self.head = LinearRegressionHead(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout, c_in=configs.enc_in, individual=self.individual)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        # x_enc = rearrange(x_enc, 'B L (C D) -> B C L D', D=1) if self.individual else x_enc
        # Multi-scale periodic patch embedding  
        xs_enc, gates_enc = self.msppe(x_enc) # L-1*[bs, L, D], [bs, L-1]
        # Encoder and Decoder
        enc_outs = []
        for i, x_enc in enumerate(xs_enc):
            num_tokens = x_enc.size(1)
            enc_out, attns1 = self.periodicencoders[i](x_enc) # [bs, VT, D]
            enc_out = rearrange(enc_out, '(B C) L D -> (B L) C D', C=C) # [B*L, C, D]
            enc_out, attns2 = self.variationalencoders[i](enc_out) # [B*L, C, D]
            enc_out = rearrange(enc_out, '(B L) C D -> (B C) L D', L=num_tokens) # [bs, L, D]
            enc_outs.append(enc_out)

        # Head
        dec_out = self.head(enc_outs, gates_enc, x_dec)

        return dec_out
