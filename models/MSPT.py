import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

import numpy as np

from einops import rearrange

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
                                

class PeriodicEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PeriodicEmbedding, self).__init__()
        self.intraperiodicity_position_embedding = PositionalEmbedding(d_model=d_model)
        self.interperiodicity_position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, period):
        B, T, N = x.shape
        # padding
        if T % period != 0:
            length = ((T // period) + 1) * period
            padding = torch.zeros([x.shape[0], length - T, x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        out = rearrange(out, 'B (F P) N -> (B F) P N', P=period)
        # print(out.shape, self.intraperiodicity_position_embedding(out).shape)
        out = self.dropout(out + self.intraperiodicity_position_embedding(out))
        out = rearrange(out, '(B F) P N -> (B P) F N', F=length//period)
        out = self.dropout(out + self.interperiodicity_position_embedding(out))
        out = rearrange(out, '(B P) F N -> B (P F) N', P=period)
        return out[:, :T, :]


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
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


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
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        # return combined.log()
        return combined


class DualAttention(nn.Module):
    def __init__(self, configs):
        super(DualAttention, self).__init__()



class TransformerEncoderLayers(nn.Module):
    def __init__(self, attention, layer, ):
        super(TransformerEncoderLayers, self).__init__()
        self.intra_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=False),
            configs.d_model, configs.n_heads)
        
    def patchify(self, x):
        # [B, T, N]
        


        
    
    def forward(self, x):
        # [B T N]
        

    


class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super(TransformerEncoder, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.e_layer = configs.e_layers

        patch_sizes = self.get_patch_sizes(configs.seq_len, exclude_zero=True)
        self.num_patch_sizes = len(patch_sizes)

        self.w_noise = nn.Parameter(torch.zeros(configs.seq_len, self.num_patch_sizes), requires_grad=True)

        self.encoders = nn.ModuleList()
        for patch in patch_sizes:
            num_patchs = int(self.seq_len / patch)
            self.encoders.append(
                TransformerEncoderLayers(device=device, d_model=d_model, d_ff=d_ff,
                                      dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                      patch_size=patch, factorized=True, layer_number=layer_number)
            )

    def get_patch_sizes(self, seq_len, exclude_zero=True):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:] if exclude_zero else 1 / torch.fft.rfftfreq(seq_len)
        patch_sizes = peroid_list.int().unique()
        return patch_sizes

    def fft_for_peroid(self, x, exclude_zero=True):
        # [B, T, C]
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
        # peroid_list [T] amplitude_list [B, T]
        int_peroid_list = peroid_list.int()
        groups_period, indices = torch.unique(int_peroid_list, return_inverse=True)
        indices = indices.unsqueeze(0).expand(amplitude_list.shape[0], -1)
        groups_amplitude = torch.bincount(indices, weights=amplitude_list)
        return groups_period, groups_amplitude

    def top_k_gating(self, x, groups_amplitude, train, noise_epsilon=1e-2):
        clean_logits = groups_amplitude
        if train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates

    def forward(self, x):
        # [B, T, C]
        period_list, amplitude_list = self.fft_for_peroid(x)
        groups_period, groups_amplitude = self.groups_by_period(period_list, amplitude_list)
        groups_amplitude = groups_amplitude.softmax(dim=-1)
        gates = self.top_k_gating(groups_amplitude, train=self.training)

        dispatcher = SparseDispatcher(self.num_patch_sizes, gates)
        encoders_input = dispatcher.dispatch(x)
        encoders_output = [self.encoders[i](encoders_input[i])[0] for i in range(self.num_patch_sizes)]
        output = dispatcher.combine(encoders_output)
        output = output + x
        return output


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

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        self.periodic_embedding = PeriodicEmbedding(d_model=configs.d_model, dropout=configs.dropout)

        self.fft_linear = nn.Linear(configs.seq_len, configs.seq_len)
        
        # Encoder
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                                configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                ) for i in range(self.top_k)
            ]
        )

        self.multi_scale_peroidicity_fusion_linear = nn.Linear(configs.d_model*self.top_k, configs.d_model, bias=False)

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
        # x_enc [B T C]

        # revin
        x_enc, x_dec = self.revin(x_enc, 'forward', x_dec)    
        
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        if self.pretrain:
            bs, L, D = enc_out.shape
            len_keep = int(L * (1 - self.mask_ratio))
            noise = torch.rand(bs, L, device=enc_out.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([bs, L], device=enc_out.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        
            enc_outs = []
            for i in range(self.top_k):
                # period = period_list[i]
                enc_out_ = self.periodic_embedding(enc_out, period)
                x_masked, x_kept = self.random_masking(enc_out_.clone(), len_keep, ids_shuffle, ids_restore)
                enc_out_, _ = self.encoders[i](x_kept)
                enc_outs.append(enc_out_)
            enc_outs = torch.cat(enc_outs, dim=-1)
            enc_outs = self.multi_scale_peroidicity_fusion_linear(enc_outs)
            dec_out = self.head(enc_outs, ids_restore, x_mask=None)

            dec_out = self.revin(dec_out, 'inverse')

            return dec_out, mask

        else:
            dec_out = self.dec_embedding(x_dec, None)
            enc_outs = []
            for i in range(self.top_k):
                period = period_list[i]
                enc_out_ = self.periodic_embedding(enc_out, period)
                enc_out_, _ = self.encoders[i](enc_out_)
                enc_outs.append(enc_out_)
            enc_outs = torch.cat(enc_outs, dim=-1)
            enc_outs = self.multi_scale_peroidicity_fusion_linear(enc_outs)
            dec_out = self.head(dec_out, enc_outs)

            dec_out = self.revin(dec_out, 'inverse')

            return dec_out