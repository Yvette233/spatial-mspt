# Model with periods embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

from einops import rearrange

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k) 
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


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

class MultiScalePeroidicEncoder(nn.Module):
    def __init__(self, inter_periodicity_attention, intra_periodicity_attention):
        super(MultiScalePeroidicEncoder, self).__init__()
        self.inter_periodicity_attention = inter_periodicity_attention
        self.intra_periodicity_attention = intra_periodicity_attention
        

class PretrainHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.top_k = configs.top_k
        self.dec_embedding = DataEmbedding(configs.d_model*configs.top_k, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model*configs.top_k))
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

        self.projections = nn.Linear(configs.d_model, 1, bias=True)
    
    def forward(self, x, ids_restore, x_mask=None):
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = self.dec_embedding(x_, x_mask)
        dec_out, _ = self.decoder(x_)
        dec_out = self.projections(dec_out)
        return dec_out
    

class FinetuneHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dec_embedding = DataEmbedding(configs.d_model*configs.top_k, configs.d_model, configs.embed, configs.freq, configs.dropout)
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


class PredictHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1) 
        else:
            x = self.flatten(x)  
            x = self.dropout(x)
            x = self.linear(x) 
        return x.transpose(2,1)   


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        self.e_layers = configs.e_layers
        self.head_type = configs.head_type
        self.mask_ratio = configs.mask_ratio
        self.individual = configs.individual

        assert self.head_type in ['pretrain', 'finetune', 'predict'], 'head type should be either pretrain, finetune, or predict'

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.periodic_embedding = PeriodicEmbedding(d_model=configs.d_model, dropout=configs.dropout)
        self.token_embedding = nn.Linear(1, configs.d_model)
        self.pos_embedding = PositionalEmbedding(d_model=configs.d_model)
        self.time_embedding = TimeFeatureEmbedding(d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq)
        self.embed_dropout = nn.Dropout(configs.dropout)
        
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

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


        if self.head_type == "pretrain":
            self.head = PretrainHead(configs) # custom head passed as a partial func with all its kwargs
        elif self.head_type == "finetune":
            self.head = FinetuneHead(configs)
        elif self.head_type == "predict":
            self.head = PredictHead(configs)
    
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
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        B, L, n_vars = x_enc.shape
        x_enc = x_enc.transpose(1, 2).unsqueeze(-1).reshape(B*n_vars, L, -1)
        x_mark_enc = x_mark_enc.repeat(n_vars, 1, 1)
        enc_out = self.embed_dropout(self.token_embedding(x_enc) + self.pos_embedding(x_enc) + self.time_embedding(x_mark_enc))
        
        period_list, period_weight = FFT_for_Period(enc_out, self.top_k)

        if self.head_type == "pretrain":
            # x_masked, x_kept, mask, ids_restore = random_masking(enc_out, self.mask_ratio)
            # print(x_masked.shape, x_kept.shape, mask.shape, ids_restore.shape)
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
                period = period_list[i]
                enc_out_ = self.periodic_embedding(enc_out, period)
                x_masked, x_kept = self.random_masking(enc_out_.clone(), len_keep, ids_shuffle, ids_restore)
                # print(x_masked.shape, x_kept.shape)
                enc_out_, _ = self.encoders[i](x_kept)
                enc_outs.append(enc_out_)
            enc_outs = torch.cat(enc_outs, dim=-1)
            dec_out = self.head(enc_outs, ids_restore, x_mask=x_mark_enc)

            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.seq_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.seq_len, 1))

            return dec_out, mask

        else:
            enc_outs = []
            for i in range(self.top_k):
                period = period_list[i]
                enc_out_ = self.periodic_embedding(enc_out, period)
                enc_out_, _ = self.encoders[i](enc_out_)
                enc_outs.append(enc_out_)

        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        
        # return dec_out[:, -self.pred_len:, :]  # [B, L, D]