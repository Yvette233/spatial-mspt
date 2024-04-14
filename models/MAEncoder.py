# Model with periods embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Revin import RevIN

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
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

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


class PretrainHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # self.dec_embedding = DataEmbedding(configs.d_model, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder_to_decoder = nn.Linear(configs.d_model, configs.d_model, bias=False)
        self.pos_embed = PositionalEmbedding(d_model=configs.d_model)
        self.embed_drop = nn.Dropout(configs.dropout)
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

        self.projection = nn.Linear(configs.d_model, configs.enc_in, bias=True)

    def forward(self, x, ids_restore, x_mask=None):
        x = self.encoder_to_decoder(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # print(mask_tokens.shape)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = self.embed_drop(x_ + self.pos_embed(x_))
        dec_out, _ = self.decoder(x_)
        dec_out = self.projection(dec_out)
        return dec_out
    

class FinetuneHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder_to_decoder = nn.Linear(configs.d_model, configs.d_model, bias=False)
        self.token_embed = nn.Linear(configs.dec_in, configs.d_model)
        self.pos_embed = PositionalEmbedding(d_model=configs.d_model)
        self.embed_drop = nn.Dropout(configs.dropout)
        # self.mask_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
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
        self.projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
    
    def forward(self, x, cross):
        x = self.token_embed(x)
        cross = self.encoder_to_decoder(cross)
        x = torch.cat([cross, x], dim=1)
        x = self.embed_drop(x + self.pos_embed(x))
        dec_out, _ = self.decoder(x)
        dec_out = self.projection(dec_out)
        return dec_out


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
        self.e_layers = configs.e_layers
        self.head_type = configs.head_type
        self.mask_ratio = configs.mask_ratio
        self.individual = configs.individual

        assert self.head_type in ['pretrain', 'finetune', 'predict'], 'head type should be either pretrain, finetune, or predict'

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.revin = RevIN(configs.enc_in)
        
        # Encoder
        self.encoder = Encoder(
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
        )

        if self.head_type == "pretrain":
            self.head = PretrainHead(configs) # custom head passed as a partial func with all its kwargs
        elif self.head_type == "finetune":
            self.head = FinetuneHead(configs)
        elif self.head_type == "predict":
            self.head = PredictHead(configs)
        elif self.head_type == "linearProbe":
            pass

        # self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # revin
        x_enc, _ = self.revin(x_enc, 'forward', x_dec)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        

        if self.head_type == "pretrain":
            x_masked, x_kept, mask, ids_restore = random_masking(enc_out, self.mask_ratio)
            
            enc_out, _ = self.encoder(x_masked)
            dec_out = self.head(enc_out, ids_restore, x_mask=None)

            dec_out = self.revin(dec_out, 'inverse')

            return dec_out, mask

        else:
            # dec_out = self.dec_embedding(x_dec, None)

            enc_out, _ = self.encoder(enc_out)
            dec_out = self.head(x_dec, enc_out)

            dec_out = self.revin(dec_out, 'inverse')
            return dec_out[:, -self.pred_len:, :]