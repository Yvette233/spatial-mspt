import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


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


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # self.padding_patch_layer = nn.CircularPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        nn.Embedding

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        # sns.lineplot(data=x.mean(0).mean(0).detach().cpu().numpy(), ax=ax[0])
        # print(x.shape, self.patch_len, self.stride)
        x = self.padding_patch_layer(x)
        # print(x.shape)
        # sns.lineplot(data=x.mean(0).mean(0).detach().cpu().numpy(), ax=ax[1])
        # plt.savefig('padding.png')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
    

class MTSTPatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(MTSTPatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((padding, 0))

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
        # T = x.shape[2]
        # length = T // self.patch_len * self.patch_len
        # x = x[:, :, -length:]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class VariablePatchEmbedding(nn.Module):
    '''
    DataEmbedding for MSPformer: Variable Patch Embedding + Positional Embedding + Temporal Embedding(only used in the decoder)
    '''
    def __init__(self, e_in, d_model, max_patch_len=365, embed_type='fixed', freq='h', dropout=0.1):
        super(VariablePatchEmbedding, self).__init__()
        self.e_in = e_in
        self.d_model = d_model
        self.scale = 1 / d_model
        
        self.value_embedding_weight = nn.Parameter(self.scale * torch.randn(max_patch_len*e_in, d_model))
        self.value_embedding_bias = nn.Parameter(torch.zeros(d_model))
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=5000)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, period):
        # x: [B, T, N] x_mark: [B, T, N] period: [1]
        B, T, N = x.shape
        # padding
        if T % period != 0:
            length = ((T // period) + 1) * period
            # zeros = torch.zeros([x.shape[0], length - T, x.shape[2]], device=x.device)
            # x = torch.cat([zeros, x], dim=1)
            x = F.pad(x, (0, 0, length - T, 0), 'replicate')
            # x = F.pad(x.permute(0, 2, 1), (length - T, 0), 'constant', 0).permute(0, 2, 1)
            # length = ((T // period)) * period
            # x = x[:, -length:, :]
        else:
            length = T
        # variable patch embedding
        x = x.reshape(B, length//period, N*period)
        x = x @ self.value_embedding_weight[:period*self.e_in, :] + self.value_embedding_bias
        # position embedding
        x = x + self.position_embedding(x) # [B*N, T//period, D] or [B, T//period, D]
        return self.dropout(x)
    

class VariablePatchEmbedding_wo_pos(nn.Module):
    '''
    DataEmbedding for MSPformer: Variable Patch Embedding + Temporal Embedding(only used in the decoder)
    '''
    def __init__(self, e_in, d_model, max_patch_len=365, embed_type='fixed', freq='h', dropout=0.1, use_timeF=False):
        super(VariablePatchEmbedding_wo_pos, self).__init__()
        self.e_in = e_in
        self.d_model = d_model
        self.use_timeF = use_timeF
        self.scale = 1 / e_in
        
        self.value_embedding_weight = nn.Parameter(self.scale * torch.randn(max_patch_len*e_in, d_model))
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, period, x_mark=None):
        # x: [B, T, N] x_mark: [B, T, N] period: [1]
        B, T, N = x.shape
        # padding
        if T % period != 0:
            # length = ((T // period) + 1) * period
            # # zeros = torch.zeros([x.shape[0], length - T, x.shape[2]], device=x.device)
            # # x = torch.cat([zeros, x], dim=1)
            # x = F.pad(x.permute(0, 2, 1), (length - T, 0), 'replicate').permute(0, 2, 1)
            # # x = F.pad(x.permute(0, 2, 1), (length - T, 0), 'constant', 0).permute(0, 2, 1)
            length = ((T // period)) * period
            x = x[:, -length:, :]
        else:
            length = T
        # variable patch embedding
        x = x.reshape(B, length//period, N*period)
        x = x @ self.value_embedding_weight[:period*self.e_in, :]
        # temporal embedding
        if self.use_timeF and x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)
