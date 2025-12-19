import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_attention import SpatialSelfAttention

# Module for TransDtSt-Part
class EncoderStack(nn.Module):   ## 类InformerStack会调用这个EncoderStack，从论文里看是为增加distilling的鲁棒性
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        # x_stack = torch.cat(x_stack, -2)
        x_stack = torch.cat((x_stack[0], x_stack[2]), -2)

        return x_stack, attns


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_PreNorm(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_PreNorm, self).__init__()
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
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        x = res + x

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None,
                 use_spatial=True, spa_kwargs=None, dropout=0.0):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.use_spatial = use_spatial
        self.spa = None                               # ← 惰性：不在 __init__ 决定 D
        self.spa_kwargs = spa_kwargs or {}
        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout

    def _ensure_spa(self, D, device):
        if (self.spa is None) or (getattr(self.spa, "embed_dim", None) != D):
            # 基于 D 的合理默认头数；可被 spa_kwargs 覆盖
            auto_heads = max(1, min(8, D // 64))
            kw = dict(
                embed_dim = D,
                n_heads   = self.spa_kwargs.get("n_heads", auto_heads),
                dropout   = self.spa_kwargs.get("dropout", 0.1),
                use_geo   = self.spa_kwargs.get("use_geo", True),
                use_sem   = self.spa_kwargs.get("use_sem", True),
                mem_window= self.spa_kwargs.get("mem_window", 5),
                inject_mode = self.spa_kwargs.get("inject_mode", "pre"),  # 与 SOTA 对齐
                use_gate    = self.spa_kwargs.get("use_gate", False),      # 与 SOTA 对齐
            )
            self.spa = SpatialSelfAttention(**kw).to(device)
            self.spa.embed_dim = D



    def forward(self, x, attn_mask=None, tau=None, delta=None,
                H=None, geo_mask=None, sem_mask=None):
        """
        x: [B, L, D] —— 时间注意输入（如补丁后的序列 token）
        H: [B, C, Nt, Ps, D] 或 None —— 真·空间张量（优先）; 若 None 会退化处理
        """
        attns = []
        B, L, D = x.shape
        device = x.device

        if self.use_spatial:
            self._ensure_spa(D, device)

        for i, attn in enumerate(self.attn_layers):
            # (1) 时间注意（Pre-Norm 结构通常更稳）
            x, a = attn(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(a)
            x = self.drop(x)

            # (2) 可选：卷积/混合层
            if self.conv_layers is not None:
                x = self.conv_layers[i](x)

            # (3) 空间注意：优先使用真·空间 5D；否则退化为 (B,1,L,1,D)
            if self.use_spatial:
                if H is not None:                       # 真·空间路径
                    H = self.spa(H, geo_mask=geo_mask, sem_mask=sem_mask)  # [B,C,Nt,Ps,D]
                    if H.shape[2] == L:                 # Nt == L 则做轻量融合
                        fused = H.mean(dim=(1, 3))      # 沿 C, Ps 平均 → [B,L,D]
                        x = x + fused                   # 残差式注入
                else:                                   # 退化路径（保证不崩）
                    H_tmp = x.unsqueeze(1).unsqueeze(3) # [B,1,L,1,D]
                    H_tmp = self.spa(H_tmp, geo_mask=None, sem_mask=None)
                    x = x + H_tmp.squeeze(3).squeeze(1)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns, H


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class DecoderLayer_PreNorm(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer_PreNorm, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        res = x
        x = self.norm1(x)
        x = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0]
        x = res + self.dropout(x)

        res = x
        x = self.norm2(x)
        cross = self.cross_norm(cross)
        x = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0]
        x = res + self.dropout(x)
        
        res = x
        x = self.norm3(x)
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        x = res + x

        return x


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
