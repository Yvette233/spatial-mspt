import torch
import torch.nn as nn
import torch.nn.functional as F


# 在文件顶部导入
from layers.spatial_attention import SpatialSelfAttention

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
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        # === 新增: 空间注意力模块 ===
        self.spa = SpatialSelfAttention(embed_dim=attn_layers[0].attention.d_model if hasattr(attn_layers[0].attention, 'd_model') else 64,
                                        n_heads=4, mem_window=5, use_geo=True, use_sem=True)

        def forward(self, x, attn_mask=None, tau=None, delta=None, geo_mask=None, sem_mask=None):
            """
            x: (B, L, D)  -- original temporal tokens
            or (B, C, Nt, Ps, D) if you pre-embed as spatio-temporal tokens (preferred)
            """
            attns = []

            # If we receive (B, C, Nt, Ps, D) as H_spatial (pre-embedded), we will bypass per-layer
            # reshape; otherwise we will treat x as (B,L,D) and expand to (B,1,L,1,D) as placeholder.
            is_spatial = False
            if x.ndim == 5 and x.shape[1] > 1:
                # format (B, C, Nt, Ps, D) OR (B, C, Nt, Ps, D)
                is_spatial = True

            # lazy init of spa to match embedding dim D (x may be 3D or 5D)
            if x.ndim == 3:
                B, L, D = x.shape
            elif x.ndim == 5:
                B, C, Nt, Ps, D = x.shape
                # flatten time dimension to L for temporal attention if needed
                L = Nt * Ps
            else:
                raise ValueError(f"Unsupported x.ndim={x.ndim}")

            if not hasattr(self, "spa") or self.spa.embed_dim != D:
                # choose n_heads reasonably: (e.g. 8 or min(8, D//8))
                n_heads = min(8, max(1, D // 8))
                self.spa = SpatialSelfAttention(embed_dim=D, n_heads=n_heads, mem_window=5, use_geo=True, use_sem=True)
                # move to same device as x if previously created on CPU
                self.spa.to(x.device)

            if self.conv_layers is not None:
                # original pipeline for conv_layers present
                for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                    delta = delta if i == 0 else None
                    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                    x = conv_layer(x)
                    attns.append(attn)

                    # === 插入空间注意力（按需） ===
                    # If x is temporal tokens (B,L,D), create placeholder spatial H with Ps=1
                    if x.ndim == 3:
                        B, L, D = x.shape
                        H_spa = x.view(B, 1, L, 1, D)   # (B, C=1, Nt=L, Ps=1, D)
                        H_spa = self.spa(H_spa, geo_mask=geo_mask, sem_mask=sem_mask)
                        x = H_spa.view(B, L, D)
                    else:
                        # if x already spatio-temporal: assume x shape is (B, C, Nt, Ps, D)
                        H_spa = self.spa(x, geo_mask=geo_mask, sem_mask=sem_mask)
                        # decide how to convert back for subsequent attn layers:
                        # simplest: flatten back to (B, L, D)
                        B, C, Nt, Ps, D = H_spa.shape
                        x = H_spa.view(B, C * Nt * Ps, D)
                # final attn_layers[-1] existing call (保持原有)
                x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
                attns.append(attn)
            else:
                # no conv layers, just sequential attn layers
                for attn_layer in self.attn_layers:
                    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                    attns.append(attn)

                    # 插入SPA同上
                    if x.ndim == 3:
                        B, L, D = x.shape
                        H_spa = x.view(B, 1, L, 1, D)
                        H_spa = self.spa(H_spa, geo_mask=geo_mask, sem_mask=sem_mask)
                        x = H_spa.view(B, L, D)
                    else:
                        B, C, Nt, Ps, D = x.shape
                        H_spa = self.spa(x, geo_mask=geo_mask, sem_mask=sem_mask)
                        x = H_spa.view(B, C * Nt * Ps, D)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns


    def forward(self, x, attn_mask=None, tau=None, delta=None,
        geo_mask=None, sem_mask=None):   # ✅ 给两个mask加默认None

        # x [B, L, D]
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            # === 新增: 在每层后添加空间注意力 ===
            # 将 x 从 (B,L,D) reshape 成 (B, C, Nt, Ps, D)
            # 这里先用简单假设：C=1, Nt=L, Ps=1（未来扩展成空间patch）
            B, L, D = x.shape
            H = x.view(B, 1, L, 1, D)       # 占位扩展维度
            H = self.spa(H, geo_mask=geo_mask, sem_mask=sem_mask)
            x = H.view(B, L, D)             # 再还原回来
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


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
