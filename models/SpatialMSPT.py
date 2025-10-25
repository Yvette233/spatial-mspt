
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_inverted, PositionalEmbedding, PositionalEmbedding2D
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

from layers.patch_embedding_3d import PatchEmbedding3D
from layers.spatial_attention import SpatialSelfAttention   # ✅ 新增




def dispatch(inp, gates):
    """
    inp:  [B, L, C]
    gates:[B, Ps]
    return:
      outs:    长度 Ps 的 list，每个是 [B_i, L, C]
      buckets: 长度 Ps 的 list，每个是 LongTensor 下标，形状 [B_i]
    """
    B = inp.size(0)
    Ps = gates.size(1)
    idx = torch.argmax(gates, dim=1)  # 每个样本路由到权重最大的专家
    buckets_list = [[] for _ in range(Ps)]
    for b in range(B):
        buckets_list[idx[b].item()].append(b)

    outs = []
    buckets = []
    device = inp.device
    for i in range(Ps):
        if len(buckets_list[i]) == 0:
            outs.append(inp.new_zeros(0, inp.size(1), inp.size(2)))
            buckets.append(torch.empty(0, dtype=torch.long, device=device))
        else:
            sel = torch.tensor(buckets_list[i], device=device, dtype=torch.long)
            outs.append(inp.index_select(0, sel))
            buckets.append(sel)
    return outs, buckets



def combine(expert_out, gates, multiply_by_gates=True):
    # expert_out: list of [B, C, L, D] 或 [B, C, 1, D]
    # gates: [B, Ps]
    B = gates.size(0)
    Ps = gates.size(1)
    # 对空专家补全为 0 张量，让 stack 不会 shape 不一致
    C = expert_out[0].size(1)
    L = expert_out[0].size(2)
    D = expert_out[0].size(3)
    safe = []
    for i, xo in enumerate(expert_out):
        if xo.size(0) == 0:
            safe.append(torch.zeros(B, C, L, D, device=gates.device, dtype=xo.dtype))
        else:
            safe.append(xo)
    # [B, Ps, C, L, D]
    stitched = torch.stack(safe, dim=1)
    if multiply_by_gates:
        # gates: [B, Ps, 1, 1, 1]
        g = gates.view(B, Ps, 1, 1, 1)
        stitched = stitched * g
    # sum over Ps
    combined = stitched.sum(dim=1)  # [B, C, L, D]
    # 避免 log/exp 逻辑（你上游已经不再用 log 空间，保持线性更快更稳）
    return combined




class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, num_features, top_k=5, d_model=512, dropout=0., adaptive=True, use_periodicity=True):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_features = num_features  # ✅ 可以为 None，稍后自动初始化
        self.initialized = False  # ✅ 延迟初始化参数
        self.d_model = d_model

        # get the patch sizes
        self.patch_sizes = self.get_patch_sizes(seq_len)

        # === define experts (periods) ===
        if hasattr(self.patch_sizes, "tolist"):
            self.periods = [int(p) for p in self.patch_sizes.tolist() if int(p) >= 1]
        else:
            self.periods = [int(p) for p in self.patch_sizes if int(p) >= 1]
        self.Ps = len(self.periods)  # number of experts

        # AFNO1D / gating params
        self.start_fc = nn.Linear(num_features, 1)
        self.num_freqs = seq_len // 2
        self.scale = 1 / d_model

        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_freqs, self.num_freqs * 4))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_freqs * 4))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_freqs * 4, self.num_freqs))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_freqs))


        # 用 Ps（专家数）而不是 seq_len-1 或 len(patch_sizes) 作为第二维
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, self.Ps))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, self.Ps))

        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_size, d_model, bias=False))
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len)))
        # 不再直接依赖外部 PositionalEmbedding2D 的 forward 签名
        # 我们自己在本类里维护两个可学习二维位置编码：按通道(C)与时间patch数(N)可广播
        self.pos_time = None  # shape: [1, 1, N, D]
        self.pos_chan = None  # shape: [1, C, 1, D]

        # self.position_embedding = PositionalEmbedding(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.adaptive = adaptive
        self.use_periodicity = use_periodicity
    
    def _build_layers(self, num_features, device=None):
        self.num_features = num_features
        self.start_fc = nn.Linear(num_features, 1)
        self.num_freqs = self.seq_len // 2
        self.scale = 1 / self.d_model
        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_freqs, self.num_freqs * 4))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_freqs * 4))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_freqs * 4, self.num_freqs))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_freqs))


        # 重新计算 patch_sizes / periods / Ps 并对齐 gating 形状
        self.patch_sizes = self.get_patch_sizes(self.seq_len)
        if hasattr(self.patch_sizes, "tolist"):
            self.periods = [int(p) for p in self.patch_sizes.tolist() if int(p) >= 1]
        else:
            self.periods = [int(p) for p in self.patch_sizes if int(p) >= 1]
        self.Ps = len(self.periods)

        self.w_gate  = nn.Parameter(torch.zeros(self.num_freqs, self.Ps))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, self.Ps))

        # 生成 patch embedding 层（与 Ps 一一对应）
        self.value_embeddings = nn.ModuleList([nn.Linear(p, self.d_model, bias=False) for p in self.periods])
        self.padding_patch_layers = nn.ModuleList([nn.ReplicationPad1d((0, ceil(self.seq_len / p) * p - self.seq_len)) for p in self.periods])
        self.pos_time = None
        self.pos_chan = None

        if device is not None:
            self.to(device)
        self.initialized = True


    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy()[::-1]
        # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        return patch_sizes

    def afno1d_for_peroid_weights(self, x, training: bool, noise_epsilon: float = 1e-2):
        """
        x: [B, L, C]
        return gates: [B, Ps]
        """
        B, L, C = x.shape

        # AMP 下强制 FP32 做 FFT，避免半精度 cuFFT 对非 2^n 长度（如 60）报错
        with torch.cuda.amp.autocast(enabled=False):
            x32 = self.start_fc(x.float()).squeeze(-1)          # [B, L]
            xf = torch.fft.rfft(x32, dim=-1, norm='ortho')      # [B, L//2+1]
            xf_ac = xf[:, 1:]                                   # 去掉 DC -> [B, num_freqs]
            power = (xf_ac.real ** 2 + xf_ac.imag ** 2)         # [B, num_freqs]

            # 两层线性（AFNO1D 简化版）
            z = power                                           # [B, F]
            z = z @ self.w1 + self.b1                           # [B, 4F]
            z = F.gelu(z)
            z = z @ self.w2 + self.b2                           # [B, F]

            clean_logits = z @ self.w_gate                      # [B, Ps]
            if training:
                raw_noise_std = z @ self.w_noise                # [B, Ps]
                noise_std = (F.softplus(raw_noise_std) + noise_epsilon)
                logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            else:
                logits = clean_logits

        # 在“专家维度”上做 top-k 稀疏
        Ps = logits.size(-1)
        top_k_eff = min(self.top_k, Ps)
        if top_k_eff < Ps:
            _, sel = torch.topk(logits, k=top_k_eff, dim=-1)    # [B, top_k_eff]
            mask = torch.zeros_like(logits).scatter(1, sel, 1.0)
            logits = logits * mask + (-1e9) * (1.0 - mask)

        gates = F.softmax(logits, dim=-1)                       # [B, Ps]
        return gates


    
    def patch_embedding(self, x, patch_size, index_of_patch):
        B, L, C = x.shape
        # do patching
        x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, L//patch_size, patch_size]
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x) 
        return self.dropout(x) # [B, C, L, D]

    def _add_pos2d(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [B, C, N, D]
        按 C（变量/网格通道）与 N（时间patch数）分别加可学习二维位置编码。
        """
        B, C, N, D = v.shape
        dev = v.device

        # 懒加载/尺寸自适应（训练时自动扩展）
        if (self.pos_time is None) or (self.pos_time.size(2) < N) or (self.pos_time.size(3) != D):
            self.pos_time = nn.Parameter(torch.zeros(1, 1, N, D, device=dev))
            nn.init.trunc_normal_(self.pos_time, std=0.02)

        if (self.pos_chan is None) or (self.pos_chan.size(1) < C) or (self.pos_chan.size(3) != D):
            self.pos_chan = nn.Parameter(torch.zeros(1, C, 1, D, device=dev))
            nn.init.trunc_normal_(self.pos_chan, std=0.02)

        # 截取对齐当前 batch 的 C、N
        pos_t = self.pos_time[:, :, :N, :]
        pos_c = self.pos_chan[:, :C, :, :]

        return v + pos_t + pos_c


    def forward(self, x):
        """
        x: [B, L, C]  已标准化
        返回：
        xs:      list 长度 Ps，每个元素 [B_i, C_flat, L_patch, D]
        gates:   [B, Ps]
        buckets: list 长度 Ps，每个元素是 LongTensor 索引，形如 [B_i]
        """
        if not self.initialized or (self.num_features is None) or (self.start_fc.in_features != x.size(-1)):
            self._build_layers(x.size(-1), device=x.device)

        B, L, C = x.shape
        gates = self.afno1d_for_peroid_weights(x, self.training)    # [B, Ps]

        xs, buckets = [], []
        for i, p in enumerate(self.periods):
            # 选出这个专家激活的样本
            sel = (gates[:, i] > 0).nonzero(as_tuple=True)[0]       # [B_i]
            buckets.append(sel)
            if sel.numel() == 0:
                xs.append(x.new_zeros(0, C, 1, self.d_model))
                continue

            x_sel = x.index_select(0, sel)                          # [B_i, L, C]
            x_sel = x_sel.transpose(1, 2).contiguous()              # -> [B_i, C, L]
            x_padded = self.padding_patch_layers[i](x_sel)          # pad 时间维
            x_patched = x_padded.unfold(dimension=-1, size=p, step=p)  # [B_i, C, N, p]
            B_i, C_i, N_i, _ = x_patched.shape

            v = self.value_embeddings[i](x_patched.reshape(B_i*C_i*N_i, p))   # [B_i*C_i*N_i, D]
            v = v.view(B_i, C_i, N_i, self.d_model)                            # [B_i, C, L_patch, D]

            # 可选：加 2D 位置编码（这里把 C 看成 “H*W” 展平，位置编码模块内部自己处理）
            v = self._add_pos2d(v)

            xs.append(v)

        return xs, gates, buckets


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x
    

class CrossDimensionAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(CrossDimensionAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, C, L, D = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        return self.out_projection(out), attn
    
class InterPeriodicityAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(InterPeriodicityAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, C, L, D = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = rearrange(queries, 'B C L D -> B L C D')
        keys = rearrange(keys, 'B C S D -> B S C D')
        values = rearrange(values, 'B C S D -> B S C D')

        # attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = rearrange(out, 'B L C D -> B C L D')

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, cross_dimension_attention, inter_periodic_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_dimension_attention = cross_dimension_attention
        self.inter_periodicity_attention = inter_periodic_attention
        self.cross_dimension_mlp = MLP(d_model, d_ff, dropout, activation)
        self.inter_periodicity_mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        res = x
        x, attn = self.cross_dimension_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res + self.dropout(x))

        # res = x
        # x = self.cross_dimension_mlp(x)
        # x = self.norm2(res + self.dropout(x))

        res = x
        x, attn = self.inter_periodicity_attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm3(res + self.dropout(x))

        res = x
        x = self.inter_periodicity_mlp(x)
        x = self.norm4(res + self.dropout(x))

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=None, enc_in=None,
                 spa_heads=4, spa_mem_window=5, use_spa=True):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.enc_in = enc_in                     # ✅ 保存展开后的空间通道数（H*W*C）
        self.use_spa = use_spa

        # 空间注意力 embed_dim 必须等于 d_model（否则线性层/残差都会对不上）
        if self.use_spa:
            self.spa = SpatialSelfAttention(
                embed_dim=d_model,               # ✅ 不要写死 64
                n_heads=spa_heads,
                mem_window=spa_mem_window,
                use_geo=True,
                use_sem=True
            )

    def forward(self, x, attn_mask=None, geo_mask=None, sem_mask=None):
        """
        x: [B, C, L, D]
        注意：L 不一定能被 Ps (= enc_in) 整除，必须做安全判断
        """
        B, C, L, D = x.shape
        attns = []

        for attn_layer in self.attn_layers:
            # 原有的周期注意力
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

            # 安全地做空间注意力（可关）
            if self.use_spa and (self.enc_in is not None) and (self.enc_in > 1):
                Ps = self.enc_in                     # 每个时间步的格点数（= H*W*C）
                if L >= Ps and (L % Ps == 0):        # 只有在 L 是 Ps 的整数倍时才 reshape
                    Nt = L // Ps
                    H = x[:, :, :Nt*Ps, :].reshape(B, C, Nt, Ps, D)
                    H = self.spa(H, geo_mask=geo_mask, sem_mask=sem_mask)
                    x = H.reshape(B, C, L, D)
                # 否则直接跳过空间注意力，保持鲁棒

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


    
class LinearPredictionHead(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(d_model, pred_len))
        
    def forward(self, xs, gates):
        # Ps*[B, C, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(xs[i][:, :, -1:, :])) # [B, C, L, D] -> [B, C, P]
        xs = combine(xs, gates)
        xs = rearrange(xs.squeeze(-2), 'B C P -> B P C') # [B, P, C]
        return xs # [bs, P, C]
    

class LinearPredictionHead2(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead2, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, pred_len)

    def forward(self, xs, gates):
        """
        xs:    list，长度 = 有效专家个数，每个张量形状 [B, C, L, D]
        gates: [B, 有效专家个数]，已在外面对齐（gates_valid）
        """
        _xs = []
        # 关键：按 xs 的实际长度遍历，而不是 self.patch_sizes
        for i in range(len(xs)):
            # 取最后一个时间步的特征
            _xs.append(xs[i][:, :, -1:, :])   # 仍是 [B, C, 1, D]

        # 将各专家输出按门控权重聚合 -> [B, C, 1, D]
        _xs = combine(_xs, gates)

        # 线性映射到 pred_len，并变形为 [B, pred_len, C]
        _xs = self.linear(self.dropout(_xs.flatten(-2)))  # [B, C, pred_len]
        _xs = _xs.transpose(1, 2).contiguous()            # [B, pred_len, C]
        return _xs




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 

        # === 新增: 时空 Patch Embedding 模块 ===
        # 让模型可以直接接受 (B, T, H, W, C) 格点输入
        self.use_spatial_patch = getattr(configs, 'use_spatial_patch', True)

        if self.use_spatial_patch:
            # 根据论文实验可调整 patch 大小
            self.patch_embed = PatchEmbedding3D(
                t_patch=getattr(configs, 't_patch', 5),   # 时间维 patch 大小
                h_patch=getattr(configs, 'h_patch', 4),   # 空间 patch 高度
                w_patch=getattr(configs, 'w_patch', 4),   # 空间 patch 宽度
                embed_dim=getattr(configs, 'd_model', 64),
                input_format='BTHWC',   # 对应 NOAA OISST 格式
                use_pos=True
            )


        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual


        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, configs.enc_in, configs.top_k, d_model=configs.d_model, dropout=configs.dropout)
        self.patch_sizes = self.msppe.patch_sizes

        self.encoders = nn.ModuleList()
        for _ in self.patch_sizes:
            self.encoders.append(
                Encoder(
                    [
                        EncoderLayer(
                            CrossDimensionAttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                                configs.d_model, configs.n_heads),
                            InterPeriodicityAttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                                configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for _ in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model),
                    d_model=configs.d_model,            # ✅ 传给 Encoder
                    enc_in=configs.enc_in,              # ✅ 传给 Encoder（= H*W*C）
                    spa_heads=min(4, configs.n_heads),  # 也可以直接用 configs.n_heads
                    spa_mem_window=5,
                    use_spa=True                        # 如果想先稳定训练，这里可以暂时 False
                )
            )



        self.head = LinearPredictionHead2(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, T, H, W, C=1]
        B, T, H, W, C = x_enc.shape
        C_flat = H * W * C

        # 展平到 [B, C_flat, T]
        x_enc = x_enc.view(B, T, C_flat).permute(0, 2, 1).contiguous()  # [B, C_flat, T]

        # ===== 标准化：沿时间维 =====
        means = x_enc.mean(dim=2, keepdim=True)                                    # [B, C_flat, 1]
        stdev = torch.sqrt(x_enc.var(dim=2, keepdim=True, unbiased=False) + 1e-5)  # [B, C_flat, 1]
        x_norm = (x_enc - means) / stdev                                           # [B, C_flat, T]

        # MSPPE 需要 [B, L, C]，转成 [B, T, C_flat]
        x_for_msppe = x_norm.permute(0, 2, 1)  # [B, T, C_flat]
        xs_enc, gates_enc, buckets = self.msppe(x_for_msppe)  # xs_enc: Ps*[B_i, C, L, D]; buckets: Ps*[B_i]

        # 编码器：先把 enc_in 改成真实格点数
        for enc in self.encoders:
            enc.enc_in = C_flat
        enc_outs = []
        valid_indices = []
        buckets_valid = []

        for i, x_e in enumerate(xs_enc):
            if x_e is None or x_e.numel() == 0 or x_e.size(0) == 0:
                continue
            enc_out, _ = self.encoders[i](x_e)           # enc_out: [B_i, C, L, D]
            enc_outs.append(enc_out)
            valid_indices.append(i)
            buckets_valid.append(buckets[i])             # [B_i]

        # 如果极端情况下全空，直接返回 0（很少见）
        if len(enc_outs) == 0:
            device = x_enc.device
            return torch.zeros((B, self.pred_len, H, W, C), device=device)

        # === 把每个专家输出散射回 B 行 ===
        enc_outs_full = []
        for enc_out, idx in zip(enc_outs, buckets_valid):
            # enc_out: [B_i, C, L, D]  ->  full: [B, C, L, D], 其他样本补 0
            C1, L1, D1 = enc_out.size(1), enc_out.size(2), enc_out.size(3)
            full = enc_out.new_zeros(B, C1, L1, D1)
            if idx.numel() > 0:
                full[idx] = enc_out
            enc_outs_full.append(full)

        # 只保留有效专家对应的门控列（顺序与 enc_outs_full 对齐）
        gates_valid = gates_enc[:, valid_indices]

        # 送入预测头（现已全是 B 行，不会再 stack 失败）
        dec_out = self.head(enc_outs_full, gates_valid)  # [B, pred_len, C_flat]



        # ===== 反标准化：沿时间维的统计量 =====
        # means/stdev 是 [B, C_flat, 1]，需要广播到 [B, pred_len, C_flat]
        dec_out = dec_out * stdev.squeeze(2).unsqueeze(1) + means.squeeze(2).unsqueeze(1)  # [B, pred_len, C_flat]

        # 还原空间: [B, pred_len, H, W, C]
        dec_out = dec_out.view(B, self.pred_len, H, W, C)

        return dec_out

