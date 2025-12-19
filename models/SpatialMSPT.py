import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention
# 👇 这一行非常重要，千万不能少！
from layers.spatial_attention import GlobalDelaySpatialEmbedding
from layers.patch_embedding_3d import PatchEmbedding3D
import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat
from layers.patch_embedding_3d import PatchEmbedding3D
from layers.spatial_attention import SpatialSelfAttention   # ✅ 新增

# ================= 辅助函数 (智能修复版) =================
def dispatch(inp, gates):
    """
    inp: [B, T, C]
    gates: [B, P] (权重矩阵)
    根据 gates 将 inp 动态分发给不同的分支，允许每个分支样本数不同。
    """
    # 1. 找出所有非零的路由关系 (样本id, 分支id)
    nonzero = torch.nonzero(gates) # [nnz, 2]
    
    # 2. 按照分支id排序，这样才能用 split 切分
    sorted_indices = torch.argsort(nonzero[:, 1]) 
    nonzero_sorted = nonzero[sorted_indices]
    
    # 3. 提取对应的样本数据
    batch_idx_sorted = nonzero_sorted[:, 0]
    inp_expanded = inp[batch_idx_sorted] 
    
    # 4. 计算每个分支分到了多少个样本
    # bincount 统计每个分支id出现的次数
    num_branches = gates.shape[1]
    part_sizes = torch.bincount(nonzero_sorted[:, 1], minlength=num_branches).cpu().tolist()
    
    # 5. 切分数据
    return torch.split(inp_expanded, part_sizes, dim=0)

def combine(xs, gates):
    """
    xs: list of [B_i, ...], 每个分支的结果，B_i 可能不相等
    gates: [B, P]
    将不同分支的结果加权融合回原始的 Batch 形状 [B, ...]
    """
    device = gates.device
    B = gates.shape[0]
    
    # 1. 自动推断输出形状 (除了 Batch 维度的其他维度)
    # 从 xs 中找一个非空的张量来获取形状
    trailing_shape = None
    for x in xs:
        if x.shape[0] > 0:
            trailing_shape = x.shape[1:]
            break
    
    # 如果所有分支都没数据（极端情况），返回全0
    if trailing_shape is None:
        # 假设是 [16, 1, 64] (根据您的报错推断)
        return torch.zeros(B, 16, 1, 64, device=device)
        
    # 2. 初始化输出容器
    combined = torch.zeros(B, *trailing_shape, device=device)
    
    # 3. 逐个分支归位
    for i, x in enumerate(xs):
        if x.shape[0] == 0: continue # 这个分支没分到数据，跳过
        
        # 找出这个分支对应的原始样本索引
        # gates[:, i] > 0 的位置
        active_idx = torch.nonzero(gates[:, i]).squeeze(-1)
        
        # 对应的权重 g: [B_active, 1, 1...]
        g = gates[active_idx, i].view(-1, *([1]*len(trailing_shape)))
        
        # 加权累加
        combined[active_idx] += x * g
            
    return combined + 1e-9

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
            eps = 1e-12 if getattr(self, "sota_mode", 0) else 0.0
            sel = (gates[:, i] > eps).nonzero(as_tuple=True)[0]

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
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, spatial_emb=None):
        # ✅ 核心改动：支持特征注入
        if spatial_emb is not None:
            x = x + spatial_emb
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class LinearPredictionHead2(nn.Module):
    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead2, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        self.linear_freq = nn.Linear(d_model, pred_len)
        self.linear_spatial = nn.Linear(d_model, pred_len)
        
    def forward(self, xs, gates, spatial_feat):
        # 频率流
        _xs = []
        for i, patch_size in enumerate(self.patch_sizes):
            _xs.append(xs[i][:, :, -1:, :])
        freq_agg = combine(_xs, gates)
        freq_out = self.linear_freq(self.dropout(freq_agg.flatten(-2)))
        
        # 空间流
        spatial_out = self.linear_spatial(self.dropout(spatial_feat))
        
        # 融合
        final_out = freq_out + spatial_out
        
        return final_out.transpose(1, 2).contiguous()

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs 
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # ✅ 告诉模型有 16 个节点
        self.num_nodes = 16  
        
        # 1. 左路：G-DASE (空间)
        self.global_spatial = GlobalDelaySpatialEmbedding(
            num_nodes=self.num_nodes,
            input_dim=1,
            embed_dim=configs.d_model,
            max_lag=5,
            top_k=4
        )

        # 2. 右路：MSPPE (频率)
        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len, configs.enc_in, configs.top_k, d_model=configs.d_model, dropout=configs.dropout)
        self.patch_sizes = self.msppe.patch_sizes

        # 3. 中间：编码器
        self.encoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                Encoder(
                   [
                        EncoderLayer(
                            CrossDimensionAttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            InterPeriodicityAttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )

        # 4. 尾部
        self.head = LinearPredictionHead2(self.patch_sizes, self.seq_len, self.pred_len, configs.d_model, dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, T, H, W, C = x_enc.shape
        x_spatial = x_enc.view(B, T, H * W, C)
        
        # 1. 左路：计算全局空间特征 [B, T, 16, D]
        spatial_emb_full = self.global_spatial(x_spatial)
        
        # 2. 预处理
        x_raw = x_spatial.squeeze(-1).permute(0, 2, 1).contiguous()
        means = x_raw.mean(dim=-1, keepdim=True).detach()
        x_raw = x_raw - means
        stdev = torch.sqrt(torch.var(x_raw, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x_raw = x_raw / stdev
        
        # 3. 右路：频率分解
        x_for_msppe = x_raw.permute(0, 2, 1) 
        # 只取前两个返回值
        msppe_out = self.msppe(x_for_msppe)
        xs_enc = msppe_out[0]
        gates_enc = msppe_out[1]
        
        # 4. 中间融合 (Feature Injection)
        enc_outs = []
        
        # ✅ 修正1：先换位再变形，确保数据不乱 [B, T, N, D] -> [B, N, D, T]
        s_emb = spatial_emb_full.permute(0, 2, 3, 1) 
        # 展平以便池化 [B*N, D, T]
        s_emb_flat = s_emb.reshape(B * self.num_nodes, self.configs.d_model, T)
        
        for i, x_e in enumerate(xs_enc):
            patch_size = self.patch_sizes[i]
            
            # ✅ 修正2：加上和 MSPPE 一样的 Padding，确保长度对齐
            # 计算需要补多少 0 才能被 patch_size 整除
            pad_l = ceil(T / patch_size) * patch_size - T
            if pad_l > 0:
                # 在时间维(最后一个维度)补 pad_l 个数据，模式为复制(replicate)
                s_emb_padded = F.pad(s_emb_flat, (0, pad_l), mode='replicate')
            else:
                s_emb_padded = s_emb_flat
            
            # 池化 [B*N, D, L_patch]
            s_emb_pooled = F.avg_pool1d(s_emb_padded, kernel_size=patch_size, stride=patch_size)
            
            # 还原维度 [B*N, D, L] -> [B, N, D, L] -> [B, N, L, D]
            s_emb_pooled = s_emb_pooled.reshape(B, self.num_nodes, self.configs.d_model, -1).permute(0, 1, 3, 2)
            
            # 筛选 batch
            num_samples = x_e.shape[0]
            if num_samples == 0:
                enc_outs.append(x_e)
                continue
            active_idx = (gates_enc[:, i] > 0).nonzero(as_tuple=True)[0]
            s_emb_sub = s_emb_pooled[active_idx]
            
            # 注入
            enc_out, _ = self.encoders[i](x_e, spatial_emb=s_emb_sub)
            enc_outs.append(enc_out)

        # 5. 汇聚
        spatial_feat_last = spatial_emb_full[:, -1, :, :]
        dec_out = self.head(enc_outs, gates_enc, spatial_feat_last)

        # 6. 恢复
        dec_out = dec_out.view(B, self.pred_len, H, W, C)
        means_reshaped = means.permute(0, 2, 1).view(B, 1, H, W, C)
        stdev_reshaped = stdev.permute(0, 2, 1).view(B, 1, H, W, C)
        dec_out = dec_out * stdev_reshaped + means_reshaped
        return dec_out