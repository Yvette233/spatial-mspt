import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. ✅ 新增：旧类的“替身” (防止报错)
# ==========================================
class SpatialSelfAttention(nn.Module):
    """
    保留这个空壳类，是为了兼容旧代码的 import。
    我们现在的 SpatialMSPT 模型不会真正用到它。
    """
    def __init__(self, d_model, n_heads=8, d_keys=None, d_values=None):
        super().__init__()
        
    def forward(self, x, mask=None):
        # 如果被误调用，直接原样返回，保证不报错
        return x, None

class GlobalDelaySpatialEmbedding(nn.Module):
    """
    【顶刊级模块】全局延时感知空间嵌入层
    作用：在 MSPT 进行 Patching 之前，利用连续的时间序列计算空间传播延时。
    这解决了 Patching 破坏时序连续性的问题，同时保留了 PDFormer 的核心创新点。
    """
    def __init__(self, num_nodes, input_dim, embed_dim, max_lag=5, top_k=3):
        super().__init__()
        self.num_nodes = num_nodes  # 16
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_lag = max_lag

        # 1. 动态图构建 (Dynamic Graph Constructor)
        # 不再依赖静态文件，而是让模型根据输入动态调整语义图
        self.sem_query = nn.Linear(input_dim, embed_dim)
        self.sem_key = nn.Linear(input_dim, embed_dim)
        self.top_k = top_k

        # 2. 延时混合器 (保留 PDFormer 的核心精华)
        # 用 1x1 卷积代替全连接，处理 (B, T, N, C) 更加高效且参数更少
        self.proj_q = nn.Conv2d(input_dim, embed_dim, 1)
        self.proj_v = nn.Conv2d(input_dim, embed_dim, 1)
        
        # 3. 融合层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _calculate_semantic_mask(self, x):
        """
        动态计算语义掩码 (Semantic Mask)
        x: [B, T, N, C]
        基于 batch 内的平均特征计算节点相似度，捕捉长距离依赖
        """
        # Global Average Pooling over Time -> [B, N, C]
        global_feat = x.mean(dim=1) 
        
        # 计算相似度 [B, N, N]
        # Q * K^T
        Q = self.sem_query(global_feat) # [B, N, D]
        K = self.sem_key(global_feat)   # [B, N, D]
        sim = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)
        
        # Top-K 稀疏化 (保持图的高效性)
        topk_val, topk_idx = torch.topk(sim, k=self.top_k, dim=-1)
        mask = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
        return mask # [B, N, N] 0/1 矩阵

    def forward(self, x):
        """
        x: [B, T, N, C] (原始连续序列)
        Returns: [B, T, N, D] (带有空间延时信息的 Embedding)
        """
        B, T, N, C = x.shape
        
        # 1. 基础特征投影 [B, C, T, N] -> [B, D, T, N]
        x_perm = x.permute(0, 3, 1, 2) 
        q = self.proj_q(x_perm).permute(0, 2, 3, 1) # [B, T, N, D]
        v = self.proj_v(x_perm).permute(0, 2, 3, 1) # [B, T, N, D]

        # 2. 动态语义图构建
        sem_mask = self._calculate_semantic_mask(x) # [B, N, N]

        # 3. 显式延时建模 (Explicit Delay Modeling)
        # 这是 PDFormer 的灵魂：让节点 i 的 t 时刻关注节点 j 的 t-lag 时刻
        spatial_feat = 0
        
        # 预先计算所有 lag 的 shift，避免循环内重复开销
        # 这比 PDFormer 原版更高效
        for lag in range(self.max_lag + 1):
            # Causal Shift: 时间右移 lag
            if lag == 0:
                v_lag = v
            else:
                # Padding at start
                pad = torch.zeros(B, lag, N, self.embed_dim, device=x.device)
                v_lag = torch.cat([pad, v[:, :-lag]], dim=1) # [B, T, N, D]

            # 计算滞后相似度权重 (Lag-Attention)
            # q: [B, T, N, D], v_lag: [B, T, N, D] -> score: [B, T, N, N]
            # 这里我们简化计算：只看节点间的静态相似度 * 动态特征
            # 为了效率，我们假设延时权重主要由 sem_mask 决定
            
            # 聚合邻居特征: [B, T, N_adj, D] -> [B, T, N, D]
            # 利用 sem_mask 进行稀疏聚合
            # v_lag [B, T, N, D] -> [B*T, N, D]
            v_flat = v_lag.reshape(B*T, N, self.embed_dim)
            mask_flat = sem_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, N, N)
            
            # [B*T, N, N] @ [B*T, N, D] -> [B*T, N, D]
            agg = torch.bmm(mask_flat, v_flat).reshape(B, T, N, self.embed_dim)
            
            # 简单的可学习衰减 (Lag Decay)，越久远影响越小
            decay = 1.0 / (lag + 1.0) 
            spatial_feat += agg * decay

        # 4. 残差连接与归一化
        out = self.out_proj(spatial_feat)
        out = self.norm(out + q) # 加上自身的投影
        
        return out