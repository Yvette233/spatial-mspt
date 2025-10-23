import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================
# 🧭 地理位置编码（GeoPosEncoding）
# ===========================
class GeoPosEncoding(nn.Module):
    """
    二维经纬度位置编码，用于空间注意力。
    可选：通过距离矩阵或格点索引生成。
    """
    def __init__(self, embed_dim, height=8, width=8):
        super(GeoPosEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        # 正弦位置编码（可用于规则网格）
        pe = torch.zeros(height * width, embed_dim)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, H*W, D]

    def forward(self, B):
        return self.pe.repeat(B, 1, 1)  # [B, H*W, D]


# ===========================
# 🌐 空间图卷积层（GraphConv）
# ===========================
class GraphConv(nn.Module):
    """
    基于邻接矩阵的图卷积层
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        # x: [B, N, D], adj: [N, N]
        support = self.fc(x)
        out = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return out


# ===========================
# 🧠 空间自注意力模块（核心创新）
# ===========================
class SpatialSelfAttention(nn.Module):
    """
    多头空间注意力 + 图卷积增强。
    支持可选地理掩码 (geo_mask) 和语义掩码 (sem_mask)
    """
    def __init__(self, embed_dim=64, n_heads=4, mem_window=5,
                 dropout=0.1, use_geo=True, use_sem=True, use_gcn=True):
        super(SpatialSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)
        self.use_geo = use_geo
        self.use_sem = use_sem
        self.use_gcn = use_gcn

        # 多头投影
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 图卷积层（可选）
        if use_gcn:
            self.gcn = GraphConv(embed_dim, embed_dim)

        # 地理位置编码
        if use_geo:
            self.geo_encoding = GeoPosEncoding(embed_dim, height=8, width=8)

        # 输出层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, geo_mask=None, sem_mask=None):
        """
        x: [B, C, Nt, Ps, D]
        geo_mask, sem_mask: [N, N] 邻接矩阵或掩码
        """
        B, C, Nt, Ps, D = x.shape
        x = x.view(B, -1, D)  # [B, N, D], N = C*Nt*Ps

        # === 加入地理位置编码 ===
        if self.use_geo:
            pos_enc = self.geo_encoding(B)
            if pos_enc.size(1) != x.size(1):
                pos_enc = F.interpolate(pos_enc.transpose(1, 2), size=x.size(1), mode='linear').transpose(1, 2)
            x = x + pos_enc

        # === 计算注意力权重 ===
        Q = self.query(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N, N]

        # === 加入掩码 ===
        if geo_mask is not None:
            scores = scores.masked_fill(geo_mask == 0, -1e9)
        if sem_mask is not None:
            scores = scores.masked_fill(sem_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)

        # === 图卷积增强 ===
        if self.use_gcn:
            # 简单构造邻接矩阵：完全图或掩码图
            if geo_mask is not None:
                adj = geo_mask.float()
            else:
                adj = torch.ones(x.size(1), x.size(1), device=x.device)
            out = out + self.gcn(out, adj)

        out = self.out_proj(out)
        out = out.view(B, C, Nt, Ps, D)
        return out


# ===========================
# ✅ 模块测试（独立可运行）
# ===========================
if __name__ == "__main__":
    B, C, Nt, Ps, D = 2, 1, 6, 8, 64
    x = torch.randn(B, C, Nt, Ps, D)
    spa = SpatialSelfAttention(embed_dim=64, n_heads=4, use_geo=True, use_gcn=True)
    out = spa(x)
    print("Input:", x.shape)
    print("Output:", out.shape)
