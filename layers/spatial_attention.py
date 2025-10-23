# MSPT/layers/spatial_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DelayAwareTransform(nn.Module):
    """
    Lightweight delay-aware memory module.
    Produces a memory vector per (B, C, Ps, D) by aggregating recent Nt' time steps.
    """
    def __init__(self, mem_window=5, mode='mean', proj_dim=None):
        super().__init__()
        assert mode in ('mean', 'ema')
        self.mem_window = mem_window
        self.mode = mode
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(proj_dim, proj_dim)

    def forward(self, H):
        # H: (B, C, Nt, Ps, D)
        B, C, Nt, Ps, D = H.shape
        if Nt == 0:
            return H.new_zeros((B, C, Ps, D))
        k = min(self.mem_window, Nt)
        recent = H[:, :, Nt - k : Nt, :, :]  # (B, C, k, Ps, D)
        if self.mode == 'mean':
            mem = recent.mean(dim=2)  # (B, C, Ps, D)
        else:
            weights = torch.linspace(1.0, 0.2, steps=k, device=H.device)
            weights = weights / weights.sum()
            mem = (recent * weights.view(1, 1, k, 1, 1)).sum(dim=2)
        if self.proj is not None:
            mem_flat = mem.reshape(-1, D)
            mem_flat = self.proj(mem_flat)
            mem = mem_flat.view(B, C, Ps, D)
        return mem  # shape (B, C, Ps, D)

class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention module combining GeoSSA and SemSSA styles and integrating DelayAwareTransform.
    Input H: (B, C, Nt, Ps, D)
    """
    def __init__(self, embed_dim, n_heads=8, dropout=0.1, use_geo=True, use_sem=True, mem_window=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_geo = use_geo
        self.use_sem = use_sem
        self.dropout = dropout

        self.attn_geo = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.attn_sem = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.combine_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.delay_mem = DelayAwareTransform(mem_window=mem_window, mode='mean', proj_dim=embed_dim)

    def _prepare_mask(self, mask, Ps, device):
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            additive = torch.zeros((Ps, Ps), device=device, dtype=torch.float32)
            additive = additive.masked_fill(~mask, float('-1e9'))
            return additive
        mask = mask.to(device).float()
        return mask

    def forward(self, H, geo_mask=None, sem_mask=None):
        """
        Args:
            H: (B, C, Nt, Ps, D)
            geo_mask: (Ps, Ps) or None (boolean or float additive mask)
            sem_mask: (Ps, Ps) or None
        Returns:
            H_out: (B, C, Nt, Ps, D)
        """
        B, C, Nt, Ps, D = H.shape
        device = H.device

        geo_attn_mask = self._prepare_mask(geo_mask, Ps, device) if self.use_geo else None
        sem_attn_mask = self._prepare_mask(sem_mask, Ps, device) if self.use_sem else None

        mem = self.delay_mem(H)  # (B, C, Ps, D)

        H_out = torch.zeros_like(H)
        for n in range(Nt):
            X = H[:, :, n, :, :]  # (B, C, Ps, D)
            X_with_mem = X + mem  # broadcast add

            # reshape to (seq_len=Ps, batch=B*C, D)
            seq = X_with_mem.permute(2, 0, 1, 3).reshape(Ps, B * C, D)
            seq_orig = X.permute(2, 0, 1, 3).reshape(Ps, B * C, D)

            out_pieces = []

            if self.use_geo:
                geo_mask_use = geo_attn_mask
                geo_out, _ = self.attn_geo(query=seq, key=seq, value=seq, attn_mask=geo_mask_use)
                out_pieces.append(geo_out)
            if self.use_sem:
                sem_mask_use = sem_attn_mask
                sem_out, _ = self.attn_sem(query=seq, key=seq, value=seq, attn_mask=sem_mask_use)
                out_pieces.append(sem_out)

            if len(out_pieces) == 0:
                combined = seq
            elif len(out_pieces) == 1:
                combined = out_pieces[0]
            else:
                concat = torch.cat(out_pieces, dim=-1)  # (Ps, B*C, 2D)
                combined = self.combine_proj(concat)    # (Ps, B*C, D)

            combined = combined.reshape(Ps, B, C, D).permute(1, 2, 0, 3)  # (B, C, Ps, D)
            combined = self.layernorm(self.dropout_layer(self.out_proj(combined)) + X)  # residual
            H_out[:, :, n, :, :] = combined

        return H_out

# End of file
