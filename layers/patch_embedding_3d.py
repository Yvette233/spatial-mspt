# MSPT/layers/patch_embedding_3d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    """
    3D Spatio-Temporal Patch Embedding

    Input (two supported formats):
        x: (B, T, H, W, C)  -- default input_format='BTHWC'
        OR
        x: (B, C, T, H, W)  -- input_format='BCTHW'

    Args:
        t_patch (int): temporal patch size (patch length in time), e.g. Sm (period)
        h_patch (int): spatial patch height
        w_patch (int): spatial patch width
        embed_dim (int): output embedding dimension D
        input_format (str): 'BTHWC' or 'BCTHW'
        use_pos (bool): whether to add learnable positional embeddings (time + space + var)
    Returns:
        H: tensor of shape (B, C, Nt, Ps, D)
            - C: number of variables/channels in input (original last dim if BTHWC)
            - Nt: number of time patches
            - Ps: number of spatial patches per time patch (Nh * Nw)
            - D: embed_dim
    Notes:
        - This implementation treats each variable/channel independently by reshaping (B*C,1,T,H,W)
          and applying a Conv3d(in=1, out=embed_dim) to produce per-variable embeddings.
        - Non-overlapping patches: conv3d kernel_size = (t_patch, h_patch, w_patch), stride same.
        - If T/H/W not divisible by patch sizes, it pads at the end with zeros (can be changed).
    """
    def __init__(self, t_patch=1, h_patch=1, w_patch=1, embed_dim=64,
                 input_format='BTHWC', use_pos=True):
        super().__init__()
        assert input_format in ('BTHWC', 'BCTHW')
        self.t_patch = t_patch
        self.h_patch = h_patch
        self.w_patch = w_patch
        self.embed_dim = embed_dim
        self.input_format = input_format
        self.use_pos = use_pos

        

        # Conv3d applied on per-variable mini-batch (B*C, 1, T, H, W) -> (B*C, D, Nt, Nh, Nw)
        self.conv = nn.Conv3d(in_channels=1, out_channels=embed_dim,
                              kernel_size=(t_patch, h_patch, w_patch),
                              stride=(t_patch, h_patch, w_patch),
                              padding=0, bias=True)

        # small linear to project if needed (kept for compatibility)
        self.proj = nn.Identity()

        if use_pos:
            # Positional embeddings will be created on first forward when sizes known
            self.time_pos = None   # shape (Nt, D)
            self.space_pos = None  # shape (Ps, D)
            self.var_pos = None    # shape (C, D)
            # we keep as nn.Parameter placeholders optionally created in reset_pos
        else:
            self.time_pos = None
            self.space_pos = None
            self.var_pos = None

    def _pad_to_divisible(self, x, t_patch, h_patch, w_patch):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        pad_t = ( ( (T + t_patch - 1) // t_patch) * t_patch ) - T
        pad_h = ( ( (H + h_patch - 1) // h_patch) * h_patch ) - H
        pad_w = ( ( (W + w_patch - 1) // w_patch) * w_patch ) - W
        # padding order for F.pad: (W_left, W_right, H_left, H_right, T_left, T_right)
        if (pad_t, pad_h, pad_w) == (0,0,0):
            return x, 0, 0, 0
        pad = (0, pad_w, 0, pad_h, 0, pad_t)
        x = F.pad(x, pad, mode='constant', value=0)
        return x, pad_t, pad_h, pad_w

    def reset_positional_embeddings(self, C, Nt, Nh, Nw, device):
        Ps = Nh * Nw
        # create learnable params
        self.time_pos = nn.Parameter(torch.zeros(1, 1, Nt, 1, self.embed_dim, device=device))
        self.space_pos = nn.Parameter(torch.zeros(1, 1, 1, Ps, self.embed_dim, device=device))
        self.var_pos = nn.Parameter(torch.zeros(1, C, 1, 1, self.embed_dim, device=device))
        # initialize
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.space_pos, std=0.02)
        nn.init.trunc_normal_(self.var_pos, std=0.02)

    def forward(self, x):
        """
        x can be (B, T, H, W, C) or (B, C, T, H, W) depending on input_format
        returns H: (B, C, Nt, Ps, D)
        """
        if self.input_format == 'BTHWC':
            # to (B, C, T, H, W)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        elif self.input_format == 'BCTHW':
            # already (B, C, T, H, W)
            pass
        else:
            raise ValueError('unsupported input_format')

        B, C, T, H, W = x.shape
        device = x.device

        # pad so dimensions divisible by patch sizes
        x, pad_t, pad_h, pad_w = self._pad_to_divisible(x, self.t_patch, self.h_patch, self.w_patch)
        B, C, Tp, Hp, Wp = x.shape

        Nt = Tp // self.t_patch
        Nh = Hp // self.h_patch
        Nw = Wp // self.w_patch
        Ps = Nh * Nw

        # reshape to (B*C, 1, T, H, W) to run Conv3d per-variable
        x_resh = x.reshape(B * C, 1, Tp, Hp, Wp)
        # conv -> (B*C, D, Nt, Nh, Nw)
        conv_out = self.conv(x_resh)  # (B*C, D, Nt, Nh, Nw)
        # reshape -> (B, C, D, Nt, Nh, Nw)
        conv_out = conv_out.view(B, C, self.embed_dim, Nt, Nh, Nw)
        # move embedding dim to last and flatten spatial patches:
        # (B, C, Nt, Ps, D)
        conv_out = conv_out.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B,C,Nt,Nh,Nw,D)
        conv_out = conv_out.view(B, C, Nt, Ps, self.embed_dim)

        H = self.proj(conv_out)  # identity by default

        # positional embeddings
        if self.use_pos:
            if (self.time_pos is None) or (self.space_pos is None) or (self.var_pos is None):
                # initialize learnable pos once with correct sizes
                self.reset_positional_embeddings(C=C, Nt=Nt, Nh=Nh, Nw=Nw, device=device)
            # broadcast and sum:
            # time_pos shape (1,1,Nt,1,D), space_pos (1,1,1,Ps,D), var_pos (1,C,1,1,D)
            H = H + self.time_pos + self.space_pos + self.var_pos

        return H  # (B, C, Nt, Ps, D)


# ---------------- Demo / unit test ----------------
if __name__ == "__main__":
    # Small test to validate shapes
    B = 2
    T = 10
    H = 8
    W = 12
    C = 3
    t_patch = 5
    h_patch = 4
    w_patch = 4
    D = 32

    # Random input B,T,H,W,C
    x = torch.randn(B, T, H, W, C)
    pe = PatchEmbedding3D(t_patch=t_patch, h_patch=h_patch, w_patch=w_patch, embed_dim=D, input_format='BTHWC', use_pos=True)
    out = pe(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # expected (B, C, Nt=2, Ps= (Hp/h_patch)*(Wp/w_patch), D)
