import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class MultiScalePeriodicPatchEmbedding(nn.Module):
    def __init__(self, seq_len, top_k=5, d_model=512, dropout=0., sparsity_threshold=0.01, hidden_size_factor=4):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        # get the patch sizes
        self.patch_sizes = self.get_patch_sizes(seq_len)
        # AFNO1D parameters
        self.freq_seq_len = seq_len // 2
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / d_model
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len, self.freq_seq_len * hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.freq_seq_len * hidden_size_factor, len(self.patch_sizes)))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, len(self.patch_sizes)))
        # Noise parameters
        self.w_noise = nn.Parameter(torch.zeros(seq_len, len(self.patch_sizes)))
        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(nn.Linear(patch_size, d_model, bias=False))
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, ceil(seq_len / patch_size) * patch_size - seq_len)))
        # self.position_embedding = PositionalEmbedding2D(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique()
        return patch_sizes
    
    def afno1d_for_peroid_weights(self, x, training=True, noise_epsilon=1e-2):
        # x [B, L, C]
        B, L, C = x.shape

        x = rearrange(x, 'B L C -> B C L') # [B, C, L] 
        # xf = torch.fft.rfft(x, dim=-1, norm='ortho') # [B, C, L//2+1]
        xf = torch.fft.rfft(x, dim=-1) # [B, C, L//2+1]
        xf_no_zero = xf[:, :, 1:] # [B, C, L//2]

        o1_real = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, C, self.freq_seq_len * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros([B, C, len(self.patch_sizes)], device=x.device)
        o2_imag = torch.zeros([B, C, len(self.patch_sizes)], device=x.device)

        o1_real = F.relu(
            torch.einsum('...i,io->...o', xf_no_zero.real, self.w1[0]) - \
            torch.einsum('...i,io->...o', xf_no_zero.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...i,io->...o', xf_no_zero.imag, self.w1[0]) + \
            torch.einsum('...i,io->...o', xf_no_zero.real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real = (
            torch.einsum('...i,io->...o', o1_real, self.w2[0]) - \
            torch.einsum('...i,io->...o', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = (
            torch.einsum('...i,io->...o', o1_imag, self.w2[0]) + \
            torch.einsum('...i,io->...o', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        xf_no_zero = torch.stack([o2_real, o2_imag], dim=-1) # [B, C, L-1, 2]
        # xf_no_zero = F.softshrink(xf_no_zero, lambd=self.sparsity_threshold) # [B, C, L-1, 2]
        xf_no_zero = torch.view_as_complex(xf_no_zero) # [B, C, L-1]


        weights = torch.abs(xf_no_zero) # [B, C, L-1]
        if training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_weights = weights + (torch.randn_like(weights) * noise_stddev)
            weights = noisy_weights.mean(dim=-2) # [B, L-1]
        else:
            weights = weights.mean(dim=-2) # [B, L-1]
        
        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(weights.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/weights.png')

        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1) # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1) # [B, top_k]

        zeros = torch.zeros_like(weights) # [B, Ps]
        gates = zeros.scatter_(-1, top_indices, top_weights) # [B, Ps]

        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(gates.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/gates.png')

        return gates # [B, Ps]

    def dispatcher(self, x, gates):
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # get according batch index for each expert
        _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        _part_sizes = (gates > 0).sum(0).tolist()
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        xs = x[_batch_index].squeeze(1)
        return list(torch.split(xs, _part_sizes, dim=0))
    
    def patch_embedding(self, x, patch_size, index_of_patch):
        B, L, C = x.shape
        # do patching
        x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size, patch_size) # [B, C, L//patch_size, patch_size]
        # x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x) # [B, C, L, D]
        x = self.value_embeddings[index_of_patch](x) # [B, C, L, D]
        x = self.position_embedding(x) # [B, C, L, D]
        return self.dropout(x) # [B, C, L, D]

    def forward(self, x):
        B, L, C = x.shape
        gates = self.afno1d_for_peroid_weights(x, self.training) # [B, Ps]
        xs = self.dispatcher(x, gates) # Ps*[B, C, L, D]
        for i, patch_size in enumerate(self.patch_sizes): 
            xs[i] = self.patch_embedding(xs[i], patch_size, i)
        return xs, gates # Ps*[B, C, L, D], [bs, Ps]
        
