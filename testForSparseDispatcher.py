import torch
import torch.nn as nn
import numpy as np
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()
    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    

inputs = torch.randn(4, 4, 2)
weights = torch.randn(4, 3)
zeros = torch.zeros(4, 3)
top_logits, top_indices = torch.topk(weights, 2)
top_logits = top_logits.softmax(dim=-1)
gates = zeros.scatter(1, top_indices, top_logits)
dispatcher = SparseDispatcher(4, gates)
experts = dispatcher.dispatch(inputs)
output = dispatcher.combine(experts)
# print(dispatcher.combine(experts).shape)
# print(dispatcher.expert_to_gates())
