import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LogSparseMask():
    def __init__(self, B, H, L_q, L_k, device="cpu"):
        mask_shape = [B, H, L_q, L_k]
        with torch.no_grad():
            log_p = []
            for i in range(14):
                log_p.append(2 ** i)
            self._mask = torch.ones(mask_shape, dtype=torch.bool).to(device)
            for i in range(L_q):
                self._mask[..., i, i] = False
                for p in log_p:
                    if i - p >= 0:
                        self._mask[..., i, i - p] = False
                for p in log_p:
                    if i + p < L_k:
                        self._mask[..., i, i + p] = False

    @property
    def mask(self):
        return self._mask

import numpy as np


class RandomMasking:
    pass


# class SegmentWiseMasking:
#     def __init__(self, length, mask_ratio, device='cpu'):
#         """
#         Initialize SegmentWiseMasking class.
        
#         Args:
#         - length (int): Length of the time series.
#         - mask_ratio (float): Ratio of masked values in the time series.
#         - device (str): Device to perform computation, 'cpu' or 'cuda'.
#         """
#         self.length = length
#         self.mask_ratio = mask_ratio
#         self.device = device
#         self._mask = None  # Private variable to store the generated mask
    
#     @property
#     def mask(self):
#         """
#         Return the generated mask.
#         """
#         if self._mask is None:
#             with torch.no_grad():
#                 self._generate_mask()
#         return self._mask
    
#     def _generate_mask(self):
#         """
#         Generate segment-wise masking with two-state DTMC.
#         """
#         # Generate a random sequence of 0s and 1s with approximately the desired mask_ratio
#         num_masked = int(self.length * self.mask_ratio)
#         mask_sequence = np.zeros(self.length, dtype=int)
#         mask_sequence[:num_masked] = 1
#         np.random.shuffle(mask_sequence)
        
#         # Define transition probabilities for the two-state DTMC
#         transition_probs = np.array([[0.9, 0.1],  # Probability of transitioning from state 0
#                                      [0.1, 0.9]]) # Probability of transitioning from state 1
        
#         # Initialize the mask array
#         mask = torch.zeros(self.length, dtype=torch.int, device=self.device)
        
#         # Generate mask segments based on the two-state DTMC
#         state = 0  # Initial state
#         for i in range(self.length):
#             mask[i] = state
#             state = np.random.choice([0, 1], p=transition_probs[state])
        
#         # Apply the mask sequence to the generated segments
#         masked_segments = []
#         segment_start = 0
#         for i in range(1, self.length):
#             if mask_sequence[i] != mask_sequence[i-1]:  # Segment boundary
#                 masked_segments.append(mask_sequence[i-1] * torch.ones(i - segment_start, dtype=torch.int, device=self.device))
#                 segment_start = i
#         # Add the last segment
#         masked_segments.append(mask_sequence[-1] * torch.ones(self.length - segment_start, dtype=torch.int, device=self.device))
        
#         # Concatenate masked segments to form the final mask
#         mask = torch.cat(masked_segments)

#         # transform to boolean
#         mask = mask.bool()
        
#         self._mask = ~mask

class SegmentWiseMasking:
    def __init__(self, length, mask_ratio, device='cpu'):
        """
        Initialize SegmentWiseMasking class.
        
        Args:
        - length (int): Length of the time series.
        - mask_ratio (float): Ratio of masked values in the time series.
        - device (str): Device to perform computation, 'cpu' or 'cuda'.
        """
        self.length = length
        self.mask_ratio = mask_ratio
        self.device = device
        self._mask = None  # Private variable to store the generated mask
    
    @property
    def mask(self):
        """
        Return the generated mask.
        """
        if self._mask is None:
            self._generate_mask()
        return self._mask
    
    def _generate_mask(self):
        """
        Generate segment-wise masking with two-state DTMC.
        """
        with torch.no_grad():
            # Generate a random sequence of 0s and 1s with approximately the desired mask_ratio
            num_masked = int(self.length * self.mask_ratio)
            mask_sequence = torch.zeros(self.length, dtype=torch.int, device=self.device)
            mask_sequence[:num_masked] = 1
            mask_sequence = mask_sequence[torch.randperm(self.length)]
            
            # Define transition probabilities for the two-state DTMC
            transition_probs = torch.tensor([[0.9, 0.1],  # Probability of transitioning from state 0
                                            [0.1, 0.9]], device=self.device)
            
            # Generate mask based on the two-state DTMC
            states = torch.zeros(self.length, dtype=torch.int, device=self.device)
            transitions = torch.randint(0, 2, (self.length,), device=self.device)
            
            stay_probs = (1 - transitions) * transition_probs[states, states]
            change_probs = transitions * transition_probs[states, 1 - states]
            
            states = torch.where(torch.rand(self.length, device=self.device) < stay_probs, states, 1 - states)
            states = torch.where(torch.rand(self.length, device=self.device) < change_probs, 1 - states, states)
            
            # Apply the mask sequence to the generated segments
            segment_starts = torch.cat((torch.tensor([0], device=self.device), torch.nonzero(mask_sequence[:-1] != mask_sequence[1:]).squeeze() + 1))
            segment_ends = torch.cat((segment_starts[1:], torch.tensor([self.length], device=self.device)))
            
            masked_segments = [mask_sequence[start:end] for start, end in zip(segment_starts, segment_ends)]
            mask = torch.cat(masked_segments)
            
            mask = mask.bool()

            self._mask = ~mask
    
# Example usage:
# length = 100
# mask_ratio = 0.3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# masking = SegmentWiseMasking(length, mask_ratio, device=device)
# mask = masking.mask
# print("Generated Mask:", ~mask)
# print("Actual Mask Ratio:", torch.mean(mask.float()))