import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_y, target_y, target_x=None, phil=None, xil=None, phih=None, xih=None, alpha=0.5, beta=0.5):
        if target_x is None:
            return F.mse_loss(input_y, target_y)
        B, H, C = target_y.shape
        B, L, C = target_x.shape
        mse_loss = F.mse_loss(input_y, target_y, reduction='none')
        # print(mse_loss.shape)
        # prior_konwledge_loss = alpha * torch.pow((torch.sum(target, dim=1, keepdim=True) / H) - phih, 2)
        prior_konwledge_loss1 = torch.pow((torch.sum(target_x, dim=1, keepdim=True) / L) - phil, 2) + torch.abs(torch.sum(torch.pow(target_x - phil, 2), dim=1, keepdim=True)/(L-1) - xil)
        prior_konwledge_loss2 = torch.pow((torch.sum(target_y, dim=1, keepdim=True) / H) - phih, 2) + torch.abs(torch.sum(torch.pow(target_y - phih, 2), dim=1, keepdim=True)/(H-1) - xih)
        # print(prior_konwledge_loss.shape)
        return torch.mean(torch.mean(mse_loss + alpha * prior_konwledge_loss1 + beta * prior_konwledge_loss2, dim=1), dim=0)


# input = torch.rand(2, 96, 1)
# target = torch.rand(2, 96, 1)
# phih = torch.rand(2, 1, 1)
# xih = torch.rand(2, 1, 1)

# loss = DishTSLoss()
# loss(input, target, phih, xih)
# print(loss(input, target))

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict

import matplotlib.pyplot as plt
import seaborn as sns

class TILDE_Q(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.gamma = gamma
    
    def ashift_loss(self, input, target):
        # input [B, T, C]
        # target [B, T, C]
        B, T, C = input.shape
        return (torch.sum(torch.abs(1 / T - F.softmax(input - target, dim=1)), dim=1, keepdim=True)).mean()
    
    def phase_loss(self, input, target, top_k=5, p=2):
        # input [B, T, C]
        # target [B, T, C]
        input_f = torch.fft.rfft(input, dim=1)
        target_f = torch.fft.rfft(target, dim=1)
        B, T, C = target_f.shape
        target_f_abs = torch.abs(target_f)
        target_f_abs[:, 0, :] = 0
        _, top_list = torch.topk(target_f_abs, top_k, dim=1) # [B, top_k, C]
        # Create a range tensor of shape [B, T, C] similar to xf's T dimension
        all_indices = torch.arange(T, device=input.device).view(1, T, 1).expand(B, -1, C)
        # Create a mask of shape [B, T, C] where top_k indices are marked with False (or 0)
        mask = torch.ones(B, T, C, dtype=torch.bool, device=input.device)
        mask.scatter_(1, top_list, 0)
        # Get the indices of the non-top_k elements
        non_top_list = all_indices[mask].view(B, T - top_k, C)
        domain_loss = torch.norm(torch.gather(target_f, 1, top_list) - torch.gather(input_f, 1, top_list), p=p, dim=1).mean()
        nondomain_loss = torch.norm(torch.gather(input_f, 1, non_top_list), p=p, dim=1, keepdim=True).mean()
        return domain_loss + nondomain_loss
    
    def amp_loss(self, input, target, p=2):
        # input [B, T, C]
        # target [B, T, C]
        self_corr = F.conv1d(target.permute(0, 2, 1), torch.flip(target.permute(0, 2, 1), dims=[-1]), groups=target.shape[2]).permute(0, 2, 1)
        cross_corr = F.conv1d(target.permute(0, 2, 1), torch.flip(input.permute(0, 2, 1), dims=[-1]), groups=target.shape[2]).permute(0, 2, 1)
        # print(self_corr.shape, cross_corr.shape)
        return torch.norm(self_corr - cross_corr, p=p, dim=1, keepdim=True).mean()
        
    def forward(self, input, target):
        # input [B, H, C]
        # target [B, L, C] 
        print(self.alpha * self.ashift_loss(input, target))
        print((1 - self.alpha) * self.phase_loss(input, target))
        print(self.gamma * self.amp_loss(input, target))
        return (self.alpha * self.ashift_loss(input, target) + (1 - self.alpha) * self.phase_loss(input, target) + self.gamma * self.amp_loss(input, target))


class TILDEQ_LOSS_OFFICIAL(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.gamma = gamma
    
    def ashift_loss(self, outputs, targets):
        T = outputs.size(-1)
        mean, o_mean = targets.mean(dim = -1, keepdim = True), outputs.mean(dim = -1, keepdim = True)
        # reduce the effect of mean value in softmax
        normed_tgt = targets - mean
        normed_out = outputs - o_mean
        #Note: because we need a signed distance function, we use simple negation instead of L1 distance
        loss = torch.mean(torch.sum(torch.abs(1 / T - torch.softmax((normed_tgt - normed_out), dim = -1)), dim = -1))

        return loss

    def phase_loss(self, outputs, targets, batch_x = None):
        # if batch_x is not None:
            # batch_x = batch_x.permute(0,2,1)
            # outputs = torch.cat([batch_x[:, :, -365:], outputs], dim = -1)
            # targets = torch.cat([batch_x[:, :, -365:], targets], dim = -1)
        T = outputs.size(-1)
        out_fourier = torch.fft.fft(outputs, dim = -1) # [B, T]
        tgt_fourier = torch.fft.fft(targets, dim = -1) # [B, T]

        # calculate dominant frequencies
        tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2) # [B, T]
        # filter out the non-dominant frequencies
        mask = (tgt_fourier_sq > (T)).float() # [B, T]
        # guarantee the number of dominant frequencies is equal or greater than T**0.5
        topk_indices = tgt_fourier_sq.topk(k = int(T ** 0.5), dim = -1).indices # [B, T**0.5]
        mask = mask.scatter_(-1, topk_indices, 1.) # [B, T]
        # guarantee that the loss function always considers the mean value
        mask[...,0] = 1. # [B, T]
        mask = torch.where(mask > 0, 1., 0.) # [B, T]
        mask = mask.bool() # [B, T]
        inv_mask = (~mask).float() # [B, T]
        # inv_mask /= torch.mean(inv_mask) # [B, T]
        zero_error = torch.abs(out_fourier) * inv_mask # [B, T]
        zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error) # [B, 1]
        mask = mask.float()
        # mask /= torch.mean(mask)
        ae = torch.abs(out_fourier - tgt_fourier) * mask
        ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
        # loss = (torch.mean(zero_error) / 2 + torch.mean(ae)) / (T ** .5)
        loss = torch.mean(zero_error) + torch.mean(ae)

        return loss

    def amp_loss(self, outputs, targets):
        T = outputs.size(-1)
        fft_size = 1 << (2 * T - 1).bit_length()
        out_fourier = torch.fft.fft(outputs, fft_size, dim = -1)
        tgt_fourier = torch.fft.fft(targets, fft_size, dim = -1)

        out_norm = torch.norm(outputs, dim = -1, keepdim = True)
        tgt_norm = torch.norm(targets, dim = -1, keepdim = True)
        tgt_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim = -1).real
        n_tgt_corr = tgt_corr / (tgt_norm * tgt_norm)

        ccorr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim = -1).real
        n_ccorr = ccorr / (tgt_norm * out_norm)
        loss = torch.mean(torch.abs(n_tgt_corr - n_ccorr))

        return loss
    
    def forward(self, outputs, targets, batch_x = None):
        #outputs = outputs.squeeze(dim = 1)
        outputs = outputs.permute(0,2,1)
        targets = targets.permute(0,2,1)

        assert not torch.isnan(outputs).any(), "Nan value detected!"
        assert not torch.isinf(outputs).any(), "Inf value detected!"

        # print(self.alpha * self.ashift_loss(outputs, targets))
        # print((1 - self.alpha) * self.phase_loss(outputs, targets))
        # print(self.gamma * self.amp_loss(outputs, targets))
        loss = self.alpha * self.ashift_loss(outputs, targets) \
                + (1 - self.alpha) * self.phase_loss(outputs, targets, batch_x) \
                + self.gamma * self.amp_loss(outputs, targets)
        assert loss == loss, "Loss Nan!"
        return loss

# class ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, input, target, mask):
#         # print(input.shape, target.shape, mask.shape)
#         loss = (input - target) ** 2
#         loss = loss.mean(dim=-1)
#         loss = (loss * mask).sum() / mask.sum()
#         return loss

from einops import rearrange
from math import ceil


# class ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.patch_sizes = self.get_patch_sizes(365)
#     def get_patch_sizes(self, seq_len):
#         # get the period list, first element is inf if exclude_zero is False
#         peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
#         patch_sizes = peroid_list.floor().int().unique()
#         return patch_sizes
    
#     def forward(self, inputs, target, masks):
#         # print(input.shape, target.shape, mask.shape)
#         losses = []
#         for input, mask, patch_size in zip(inputs, masks, self.patch_sizes):
#             B, L, C = target.shape
#             temp_target = target.clone()
#             input = rearrange(input, 'B L C -> B C L') # [B, C, L]
#             input = F.pad(input, (0, ceil(L / patch_size) * patch_size - L), mode='constant', value=0)
#             input = input.unfold(-1, patch_size, patch_size)[:, -1, :, :] # [B, L, P]
#             temp_target = rearrange(temp_target, 'B L C -> B C L') # [B, C, L]
#             temp_target = F.pad(temp_target, (0, ceil(L / patch_size) * patch_size - L), mode='constant', value=0)
#             temp_target = temp_target.unfold(-1, patch_size, patch_size)[:, -1, :, :] # [B, L, P]
#             sst_mask = mask[:, -1, :] # [B, L]

#             loss = (input - temp_target) ** 2 # [B, L, P]
#             loss = loss.mean(dim=-1) # [B, L]
#             # if not training:
#             #     print(f"{patch_size}:1:{loss}")
#             loss = (loss * sst_mask).sum() / sst_mask.sum() if sst_mask.sum() >= 1 else loss[0][0]
#             # if not training:
#             #     print(f"{patch_size}:2:{loss}")
#             losses.append(loss) # [B]
#         losses = torch.stack(losses)
#         return losses.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, input, target, mask):
        # print(input.shape, target.shape, mask.shape)
        patch_size = self.patch_size
        B, L, C = target.shape
        input = rearrange(input, 'B L C -> B C L') # [B, C, L]
        input = F.pad(input, (0, ceil(L / patch_size) * patch_size - L), mode='constant', value=0)
        input = input.unfold(-1, patch_size, patch_size)[:, -1, :, :] # [B, L, P]
        target = rearrange(target, 'B L C -> B C L')
        target = F.pad(target, (0, ceil(L / patch_size) * patch_size - L), mode='constant', value=0)
        target = target.unfold(-1, patch_size, patch_size)[:, -1, :, :]
        mask = mask[:, -1, :]

        loss = (input - target) ** 2 # [B, L, P]
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum() if mask.sum() >= 1 else loss[0][0]
        return loss

class MarineHeatwaveMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target, mask):
        loss = (input - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).mean()
        return loss