import torch
import torch.nn as nn
import torch.nn.functional as F


# 频域卷积=2pi*时域相乘，时域卷积=频域相乘
class AFFT(nn.Module):
    def __init__(self,):
        super(AFFT, self).__init__()

    def forward(self, x):
        B, L, C = x.size()
        # rfft
        xf = torch.fft.rfft(x, dim=-2, norm='ortho')


class PeriodGuidedMultiScalePatchEmbeding(nn.Module):
    def __init__(self,):
        super(PeriodGuidedMultiScalePatchEmbeding, self).__init__()


class CrossDimensionalDualPeriodicityEncoder(nn.Module):
    def __init__(self,):
        super(CrossDimensionalDualPeriodicityEncoder, self).__init__()

class CrossScaleAggregator(nn.Module):
    def __init__(self,):
        super(CrossScaleAggregator, self).__init__()
    

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

