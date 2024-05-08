import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True, subtract_last=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine: # args.affine: use affine layers or not
            self.gamma = nn.Parameter(torch.ones(num_features)) # args.n_series: number of series
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        self.avg = torch.mean(batch_x, axis=1, keepdim=True).detach() # b*1*d
        self.var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d
        if self.subtract_last:
            self.last = batch_x[:,-1:,:] # b*1*d

    def forward_process(self, batch_input):
        if self.subtract_last:
            batch_input = batch_input - self.last
        else:
            batch_input = batch_input - self.avg
        batch_input = batch_input / torch.sqrt(self.var + self.eps)
        if self.affine:
            batch_input = batch_input * self.gamma + self.beta
        return batch_input

    def inverse_process(self, batch_input):
        if self.affine:
            batch_input = ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.var + self.eps)
        if self.subtract_last:
            batch_input = batch_input + self.last
        else:
            batch_input = batch_input + self.avg
        return batch_input
