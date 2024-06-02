import torch
from torch import nn



class naive(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, tgt):
        out = tgt[:,0].unsqueeze(-1).repeat(1,tgt.shape[1]) # We repeat for x number of times

        return out


# This function takes a constant during initialization and simply returns this value every time. Used for median
class const(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, src, tgt):
        return torch.full_like(tgt, self.val)
