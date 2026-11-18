"""Parameter-free timm layer equivalents using the installed PyTorch."""
import torch
from torch import nn
from torch.nn.init import trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
        return x.div(keep_prob) * mask
