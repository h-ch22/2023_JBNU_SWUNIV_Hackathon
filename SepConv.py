import torch.nn as nn
import torch

from MBConv import MBConv
from Swish import Swish
from SEBlock import SEBlock


class SepConv(nn.Module):
    expand = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )
        self.se = SEBlock(in_channels * SepConv.expand, se_scale)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )
        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)
        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x

        return x