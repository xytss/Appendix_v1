import torch
import torch.nn as nn

class DFL(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv(x))
