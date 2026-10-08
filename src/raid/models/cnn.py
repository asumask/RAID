"""Step1A: simple residual CNN for 800m -> 160m temperature downscaling.

Input : (B, in_channels, H, W). Channel 0 must be the normalized 800m
        temperature already upsampled to the 160m grid. The other channels are
        static fields (land mask, bathymetry, ...).
Output: (B, 1, H, W) normalized 160m temperature.

The network predicts a correction that is added to channel 0, so the output at
the start of training equals the upsampled input (the bicubic baseline).
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SimpleSRCNN(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, n_blocks: int = 4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

        # Start exactly at the bicubic baseline: the correction is zero at init.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        feat = self.input_conv(x)
        feat = self.res_blocks(feat)
        return x[:, :1] + self.output_conv(feat)


    