import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out):
    ## GroupNorm, not BatchNorm: training uses batches of 8 patches, validation batch 1 full grids
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth
        chs = [base_channels * 2**i for i in range(depth + 1)]           # 32, 64, 128, 256
        ## enc[0] at full resolution, enc[depth] is the bottleneck
        self.enc = nn.ModuleList([conv_block(in_channels, chs[0])] +
                                 [conv_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.dec = nn.ModuleList([conv_block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.output_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

        # Start exactly at the bicubic baseline: the correction is zero at init.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2**self.depth
        h = F.pad(x, (0, -W % m, 0, -H % m), mode="replicate")

        skips = []
        for i, block in enumerate(self.enc):
            if i > 0:
                h = F.max_pool2d(h, 2)
            h = block(h)
            skips.append(h)
        skips.pop()                                                       # bottleneck = h
        ## bilinear upsampling + conv instead of ConvTranspose2d: no checkerboard artefacts
        for block in self.dec:
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skips.pop()], dim=1))

        return x[:, :1] + self.output_conv(h)[..., :H, :W]


