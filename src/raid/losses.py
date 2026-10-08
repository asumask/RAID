import torch

## errors includes only sea cells.
## the average is computed by dividing a number of sea cells only 

def masked_mse(pred, target, mask):
    # pred, target, mask: (B, 1, H, W)  mask: 1=sea, 0=land
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)

