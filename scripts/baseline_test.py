import torch
import torch.nn.functional as F
from raid.trainer import masked_mse   # works because trainer.py uses it

# ---- paste from train.py main(): cfg, train_loader, val_loader ----
device = torch.device("cuda")

tot, n = 0.0, 0
with torch.no_grad():
    for x, y, mask in val_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        up = F.interpolate(x[:, :y.shape[1]], size=y.shape[-2:],
                           mode="bicubic", align_corners=False)
        tot += masked_mse(up, y, mask).item(); n += 1
print("bicubic baseline val MSE:", tot / n)


