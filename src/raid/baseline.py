import numpy as np

## unit should be backed to degreeC
## use sea cells only
## Compute based on val (validation dat) and test (test day). (do not incl. train)


x = np.load(...)   # 800m_on_160m_grid(upsampled)  (days, H, W)
y = np.load(...)   # 160m(answer)                  (days, H, W)
mask = np.load("data/processed/static/land_mask.npy")  # 1=sea

def masked_rmse(pred, target, mask):
    m = mask.astype(bool)
    diff = (pred - target)[:, m]          # sea cell only
    return np.sqrt((diff ** 2).mean())

def masked_bias(pred, target, mask):
    m = mask.astype(bool)
    return (pred - target)[:, m].mean()

val_days = [...]   # index for validation days
print("RMSE:", masked_rmse(x[val_days], y[val_days], mask))
print("bias:", masked_bias(x[val_days], y[val_days], mask))