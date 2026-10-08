import torch
from raid.trainer import masked_mse

# ---- paste from train.py main(): cfg, train_loader, model (fresh, untrained) ----
device = torch.device("cuda")
model = model.to(device)

x, y, mask = next(iter(train_loader))
x, y, mask = x.to(device), y.to(device), mask.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(500):
    opt.zero_grad()
    loss = masked_mse(model(x), y, mask)
    loss.backward(); opt.step()
    if i % 50 == 0: print(i, loss.item())
    