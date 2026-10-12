import torch
from pathlib import Path
from raid.losses import masked_mse

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = masked_mse(model(x), y, mask)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        total += masked_mse(model(x), y, mask).item()
    return total / len(loader)

def fit(model, train_loader, val_loader, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    ## decay lr from cfg.train.lr to 0 over the run, stepped once per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = validate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        print(f"epoch {epoch}: train {tr:.4f}  val {va:.4f}  lr {lr:.2e}")
        if va < best:
            best = va
            torch.save(model.state_dict(), out / "best.pt")

