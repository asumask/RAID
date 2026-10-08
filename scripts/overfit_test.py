## sanity check: a fresh model must drive the loss on ONE fixed training batch towards 0.
## python3 scripts/overfit_test.py --config configs/step1a_cnn.yaml --steps 500 --lr 1e-3

import argparse

import torch
import yaml

from raid.data.dataset import build_dataloaders
from raid.losses import masked_mse
from train import build_model, dict_to_namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    x, y, mask = next(iter(train_loader))  # one batch, kept fixed for every step
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    print(f"batch: x {tuple(x.shape)}, y {tuple(y.shape)}, sea fraction {mask.mean():.2f}")
    print(f"baseline (input channel 0) loss: {masked_mse(x[:, :1], y, mask).item():.6f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for i in range(args.steps + 1):
        opt.zero_grad()
        loss = masked_mse(model(x), y, mask)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"step {i:4d}  loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
