## bicubic baseline on the validation split: the 800m field already reprojected (cubic) onto
## the 160m grid by preprocess.py, i.e. input channel 0, compared directly against the 160m target.
## python3 scripts/baseline_test.py --config configs/step1a_cnn.yaml --fold val2025

import argparse

import torch
import yaml

from raid.data.dataset import build_dataloaders
from raid.losses import masked_mse
from raid.config import apply_fold, dict_to_namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=str, required=True, help="CV fold name from split.folds, e.g. val2025")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = dict_to_namespace(apply_fold(yaml.safe_load(f), args.fold))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = build_dataloaders(cfg)
    var = cfg.data.target_vars[0]
    std = val_loader.dataset.std[var]

    ## sums over all sea cells of all validation days, so every cell weighs the same
    sq_err, err, n_cells = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            diff = (x[:, :1] - y) * mask
            sq_err += (diff**2).sum().item()
            err += diff.sum().item()
            n_cells += mask.sum().item()

    mse = sq_err / n_cells
    print(f"bicubic baseline on {len(val_loader.dataset)} val days ({var})")
    print(f"  MSE (normalised): {mse:.6f}")
    print(f"  RMSE: {mse**0.5 * std:.4f} degC")
    print(f"  bias: {err / n_cells * std:+.4f} degC  (input - target)")


if __name__ == "__main__":
    main()
