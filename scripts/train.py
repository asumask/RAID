import sys
import argparse
import yaml
import torch
from types import SimpleNamespace

from raid.data.dataset import build_dataloaders
from raid.models.cnn import SimpleSRCNN
from raid.trainer import fit

# usage: python3 scripts/train.py --config /home/a22740/asumak/gitlab/raid/configs/step1a_cnn.yaml --epochs 50
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --overfit_batch
""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--overfit_batch", action="store_true", help="Debug: train on one fixed batch only")
    return parser.parse_args()


def load_config(config_path, args):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    return dict_to_namespace(cfg)


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


def build_model(cfg):
    m = cfg.model
    if m.name == "cnn":
        return SimpleSRCNN(
            in_channels=getattr(m, "in_channels", 3),
            hidden_dim=getattr(m, "hidden_dim", 64),
            n_blocks=getattr(m, "n_blocks", 4),
        )
    raise ValueError(f"Unknown model: {m.name}")

def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    from raid.trainer import fit, overfit_one_batch
    if args.overfit_batch:
        
        """
        import numpy as np
        from raid.losses import masked_mse
        x, y, mask = train_loader.dataset[0]                    # one fixed sample, no random draw
        x, y, mask = x[None], y[None], mask[None]   # add batch dim
        a = masked_mse(x[:, :1], y, mask).item()

        day = train_loader.dataset.dates[0]   # the day string of train_ds[0] — use the dataset's own day list (e.g. train_ds.days[0])
        x_raw = np.load(f"data/processed/800m_on_160m_grid/{day}_temp.npy")
        y_raw = np.load(f"data/processed/160m/{day}_temp.npy")
        m = np.load("data/processed/static/land_mask.npy").astype(bool)
        rmse = np.sqrt(((x_raw - y_raw)[m] ** 2).mean())
        b = (rmse / 3.51723693) ** 2

        print(a, b)
        """
        
        overfit_one_batch(model, train_loader, cfg, device)
        return

    fit(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()