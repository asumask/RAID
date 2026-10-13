import sys
import argparse
import yaml
import torch

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import build_dataloaders
from raid.log import tee_output
from raid.models import build_model
from raid.trainer import fit


# usage: CUDA_VISIBLE_DEVICES=1 
#        python3 scripts/train.py --config /home/a22740/asumak/gitlab/raid/configs/step1a_cnn.yaml 
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --fold val2025 --output_dir 
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --fold val2024 --output_dir 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--fold", type=str, default=None,
                        help="CV fold name from split.folds, e.g. val2025 or val2024")    
    
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
    if args.fold is not None:
        apply_fold(cfg, args.fold)

    test_years = set(cfg["split"].get("test", {}).get("years", []))
    train_years, val_years = set(cfg["split"]["train"]["years"]), set(cfg["split"]["val"]["years"])
    assert not (train_years & val_years), f"years {train_years & val_years} in both train and val"
    used = train_years | val_years
    assert not (used & test_years), f"test years {used & test_years} used in train/val"

    return dict_to_namespace(cfg)


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tee_output(f"{cfg.output_dir}/train.log")
    print(f"device {device}  train {cfg.split.train.years}  val {cfg.split.val.years}  output_dir {cfg.output_dir}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    fit(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()