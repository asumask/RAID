## evaluate a trained run on the test split: model vs bicubic baseline (input channel 0), in degC.
## inputs are normalised with the run's own training stats (norm_stats.npz), as in training.
## python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025


## usage: 
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2024
#Results go to <run_dir>/eval_test

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from raid.config import dict_to_namespace
from raid.data.dataset import DownscalingDataset, denormalize
from raid.log import tee_output
from raid.metrics import masked_bias, masked_rmse
from raid.models import build_model



def load_stats(path, names):
    s = np.load(path)
    return {v: (float(s[f"{v}_mean"]), float(s[f"{v}_std"])) for v in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="dir with best.pt + norm_stats.npz (default: output_dir from config)")
    parser.add_argument("--split", type=str, default="test", help="split name from the config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
    run = Path(args.run_dir or cfg.output_dir)
    out = run / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    tee_output(out / "evaluate.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_vars, tgt_vars = cfg.data.input_vars, cfg.data.target_vars
    stats = load_stats(run / "norm_stats.npz", dict.fromkeys(in_vars + tgt_vars))
    ds = DownscalingDataset(cfg, years=getattr(cfg.split, args.split).years, stats=stats)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    var = tgt_vars[0]
    sea = ds.sea_mask.numpy()
    T, (H, W) = len(ds), sea.shape
    ## (T, H, W) degC, land = NaN; written day by day so the whole array never sits in RAM
    pred_all = np.lib.format.open_memmap(out / f"pred_{var}.npy", mode="w+", dtype=np.float32, shape=(T, H, W))
    ## per day: rmse_model, rmse_baseline, bias_model, bias_baseline (bias = pred - target)
    daily = np.zeros((T, 4), np.float32)
    sq_model, sq_base, bias_sum = (np.zeros((H, W)) for _ in range(3))

    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            p = denormalize(model(x.to(device))[0, 0].cpu().numpy(), *stats[var])
            t = denormalize(y[0, 0].numpy(), *stats[var])
            b = denormalize(x[0, 0].numpy(), *stats[in_vars[0]])   # bicubic baseline
            pred_all[i] = np.where(sea, p, np.nan)
            daily[i] = (masked_rmse(p, t, sea), masked_rmse(b, t, sea),
                        masked_bias(p, t, sea), masked_bias(b, t, sea))
            sq_model += (p - t) ** 2
            sq_base += (b - t) ** 2
            bias_sum += p - t
    pred_all.flush()

    land = lambda a: np.where(sea, a, np.nan).astype(np.float32)
    np.save(out / "dates.npy", np.array(ds.dates))                       # (T,) 'YYYYMMDD'
    np.save(out / "daily_metrics.npy", daily)                            # (T, 4), columns above
    np.save(out / f"rmse_map_model_{var}.npy", land(np.sqrt(sq_model / T)))
    np.save(out / f"rmse_map_baseline_{var}.npy", land(np.sqrt(sq_base / T)))
    np.save(out / f"bias_map_model_{var}.npy", land(bias_sum / T))

    ## same sea mask every day, so the overall RMSE is the root of the mean daily MSE
    rmse_m, rmse_b = np.sqrt((daily[:, :2] ** 2).mean(0))
    print(f"{args.split}: {T} days ({ds.dates[0]} -> {ds.dates[-1]}), run {run}")
    print(f"  model    RMSE {rmse_m:.4f} degC  bias {daily[:, 2].mean():+.4f}")
    print(f"  baseline RMSE {rmse_b:.4f} degC  bias {daily[:, 3].mean():+.4f}")
    print(f"  skill vs baseline: {1 - rmse_m / rmse_b:+.1%}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
