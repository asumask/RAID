## evaluate a trained run on the test split: model vs bicubic baseline (input channel 0), in degC.
## inputs are normalised with the run's own training stats (norm_stats.npz), as in training.
## python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025


## usage: 
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2024
#Results go to <run_dir>/eval_test
## --fold sets the run dir (<output_dir>/<fold>) and the fold's val years, for scoring CV runs:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold val2024 --split val
## --fold ensemble averages the predictions of all fold models (test split only),
## results go to <output_dir>/ensemble/eval_test:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold ensemble
## --output_dir overrides output_dir from the config (same as train.py), e.g. for runs kept elsewhere:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold ensemble --output_dir /some/where/unet_try2

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import DownscalingDataset, denormalize
from raid.log import tee_output
from raid.metrics import masked_bias, masked_rmse
from raid.models import build_model



def load_stats(path, names):
    s = np.load(path)
    return {v: (float(s[f"{v}_mean"]), float(s[f"{v}_std"])) for v in names}


def load_members(cfg, run_dirs, years, device):
    ## one (model, dataset, stats) per run: each model sees inputs normalised with its own training stats
    names = dict.fromkeys(cfg.data.input_vars + cfg.data.target_vars)
    members = []
    for run in run_dirs:
        stats = load_stats(run / "norm_stats.npz", names)
        ds = DownscalingDataset(cfg, years=years, stats=stats)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.eval()
        members.append((model, ds, stats))
    dates = members[0][1].dates
    for run, (_, ds, _) in zip(run_dirs, members):
        assert ds.dates == dates, f"{run}: different days than {run_dirs[0]}"
    return members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="override output_dir from config, as in train.py: runs are read from "
                             "<output_dir>[/<fold>], results written next to them")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="dir with best.pt + norm_stats.npz (default: output_dir from config)")
    parser.add_argument("--split", type=str, default="test", help="split name from the config")
    parser.add_argument("--fold", type=str, default=None,
                        help="CV fold name from split.folds (run dir <output_dir>/<fold>, val = the fold's val "
                             "years), or 'ensemble' to average all fold models (test split only)")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
    ensemble = args.fold == "ensemble"
    if ensemble:
        ## each fold model was trained on the other folds' val years, so only the test split is unseen by all
        if args.split != "test":
            parser.error("--fold ensemble only works with --split test")
        if args.run_dir is not None:
            parser.error("--fold ensemble reads <output_dir>/<fold> for every fold; --run_dir is not used")
        run_dirs = [Path(raw["output_dir"]) / name for name in raw["split"]["folds"]]
        out = Path(raw["output_dir"]) / "ensemble" / f"eval_{args.split}"
    else:
        if args.fold is not None:
            apply_fold(raw, args.fold)
        run_dirs = [Path(args.run_dir or raw["output_dir"])]
        out = run_dirs[0] / f"eval_{args.split}"
    cfg = dict_to_namespace(raw)
    out.mkdir(parents=True, exist_ok=True)
    tee_output(out / "evaluate.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    members = load_members(cfg, run_dirs, getattr(cfg.split, args.split).years, device)
    loaders = [DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)
               for _, ds, _ in members]
    ds, stats = members[0][1], members[0][2]     # target and baseline in degC are the same for every member

    var, in_var = cfg.data.target_vars[0], cfg.data.input_vars[0]
    sea = ds.sea_mask.numpy()
    T, (H, W) = len(ds), sea.shape
    ## (T, H, W) degC, land = NaN; written day by day so the whole array never sits in RAM
    pred_all = np.lib.format.open_memmap(out / f"pred_{var}.npy", mode="w+", dtype=np.float32, shape=(T, H, W))
    ## per day: rmse_model, rmse_baseline, bias_model, bias_baseline (bias = pred - target)
    daily = np.zeros((T, 4), np.float32)
    sq_model, sq_base, bias_sum = (np.zeros((H, W)) for _ in range(3))

    with torch.no_grad():
        for i, batches in enumerate(zip(*loaders)):
            ## ensemble = mean of the members' predictions in degC (a single run is a 1-member ensemble)
            p = np.mean([denormalize(model(x.to(device))[0, 0].cpu().numpy(), *st[var])
                         for (model, _, st), (x, _, _) in zip(members, batches)], axis=0)
            x, y, _ = batches[0]
            t = denormalize(y[0, 0].numpy(), *stats[var])
            b = denormalize(x[0, 0].numpy(), *stats[in_var])   # bicubic baseline
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
    print(f"{args.split}{f' (fold {args.fold})' if args.fold else ''}: {T} days ({ds.dates[0]} -> {ds.dates[-1]}), "
          f"run{'s' if ensemble else ''} {', '.join(map(str, run_dirs))}")
    print(f"  model    RMSE {rmse_m:.4f} degC  bias {daily[:, 2].mean():+.4f}")
    print(f"  baseline RMSE {rmse_b:.4f} degC  bias {daily[:, 3].mean():+.4f}")
    print(f"  skill vs baseline: {1 - rmse_m / rmse_b:+.1%}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
