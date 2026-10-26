## per-cell RMSE vs the 160m target on the test split: the 800m baseline and every model
## (one per CV fold, their ensemble, the full model), from the maps evaluate.py saved:
##   row 1: RMSE map per model, one shared colour scale
##   row 2: RMSE_model - RMSE_baseline (blue = model better than the 800m baseline, red = worse)
## needs evaluate_folds.sh to have run first (it writes <output_dir>/<var>/<fold>/eval_test):
## python3 scripts/rmse_maps.py --config configs/step1b_unet.yaml
## python3 scripts/rmse_maps.py --config configs/step1b_unet.yaml --output_dir /some/where/unet_try2

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

from raid.config import var_dir

## sequential ramp for RMSE (same as error_map.py), diverging blue/red for the change; land in neutral gray
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#3987e5", "#9ec5f4", "#f4f3ef", "#f4a99e", "#e05a47", "#8f1d12"]
LAND = "#d9d8d4"
CMAP = LinearSegmentedColormap.from_list("rmse", BLUE_RAMP)
CMAP.set_bad(LAND)
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", DIVERGING)
DIFF_CMAP.set_bad(LAND)
INK, MUTED = "#1f1f1d", "#6b6a66"
UNITS = {"temp": "degC", "salt": "psu"}


def domain_rmse(rmse_map, sea):
    ## same sea mask every day, so the root of the mean per-cell MSE is the overall RMSE
    return float(np.sqrt((rmse_map[sea] ** 2).mean()))


def load_runs(base, names, var):
    ## per run: model RMSE map; baseline map and dates from the first run, checked against the others
    model, baseline, dates = {}, None, None
    for name in names:
        ev = base / name / "eval_test"
        if not (ev / f"rmse_map_model_{var}.npy").exists():
            raise FileNotFoundError(f"{ev}: no rmse_map_model_{var}.npy, run evaluate_folds.sh first")
        model[name] = np.load(ev / f"rmse_map_model_{var}.npy")
        b, d = np.load(ev / f"rmse_map_baseline_{var}.npy"), np.load(ev / "dates.npy")
        if baseline is None:
            baseline, dates = b, d
        elif not (np.array_equal(d, dates) and np.allclose(b, baseline, equal_nan=True)):
            raise ValueError(f"{ev}: different test days or baseline than {base / names[0]}")
    return model, baseline, dates


def show(fig, ax, data, cmap, vmin, vmax, title):
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def colorbar(fig, im, axes, label, extend):
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, extend=extend)
    cb.set_label(label, color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)


def plot(model, baseline, sea, var, unit, n_days, out_png):
    names = list(model)
    H, W = sea.shape
    panel_w = 3.2
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(panel_w * (len(names) + 1) + 1, 2 * panel_w * H / W + 1.2),
                             facecolor="white", squeeze=False, gridspec_kw={"hspace": 0.35})
    fig.suptitle(f"RMSE vs 160m target, {var}, test split ({n_days} days); colour scales capped at the 99th percentile", color=INK, fontsize=12, x=0.01, ha="left")

    ## row 1: one colour scale for all panels, capped at the pooled 99th percentile
    land = lambda a: np.where(sea, a, np.nan)
    vmax = np.percentile(np.concatenate([baseline[sea]] + [m[sea] for m in model.values()]), 99)
    im = show(fig, axes[0, 0], land(baseline), CMAP, 0, vmax, f"800m baseline\n{domain_rmse(baseline, sea):.3f} {unit}")
    for ax, name in zip(axes[0, 1:], names):
        show(fig, ax, land(model[name]), CMAP, 0, vmax, f"{name}\n{domain_rmse(model[name], sea):.3f} {unit}")
    colorbar(fig, im, axes[0, :], f"RMSE [{unit}]", "max")

    ## row 2: change vs the baseline, symmetric scale around 0 at the pooled 99th percentile of |change|
    diffs = {name: model[name] - baseline for name in names}
    lim = np.percentile(np.abs(np.concatenate([d[sea] for d in diffs.values()])), 99)
    axes[1, 0].axis("off")
    for ax, name in zip(axes[1, 1:], names):
        better = (diffs[name][sea] < 0).mean()
        im = show(fig, ax, land(diffs[name]), DIFF_CMAP, -lim, lim, f"{name} - baseline\nbetter on {better:.0%} of sea")
    colorbar(fig, im, axes[1, :], f"change [{unit}], < 0 = better", "both")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="override output_dir from config, as in evaluate.py: runs are read from "
                             "<output_dir>/<var>/<fold>/eval_test")
    parser.add_argument("--out", type=str, default=None, help="output dir (default: <output_dir>/<var>/rmse_maps)")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
    var = raw["data"]["target_vars"][0]
    unit = UNITS.get(var, "")
    base = Path(var_dir(raw))
    names = list(raw["split"]["folds"]) + ["ensemble", "full"]
    out = Path(args.out or base / "rmse_maps")
    out.mkdir(parents=True, exist_ok=True)

    model, baseline, dates = load_runs(base, names, var)
    sea = np.isfinite(baseline)
    print(f"{var}, test split: {len(dates)} days ({dates[0]} -> {dates[-1]}), runs in {base}")
    print(f"  {'':<10} {'RMSE ' + unit:>9}  {'vs baseline':>11}  {'better cells':>12}")
    print(f"  {'baseline':<10} {domain_rmse(baseline, sea):>9.4f}")
    for name in names:
        r = domain_rmse(model[name], sea)
        print(f"  {name:<10} {r:>9.4f}  {r - domain_rmse(baseline, sea):>+11.4f}  "
              f"{(model[name][sea] < baseline[sea]).mean():>12.0%}")

    np.savez(out / f"rmse_maps_{var}_test.npz", baseline=baseline, sea=sea, dates=dates,
             **{f"rmse_{n}": model[n] for n in names}, **{f"diff_{n}": model[n] - baseline for n in names})
    png = out / f"rmse_maps_{var}_test.png"
    plot(model, baseline, sea, var, unit, len(dates), png)
    print(f"\nsaved {png}\n      {out / f'rmse_maps_{var}_test.npz'}")


if __name__ == "__main__":
    main()
