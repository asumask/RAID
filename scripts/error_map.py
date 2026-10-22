## where does the baseline error live? time-mean of |target - reprojected 800m| over a split,
## mapped and broken down by distance to coast, depth and bathymetry slope.
##   coastal / fjord / steep-slope structure -> predictable from static fields (add them as inputs)
##   spread over open water (eddies, filaments) -> mostly chaotic; MSE floor is near the baseline
## python3 scripts/error_map.py --config configs/step1a_cnn.yaml --split val --fold val2025
## python3 scripts/error_map.py --config configs/step1a_cnn.yaml --split test

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

from raid.config import apply_fold
from raid.data.file_utils import daily_filename

CELL_KM = 0.16  # 160m grid spacing

## single-hue sequential ramp (light = small error, dark = large); land drawn in neutral gray
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("err", BLUE_RAMP)
CMAP.set_bad("#d9d8d4")
LINE = "#2a78d6"
INK, MUTED, GRID = "#1f1f1d", "#6b6a66", "#e6e5e1"

DIST_BINS = [0, 0.5, 1, 2, 5, 10, np.inf]  # km
DEPTH_BINS = [0, 25, 50, 100, 200, 400, np.inf]  # m
SLOPE_BINS = [0, 0.005, 0.01, 0.02, 0.05, 0.1, np.inf]  # m/m


def find_dates(dir_800m, dir_160m, var, years):
    years = {str(y) for y in years}
    dates = []
    for path in sorted(dir_800m.glob(daily_filename(var, "[0-9]" * 8))):
        date = path.name[:8]
        if date[:4] in years and (dir_160m / daily_filename(var, date)).exists():
            dates.append(date)
    return dates


def time_mean_errors(dir_800m, dir_160m, var, dates):
    ## running sums in float64; one day in memory at a time
    abs_sum = sq_sum = bias_sum = count = None
    for date in dates:
        diff = (np.load(dir_160m / daily_filename(var, date)) - np.load(dir_800m / daily_filename(var, date))).astype(np.float64)
        ok = np.isfinite(diff)
        d = np.where(ok, diff, 0.0)
        if abs_sum is None:
            abs_sum, sq_sum, bias_sum, count = (np.zeros_like(d) for _ in range(4))
        abs_sum += np.abs(d)
        sq_sum += d**2
        bias_sum += d
        count += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return abs_sum / count, sq_sum / count, bias_sum / count


def binned(err_abs, err_sq, bias, field, bins, sea):
    ## per bin: area share, share of total squared error, mean |err|, systematic share
    ## systematic share = bias^2 / mean squared error: the part a fixed per-cell correction
    ## (learnable from static inputs) could remove; the rest varies from day to day
    rows = []
    total_sq = err_sq[sea].sum()
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = sea & (field >= lo) & (field < hi)
        if not sel.any():
            continue
        rows.append((lo, hi, sel.sum() / sea.sum(), err_sq[sel].sum() / total_sq, err_abs[sel].mean(),
                     (bias[sel] ** 2).sum() / err_sq[sel].sum()))
    return rows


def fmt_bin(lo, hi):
    return f">= {lo:g}" if np.isinf(hi) else f"{lo:g}-{hi:g}"


def print_table(title, unit, rows):
    print(f"\n{title}")
    print(f"  {'bin [' + unit + ']':>14}  {'area':>6}  {'sq.err':>6}  {'mean |err| degC':>15}  {'systematic':>10}")
    for lo, hi, area, sq, mae, sys_share in rows:
        print(f"  {fmt_bin(lo, hi):>14}  {area:6.1%}  {sq:6.1%}  {mae:15.3f}  {sys_share:10.0%}")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(err_abs, sea, panels, var, split, n_days, out_png):
    fig = plt.figure(figsize=(12, 11.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[2.7, 1], hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    shown = np.where(sea, err_abs, np.nan)
    vmax = np.nanpercentile(shown, 99)
    im = ax.imshow(shown, origin="lower", cmap=CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(
        f"Time-mean |target - reprojected 800m|, {var}, {split} ({n_days} days)",
        color=INK, fontsize=12, loc="left",
    )
    ax.set_anchor("W")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="max")
    cb.set_label("degC (colour capped at 99th percentile)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    ## small multiples: one measure, one axis each
    ymax = max(r[4] for _, _, rows in panels for r in rows) * 1.1
    for i, (label, unit, rows) in enumerate(panels):
        a = fig.add_subplot(gs[1, i])
        xs = np.arange(len(rows))
        a.plot(xs, [r[4] for r in rows], color=LINE, linewidth=2, marker="o", markersize=6)
        a.set_xticks(xs, [fmt_bin(r[0], r[1]) for r in rows], rotation=30, ha="right")
        a.set_ylim(0, ymax)
        a.set_title(f"by {label} [{unit}]", color=INK, fontsize=10, loc="left")
        if i == 0:
            a.set_ylabel("mean |err| [degC]", color=MUTED, fontsize=9)
        style(a)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--split", type=str, default="val", help="split name from the config (train/val/...)")
    parser.add_argument("--fold", type=str, default=None,
                        help="CV fold name from split.folds; needed for --split train/val")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default: <output_dir>[/<fold>]/error_map)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.fold is not None:
        apply_fold(cfg, args.fold)
    elif args.split in ("train", "val"):
        parser.error(f"--split {args.split} needs --fold: train/val years come from split.folds")
    root = Path(cfg["data"]["processed_dir"])
    dir_800m, dir_160m, static_dir = root / "800m_on_160m_grid", root / "160m", root / "static"
    var = cfg["data"]["target_vars"][0]
    out = Path(args.out or Path(cfg.get("output_dir", "experiments")) / "error_map")
    out.mkdir(parents=True, exist_ok=True)

    dates = find_dates(dir_800m, dir_160m, var, cfg["split"][args.split]["years"])
    if not dates:
        raise RuntimeError(f"no {var} files for split '{args.split}' in {root}")
    print(f"{var}, split '{args.split}': {len(dates)} days ({dates[0]} -> {dates[-1]})")

    err_abs, err_sq, bias = time_mean_errors(dir_800m, dir_160m, var, dates)

    sea = np.load(static_dir / "land_mask.npy").astype(bool) & np.isfinite(err_abs)
    h = np.load(static_dir / "bathymetry.npy").astype(np.float64)
    dist_km = ndimage.distance_transform_edt(sea) * CELL_KM  # distance to the nearest land cell
    gy, gx = np.gradient(h, CELL_KM * 1000)
    slope = np.hypot(gx, gy)

    print(f"overall: mean |err| {err_abs[sea].mean():.3f} degC, RMSE {np.sqrt(err_sq[sea].mean()):.3f} degC, "
          f"bias {bias[sea].mean():+.3f} degC (target - input)")
    panels = [
        ("distance to coast", "km", binned(err_abs, err_sq, bias, dist_km, DIST_BINS, sea)),
        ("depth", "m", binned(err_abs, err_sq, bias, h, DEPTH_BINS, sea)),
        ("bathymetry slope", "m/m", binned(err_abs, err_sq, bias, slope, SLOPE_BINS, sea)),
    ]
    for label, unit, rows in panels:
        print_table(f"by {label}", unit, rows)
    print("\n  area = share of sea cells in the bin; sq.err = share of the total squared error (what MSE sees)")
    print("  systematic = bias^2 / mean squared error: share a fixed per-cell correction could remove")
    print(f"  systematic share overall: {(bias[sea] ** 2).sum() / err_sq[sea].sum():.0%}")

    ## where is the error concentrated? share of squared error carried by the worst cells
    sq = np.sort(err_sq[sea])[::-1]
    cum = np.cumsum(sq) / sq.sum()
    for frac in (0.05, 0.10, 0.25):
        print(f"  worst {frac:.0%} of sea cells carry {cum[int(frac * sq.size) - 1]:.0%} of the squared error")

    np.savez(out / f"error_maps_{var}_{args.split}.npz", mean_abs=err_abs, mean_sq=err_sq, bias=bias,
             dist_km=dist_km, slope=slope, sea=sea, dates=np.array(dates))
    png = out / f"error_map_{var}_{args.split}.png"
    plot(err_abs, sea, panels, var, args.split, len(dates), png)
    print(f"\nsaved {png}\n      {out / f'error_maps_{var}_{args.split}.npz'}")


if __name__ == "__main__":
    main()
