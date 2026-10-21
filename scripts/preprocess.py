import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import xarray as xr

from raid.data.preprocessing import extract_variable, reproject_800m_to_160m, save_static_fields, compute_stats, set_rho_variables
from raid.data.file_utils import list_files, get_paired_dates, _extract_date, daily_filename

## create source files with same coordinates
## CUDA_VISIBLE_DEVICES=0
## python3 scripts/preprocess.py --config /home/a22740/asumak/gitlab/raid/configs/base.yaml
## --method sets the 800m -> 160m interpolation (default cubic), e.g. for the old bilinear data:
## python3 scripts/preprocess.py --config configs/base.yaml --method linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="cubic", choices=["linear", "cubic"],
                        help="interpolation of the 800m input onto the 160m grid (xarray interp)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    files_800m = list_files(cfg["data"]["raw_800m"], years=[2023, 2024, 2025, 2026])
    files_160m = list_files(cfg["data"]["raw_160m"], years=[2023, 2024, 2025, 2026])
    dates = get_paired_dates(files_800m, files_160m)
    input_vars = cfg["data"]["input_vars"]
    target_vars = cfg["data"]["target_vars"]
    levels = cfg["data"].get("levels", {})

    root = Path(cfg["data"]["processed_dir"])
    out_800m = root / "800m_on_160m_grid"
    out_160m = root / "160m"
    out_800m.mkdir(parents=True, exist_ok=True)
    out_160m.mkdir(parents=True, exist_ok=True)

    ## get the grid from a fixed 2025 reference file (loaded once, cached)
    ## -------------------------------------
    
    for date in dates:
        path_800m = next(f for f in files_800m if _extract_date(f) == date)
        path_160m = next(f for f in files_160m if _extract_date(f) == date)

        ds_800m = xr.open_dataset(path_800m)
        ds_160m = xr.open_dataset(path_160m)
        ## xi_rho and eta_rho are missing in the older data.
        ds_800m= set_rho_variables(ds_800m, "800m")
        ds_160m= set_rho_variables(ds_160m, "160m")

        for name in input_vars:
            arr = reproject_800m_to_160m(ds_800m, ds_160m, name, levels.get(name, -1), method=args.method)
            np.save(out_800m / daily_filename(name, date), arr)

        for name in target_vars:
            arr = extract_variable(ds_160m, "160m", name, levels.get(name, -1))
            np.save(out_160m / daily_filename(name, date), arr.values.astype(np.float32))
    
    
    out_static = root / "static"
    out_static.mkdir(parents=True, exist_ok=True)
    save_static_fields(out_static, cfg["data"]["static_inputs"])

    ## stats from the 160m targets; input-only variables fall back to the reprojected 800m files
    train_years = cfg["split"]["train"]["years"]
    input_only = [v for v in input_vars if v not in target_vars]
    stats = compute_stats(out_160m, target_vars, train_years)
    stats.update(compute_stats(out_800m, input_only, train_years))
    np.savez(out_static / "preprocessing_stats.npz", **stats)
    


if __name__ == "__main__":
    main()
