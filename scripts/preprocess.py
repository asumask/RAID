import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import xarray as xr

from raid.data.preprocessing import extract_variable, reproject_800m_to_160m, save_static_fields, set_rho_variables
from raid.data.file_utils import list_files, get_paired_dates, _extract_date, daily_filename

## create source files with same coordinates
## CUDA_VISIBLE_DEVICES=0
## python3 scripts/preprocess.py --config /home/a22740/asumak/gitlab/raid/configs/base.yaml
## --method sets the 800m -> 160m interpolation (default cubic), e.g. for the old bilinear data:
## python3 scripts/preprocess.py --config configs/base.yaml --method linear
## by default existing outputs are skipped; add --overwrite to reprocess everything:
## python3 scripts/preprocess.py --config configs/base.yaml --overwrite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="cubic", choices=["linear", "cubic"],
                        help="interpolation of the 800m input onto the 160m grid (xarray interp)")
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess all dates; by default only dates with missing output files are processed")
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
        todo_in = [n for n in input_vars
                   if args.overwrite or not (out_800m / daily_filename(n, date)).exists()]
        todo_tgt = [n for n in target_vars
                    if args.overwrite or not (out_160m / daily_filename(n, date)).exists()]
        if not todo_in and not todo_tgt:
            continue

        path_800m = next(f for f in files_800m if _extract_date(f) == date)
        path_160m = next(f for f in files_160m if _extract_date(f) == date)

        ds_800m = xr.open_dataset(path_800m)
        ds_160m = xr.open_dataset(path_160m)
        ## xi_rho and eta_rho are missing in the older data.
        ds_800m= set_rho_variables(ds_800m, "800m")
        ds_160m= set_rho_variables(ds_160m, "160m")

        for name in todo_in:
            arr = reproject_800m_to_160m(ds_800m, ds_160m, name, levels.get(name, -1), method=args.method)
            np.save(out_800m / daily_filename(name, date), arr)

        for name in todo_tgt:
            arr = extract_variable(ds_160m, "160m", name, levels.get(name, -1))
            np.save(out_160m / daily_filename(name, date), arr.values.astype(np.float32))
    
    
    out_static = root / "static"
    out_static.mkdir(parents=True, exist_ok=True)
    save_static_fields(out_static, cfg["data"]["static_inputs"])


if __name__ == "__main__":
    main()
