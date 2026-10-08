import glob
from pathlib import Path


def list_files(cfg, years):
    files = []
    for year in years:
        pattern = f"{cfg["path"]}/{year}/{cfg["location_name"]}_{cfg["resolution_tag"]}_avg.nc4_*"
        files.extend(sorted(glob.glob(pattern)))
    return files


def _extract_date(filepath):
    return Path(filepath).name.split("_")[-1][:8]


def daily_filename(name, date):
    ## single source of truth for preprocessed file names: YYYYMMDD_<var>.npy
    return f"{date}_{name}.npy"


def get_paired_dates(files_800m, files_160m):
    dates_800m = {_extract_date(f) for f in files_800m}
    dates_160m = {_extract_date(f) for f in files_160m}
    return sorted(dates_800m & dates_160m)
