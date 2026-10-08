## python -m pytest tests/test_splits.py

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from raid.data.dataset import DownscalingDataset
from raid.trainer import dict_to_namespace

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def load_cfg(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_split_years_in_config_dont_overlap(config_path):
    splits = {name: set(s["years"]) for name, s in load_cfg(config_path)["split"].items()}

    for name, years in splits.items():
        assert years, f"split '{name}' has no years"
    for (a, ya), (b, yb) in combinations(splits.items(), 2):
        assert not ya & yb, f"splits '{a}' and '{b}' share years {sorted(ya & yb)}"


def write_processed_dir(root, dates, var="temp", shape=(4, 5)):
    ## minimal processed layout read by DownscalingDataset, one file per day and resolution
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    for date in dates:
        np.save(root / "800m_on_160m_grid" / f"{var}_{date}.npy", np.zeros(shape, np.float32))
        np.save(root / "160m" / f"{var}_{date}.npy", np.zeros(shape, np.float32))
    np.save(root / "static" / "land_mask.npy", np.ones(shape, np.uint8))
    np.save(root / "static" / "bathymetry.npy", np.arange(1, 21, dtype=np.float32).reshape(shape))
    np.savez(root / "static" / "preprocessing_stats.npz", **{f"{var}_mean": 0.0, f"{var}_std": 1.0})


def test_datasets_for_each_split_have_disjoint_days(tmp_path):
    raw = load_cfg(ROOT / "configs" / "base.yaml")
    all_years = sorted({y for s in raw["split"].values() for y in s["years"]})
    ## include year boundaries, where an off-by-one in year filtering would leak days
    dates = [f"{y}{md}" for y in all_years for md in ("0101", "0615", "1231")]
    write_processed_dir(tmp_path, dates, var=raw["data"]["input_vars"][0])

    raw["data"]["processed_dir"] = str(tmp_path)
    raw["data"]["static_inputs"] = ["bathymetry", "land_mask"]
    cfg = dict_to_namespace(raw)

    days = {}
    for name, split in raw["split"].items():
        days[name] = set(DownscalingDataset(cfg, split["years"]).dates)
        wrong = {d for d in days[name] if int(d[:4]) not in split["years"]}
        assert not wrong, f"split '{name}' picked days outside its years: {sorted(wrong)}"

    for (a, da), (b, db) in combinations(days.items(), 2):
        assert not da & db, f"splits '{a}' and '{b}' share days {sorted(da & db)}"
