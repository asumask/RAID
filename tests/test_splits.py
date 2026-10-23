## python -m pytest tests/test_splits.py

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from raid.data.dataset import DownscalingDataset
from raid.data.file_utils import daily_filename
from raid.config import apply_fold, dict_to_namespace

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("step1*.yaml"))     # training configs; base.yaml is preprocessing only


def load_cfg(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_split_years_in_config_dont_overlap(config_path):
    splits = {name: set(s["years"]) for name, s in load_cfg(config_path)["split"].items()
              if name not in ("folds", "full")}

    for name, years in splits.items():
        assert years, f"split '{name}' has no years"
    for (a, ya), (b, yb) in combinations(splits.items(), 2):
        assert not ya & yb, f"splits '{a}' and '{b}' share years {sorted(ya & yb)}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_cv_folds_dont_overlap_or_touch_test_years(config_path):
    split = load_cfg(config_path)["split"]
    test_years = set(split.get("test", {}).get("years", []))

    for name, fold in split.get("folds", {}).items():
        train, val = set(fold["train"]), set(fold["val"])
        assert train and val, f"fold '{name}' has an empty train or val"
        assert not train & val, f"fold '{name}' has years {sorted(train & val)} in both train and val"
        assert not (train | val) & test_years, f"fold '{name}' uses test years {sorted((train | val) & test_years)}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_full_model_doesnt_touch_test_years(config_path):
    split = load_cfg(config_path)["split"]
    test_years = set(split.get("test", {}).get("years", []))
    train = set(split["full"]["train"])
    assert train, "split.full has no train years"
    assert not train & test_years, f"split.full uses test years {sorted(train & test_years)}"


def write_processed_dir(root, dates, var="temp", shape=(4, 5)):
    ## minimal processed layout read by DownscalingDataset, one file per day and resolution
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    for date in dates:
        np.save(root / "800m_on_160m_grid" / daily_filename(var, date), np.zeros(shape, np.float32))
        np.save(root / "160m" / daily_filename(var, date), np.zeros(shape, np.float32))
    np.save(root / "static" / "land_mask.npy", np.ones(shape, np.uint8))
    np.save(root / "static" / "bathymetry.npy", np.arange(1, 21, dtype=np.float32).reshape(shape))


@pytest.mark.parametrize("fold", load_cfg(ROOT / "configs" / "step1a_cnn.yaml")["split"]["folds"])
def test_datasets_for_each_split_have_disjoint_days(tmp_path, fold):
    raw = apply_fold(load_cfg(ROOT / "configs" / "step1a_cnn.yaml"), fold)
    splits = {name: s for name, s in raw["split"].items() if name not in ("folds", "full")}
    all_years = sorted({y for s in splits.values() for y in s["years"]})
    ## include year boundaries, where an off-by-one in year filtering would leak days
    dates = [f"{y}{md}" for y in all_years for md in ("0101", "0615", "1231")]
    write_processed_dir(tmp_path, dates, var=raw["data"]["input_vars"][0])

    raw["data"]["processed_dir"] = str(tmp_path)
    raw["data"]["static_inputs"] = ["bathymetry", "land_mask"]
    cfg = dict_to_namespace(raw)

    days = {}
    for name, split in splits.items():
        days[name] = set(DownscalingDataset(cfg, split["years"]).dates)
        wrong = {d for d in days[name] if int(d[:4]) not in split["years"]}
        assert not wrong, f"split '{name}' picked days outside its years: {sorted(wrong)}"

    for (a, da), (b, db) in combinations(days.items(), 2):
        assert not da & db, f"splits '{a}' and '{b}' share days {sorted(da & db)}"
