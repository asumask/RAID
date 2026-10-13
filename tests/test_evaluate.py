## python3 -m pytest tests/test_evaluate.py
## end-to-end smoke test of scripts/evaluate.py on a few fake days; an untrained SimpleSRCNN
## starts exactly at the bicubic baseline, so model and baseline RMSE must match

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from raid.data.file_utils import daily_filename
from raid.config import dict_to_namespace
from raid.models import build_model

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (40, 60)
DATES = ["20260101", "20260102", "20260103"]


def test_evaluate_runs_and_untrained_model_equals_baseline(tmp_path):
    root = tmp_path / "processed"
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    land = np.ones(SHAPE, np.uint8)
    land[:5, :10] = 0
    for d in DATES:
        t = (10 + rng.normal(size=SHAPE)).astype(np.float32)
        t[land == 0] = np.nan
        np.save(root / "160m" / daily_filename("temp", d), t)
        np.save(root / "800m_on_160m_grid" / daily_filename("temp", d),
                t + 0.3 * rng.normal(size=SHAPE).astype(np.float32))
    np.save(root / "static" / "land_mask.npy", land)
    for name in ("bathymetry", "distance_to_coast"):
        np.save(root / "static" / f"{name}.npy", rng.uniform(1, 100, SHAPE).astype(np.float32))

    cfg = yaml.safe_load((ROOT / "configs" / "step1a_cnn.yaml").read_text())
    cfg["data"]["processed_dir"] = str(root)
    cfg["train"]["num_workers"] = 0
    run = tmp_path / "run"
    run.mkdir()
    cfg["output_dir"] = str(run)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    np.savez(run / "norm_stats.npz", temp_mean=10.0, temp_std=1.0)
    torch.save(build_model(dict_to_namespace(cfg)).state_dict(), run / "best.pt")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(tmp_path / "cfg.yaml")],
                   check=True)

    out = run / "eval_test"
    pred = np.load(out / "pred_temp.npy")
    assert pred.shape == (len(DATES), *SHAPE)
    assert np.isnan(pred[:, land == 0]).all() and np.isfinite(pred[:, land == 1]).all()
    assert list(np.load(out / "dates.npy")) == DATES
    daily = np.load(out / "daily_metrics.npy")
    assert daily.shape == (len(DATES), 4)
    np.testing.assert_allclose(daily[:, 0], daily[:, 1], rtol=1e-5)   # model RMSE == baseline RMSE
    for name in ("rmse_map_model_temp", "rmse_map_baseline_temp", "bias_map_model_temp"):
        assert np.load(out / f"{name}.npy").shape == SHAPE
