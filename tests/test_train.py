## python3 -m pytest tests/test_train.py
## end-to-end smoke test of scripts/train.py --fold full on a few fake days

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from test_evaluate import make_fake_setup

ROOT = Path(__file__).resolve().parents[1]


def test_train_full_uses_all_non_test_years_and_saves_last_epoch(tmp_path):
    ## a day in every year, so a wrong year selection would show in norm_stats / the log
    dates = ["20231201", "20240301", "20250301", "20260301"]
    cfg_path, _ = make_fake_setup(tmp_path, dates, tmp_path / "unused")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["train"].update(epochs=2, batch_size=2, patch_size=None)
    cfg_path.write_text(yaml.safe_dump(cfg))

    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
                    "--fold", "full"], check=True)

    run = tmp_path / "run" / "full"
    assert (run / "best.pt").exists()
    log = (run / "train.log").read_text()
    assert "train [2023, 2024, 2025]  val []" in log
    assert "epoch 1: train" in log and " val " not in log.split("epoch 0:")[1]

    ## stats from the three training days only, not the 2026 test day
    stats = np.load(run / "norm_stats.npz")
    root = Path(cfg["data"]["processed_dir"]) / "160m"
    sea = np.load(Path(cfg["data"]["processed_dir"]) / "static" / "land_mask.npy").astype(bool)
    train = np.concatenate([np.load(root / f"{d}_temp.npy")[sea] for d in dates[:3]])
    np.testing.assert_allclose(stats["temp_mean"], train.mean(), rtol=1e-6)
