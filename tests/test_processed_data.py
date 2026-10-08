## python -m pytest tests/test_processed_data.py
##

from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
PROCESSED = ROOT / CFG["data"]["processed_dir"]
DIR_800M = PROCESSED / "800m_on_160m_grid"
DIR_160M = PROCESSED / "160m"
DIR_STATIC = PROCESSED / "static"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (DIR_800M.is_dir() and DIR_160M.is_dir() and DIR_STATIC.is_dir()),
        reason=f"processed data not found under {PROCESSED}; run scripts/preprocess.py first",
    ),
]

FILES_800M = sorted(DIR_800M.glob("*.npy")) if DIR_800M.is_dir() else []


@pytest.fixture(scope="module")
def land_mask():
    return np.load(DIR_STATIC / "land_mask.npy")


@pytest.fixture(scope="module")
def bathymetry():
    return np.load(DIR_STATIC / "bathymetry.npy")


def test_mask_and_bathymetry_match(land_mask, bathymetry):
    assert land_mask.shape == bathymetry.shape
    assert set(np.unique(land_mask)) <= {0, 1}
    assert np.isfinite(bathymetry).all()
    ## ROMS keeps h >= hmin > 0 on every cell, land included
    assert (bathymetry > 0).all()


def test_every_800m_file_has_a_160m_counterpart():
    missing = [f.name for f in FILES_800M if not (DIR_160M / f.name).exists()]
    assert not missing, f"{len(missing)} reprojected 800m files without a 160m target, e.g. {missing[:5]}"


@pytest.mark.parametrize("path_800m", FILES_800M, ids=lambda p: p.stem)
def test_800m_and_160m_arrays_match_grid(path_800m, land_mask, bathymetry):
    path_160m = DIR_160M / path_800m.name
    if not path_160m.exists():
        pytest.skip(f"no 160m counterpart for {path_800m.name}")
    arr_800m = np.load(path_800m)
    arr_160m = np.load(path_160m)
    land = land_mask == 0

    assert arr_800m.shape == arr_160m.shape == land_mask.shape == bathymetry.shape
    np.testing.assert_array_equal(np.isnan(arr_800m), land, err_msg="800m NaNs differ from the land mask")
    np.testing.assert_array_equal(np.isnan(arr_160m), land, err_msg="160m NaNs differ from the land mask")
