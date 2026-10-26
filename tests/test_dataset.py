## python -m pytest tests/test_dataset.py

import numpy as np
import pytest
import torch

from raid.data.dataset import denormalize, normalize

## (mean, std) pairs: the current temp stats, plus wider/narrower ranges for other variables
STATS = [(7.2665, 0.9509), (0.0, 1.0), (-1.5, 25.0), (1000.0, 0.01)]


def sample_values(mean, std, shape=(40, 50), seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, 3 * std, size=shape).astype(np.float32)


@pytest.mark.parametrize("mean,std", STATS)
def test_normalize_then_denormalize_recovers_values_numpy(mean, std):
    arr = sample_values(mean, std)

    out = denormalize(normalize(arr, mean, std), mean, std)

    assert out.shape == arr.shape
    np.testing.assert_allclose(out, arr, rtol=1e-6, atol=1e-5 * max(abs(mean), std))


@pytest.mark.parametrize("mean,std", STATS)
def test_normalize_then_denormalize_recovers_values_torch(mean, std):
    arr = torch.from_numpy(sample_values(mean, std))

    out = denormalize(normalize(arr, mean, std), mean, std)

    assert out.dtype == arr.dtype
    torch.testing.assert_close(out, arr, rtol=1e-6, atol=1e-5 * max(abs(mean), std))


def test_normalize_then_denormalize_keeps_nans_in_place():
    mean, std = STATS[0]
    arr = sample_values(mean, std)
    arr[:5, :10] = np.nan

    out = denormalize(normalize(arr, mean, std), mean, std)

    np.testing.assert_array_equal(np.isnan(out), np.isnan(arr))


def test_normalize_gives_zero_mean_unit_std_with_matching_stats():
    arr = sample_values(7.0, 1.0, shape=(200, 300))

    z = normalize(arr, arr.mean(dtype=np.float64), arr.std(dtype=np.float64))

    assert abs(z.mean()) < 1e-4
    assert abs(z.std() - 1) < 1e-4




## ---- patch cropping ----------------------------------------------------------------

from types import SimpleNamespace

from raid.data.dataset import DownscalingDataset
from raid.data.file_utils import daily_filename

GRID = (200, 300)
DATES = ["20240101", "20240102", "20240103"]


@pytest.fixture
def make_train_dataset(tmp_path):
    """Build a DownscalingDataset over a small synthetic processed/ folder.

    Target temp is a steep ramp (10 per cell along rows and columns) with mean 0 / std 1
    stats: crops of x and y offset by even one cell differ by >= 10 on average, while
    aligned crops differ only by the small 800m noise (~0.08).
    """
    rng = np.random.default_rng(0)
    rows, cols = np.indices(GRID)
    ramp = (10 * (rows + cols)).astype(np.float32)

    for sub in ("800m_on_160m_grid", "160m", "static"):
        (tmp_path / sub).mkdir()
    for date in DATES:
        noise = rng.normal(0, 0.1, GRID).astype(np.float32)
        np.save(tmp_path / "800m_on_160m_grid" / daily_filename("temp", date), ramp + noise)
        np.save(tmp_path / "160m" / daily_filename("temp", date), ramp)

    land = np.ones(GRID, dtype=np.uint8)
    land[:20, :30] = 0
    np.save(tmp_path / "static" / "land_mask.npy", land)
    np.save(tmp_path / "static" / "bathymetry.npy", rng.uniform(10, 800, GRID).astype(np.float32))

    cfg = SimpleNamespace(
        data=SimpleNamespace(
            processed_dir=str(tmp_path),
            input_vars=["temp"],
            target_vars=["temp"],
            static_inputs=["bathymetry", "land_mask"],
        )
    )

    def make(patch_size=None, edge_pixels=0):
        return DownscalingDataset(cfg, years=[2024], patch_size=patch_size, edge_pixels=edge_pixels)

    return make


def test_crop_shapes_and_alignment(make_train_dataset):
    ds = make_train_dataset(patch_size=64)
    x, y, mask = ds[0]
    assert x.shape[-2:] == (64, 64)
    assert y.shape[-2:] == (64, 64)
    assert mask.shape[-2:] == (64, 64)
    # channel 0 vs target must still be a small difference, i.e. the crops are aligned
    assert (x[0] - y[0]).abs().mean() < 5.0


## ---- edge trimming -------------------------------------------------------------------

from raid.data.dataset import trim_edges


@pytest.mark.parametrize("as_torch", [False, True])
def test_trim_edges_zeroes_frame_only(as_torch):
    mask = np.ones((10, 12), dtype=bool)
    if as_torch:
        mask = torch.from_numpy(mask)

    out = trim_edges(mask, 2)

    out = np.asarray(out)
    assert not out[:2].any() and not out[-2:].any()
    assert not out[:, :2].any() and not out[:, -2:].any()
    assert out[2:-2, 2:-2].all()
    assert np.asarray(mask).all()       # input left untouched


def test_trim_edges_zero_is_a_noop():
    mask = np.random.default_rng(0).random((8, 9)) > 0.5
    np.testing.assert_array_equal(trim_edges(mask, 0), mask)


def test_edge_pixels_trims_loss_mask_but_not_sea_mask(make_train_dataset):
    plain = make_train_dataset()
    ds = make_train_dataset(edge_pixels=5)

    _, _, mask = ds[0]
    assert torch.equal(ds.sea_mask, plain.sea_mask)
    assert mask[0, :5].sum() == 0 and mask[0, -5:].sum() == 0
    assert mask[0, :, :5].sum() == 0 and mask[0, :, -5:].sum() == 0
    torch.testing.assert_close(mask[0, 5:-5, 5:-5], plain.sea_mask[5:-5, 5:-5].float())
