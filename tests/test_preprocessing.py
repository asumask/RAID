## python -m pytest tests/test_preprocessing.py
##

import numpy as np
import pytest
import xarray as xr

from raid.data.preprocessing import GRID_PARAMS, extract_variable, fill_nearest, reproject_800m_to_160m


def make_ds(res, nx, ny, dx, offset, land):
    ## synthetic ROMS-like file: projected coords include the false easting/northing like the real data
    p = GRID_PARAMS[res]
    x = p["false_easting"] + offset + np.arange(nx) * dx
    y = p["false_northing"] + offset + np.arange(ny) * dx
    xx, yy = np.meshgrid(x - p["false_easting"], y - p["false_northing"])
    temp = (5 + 1e-4 * xx + 2e-4 * yy).astype(np.float32)
    temp = np.broadcast_to(temp, (1, 3, ny, nx))
    mask = np.ones((ny, nx), dtype=np.float64)
    mask[land] = 0
    return xr.Dataset(
        {
            "temp": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), temp),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
        },
        coords={"xi_rho": x, "eta_rho": y},
    )


def make_pair():
    ## 160m domain sits inside the 800m domain, with land in different places on each grid
    ds_800m = make_ds("800m", nx=60, ny=50, dx=800, offset=0, land=(slice(0, 5), slice(0, 10)))
    ds_160m = make_ds("160m", nx=70, ny=40, dx=160, offset=12000, land=(slice(10, 20), slice(30, 50)))
    return ds_800m, ds_160m


def test_800m_and_160m_arrays_have_identical_shape():
    ds_800m, ds_160m = make_pair()
    arr_800m = reproject_800m_to_160m(ds_800m, ds_160m, "temp")
    arr_160m = extract_variable(ds_160m, "160m", "temp").values

    assert arr_800m.shape == arr_160m.shape == ds_160m["mask_rho"].shape


def test_800m_and_160m_arrays_share_the_160m_land_mask():
    ds_800m, ds_160m = make_pair()
    arr_800m = reproject_800m_to_160m(ds_800m, ds_160m, "temp")
    arr_160m = extract_variable(ds_160m, "160m", "temp").values
    land = ds_160m["mask_rho"].values == 0

    np.testing.assert_array_equal(np.isnan(arr_800m), land)
    np.testing.assert_array_equal(np.isnan(arr_160m), land)


def only_one_sea_cell(ny=50, nx=60):
    land = np.ones((ny, nx), dtype=bool)
    land[ny // 2, nx // 2] = False
    return land


## land layouts on a 50x60 grid; every layout leaves at least one sea cell
LAND_LAYOUTS = {
    "corner": (slice(0, 5), slice(0, 10)),
    "full_edge": (slice(None), slice(0, 20)),
    "interior_island": (slice(20, 30), slice(25, 40)),
    "single_sea_cell": only_one_sea_cell(),
}


@pytest.mark.parametrize("land", LAND_LAYOUTS.values(), ids=LAND_LAYOUTS.keys())
def test_no_nans_left_after_filling(land):
    ds = make_ds("800m", nx=60, ny=50, dx=800, offset=0, land=land)
    src = extract_variable(ds, "800m", "temp")
    assert np.isnan(src.values).any()

    filled = fill_nearest(src)

    assert not np.isnan(filled.values).any()
    sea = ~np.isnan(src.values)
    np.testing.assert_array_equal(filled.values[sea], src.values[sea])
