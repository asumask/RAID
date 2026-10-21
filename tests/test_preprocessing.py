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


def add_uv(ds, u, v):
    ## staggered u (eta_rho x xi_rho-1) and v (eta_rho-1 x xi_rho); masked faces are NaN like decoded fill values
    sea = ds["mask_rho"].values == 1
    u = np.where(sea[:, :-1] & sea[:, 1:], u, np.nan)
    v = np.where(sea[:-1] & sea[1:], v, np.nan)
    return ds.assign(
        u=(("ocean_time", "s_rho", "eta_u", "xi_u"), np.broadcast_to(u, (1, 3, *u.shape))),
        v=(("ocean_time", "s_rho", "eta_v", "xi_v"), np.broadcast_to(v, (1, 3, *v.shape))),
    )


def test_uv_are_averaged_onto_rho_points():
    ds = make_ds("800m", nx=6, ny=5, dx=800, offset=0, land=(slice(0, 2), slice(0, 2)))
    xi = np.arange(5, dtype=np.float32)
    ds = add_uv(ds, u=np.tile(xi, (5, 1)), v=np.tile(xi[:4, None], (1, 6)))
    sea = ds["mask_rho"].values == 1

    u = extract_variable(ds, "800m", "u").values
    v = extract_variable(ds, "800m", "v").values

    assert u.shape == v.shape == sea.shape
    np.testing.assert_array_equal(np.isnan(u), ~sea)
    np.testing.assert_array_equal(np.isnan(v), ~sea)
    ## interior sea cell: mean of its two faces
    assert u[3, 2] == 0.5 * (1 + 2)
    assert v[3, 2] == 0.5 * (2 + 3)
    ## sea cell next to land: the land face counts as 0
    assert u[0, 2] == 0.5 * (0 + 2)
    assert v[2, 0] == 0.5 * (0 + 2)
    ## outermost column/row copies its only face
    assert u[3, 5] == 4
    assert v[4, 3] == 3


def test_uv_reproject_to_the_160m_grid():
    ds_800m, ds_160m = make_pair()
    ds_800m = add_uv(ds_800m, u=np.ones((50, 59), np.float32), v=np.ones((49, 60), np.float32))
    land = ds_160m["mask_rho"].values == 0

    for name in ("u", "v"):
        arr = reproject_800m_to_160m(ds_800m, ds_160m, name)
        assert arr.shape == land.shape
        np.testing.assert_array_equal(np.isnan(arr), land)
