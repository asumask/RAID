import sys
import numpy as np
import xarray as xr
from scipy import ndimage


GRID_PARAMS = {
    "800m": {"false_easting": 3369600, "false_northing": 1844800},
    "160m": {"false_easting": 3016640, "false_northing": 1438720},
}


REFERENCE_FILE_160M = "/data/osea/s3-osea/Norfjords/A04_v3/avg/2025/norfjords_160m_avg.nc4_2025010112"
REFERENCE_FILE_800M = "/data/osea/s3-osea/Norkyst/avg/2025/norkyst_800m_avg.nc4_2025010112"


def distance_to_coast(ds):
    ## km from each sea cell to the nearest land cell (0 on land); the open domain edge is not coast
    sea = ds["mask_rho"].values == 1
    cell_km = float(ds["xi_rho"][1] - ds["xi_rho"][0]) / 1000
    return ndimage.distance_transform_edt(sea) * cell_km


## static input name -> (how to get it from the 160m reference file, dtype)
STATIC_FIELDS = {
    "bathymetry": (lambda ds: ds["h"].values, np.float32),
    "land_mask": (lambda ds: ds["mask_rho"].values, np.uint8),
    "distance_to_coast": (distance_to_coast, np.float32),
}


def save_static_fields(out_dir, names):
    ## static fields are time-invariant, so take them from the reference grid.
    ## land_mask is always saved: the dataset needs it for the loss mask
    ds = xr.open_dataset(REFERENCE_FILE_160M)
    for name in dict.fromkeys(["land_mask", *names]):
        get, dtype = STATIC_FIELDS[name]
        np.save(out_dir / f"{name}.npy", get(ds).astype(dtype))


def get_true_coords(ds, res):
    p = GRID_PARAMS[res]
    x = ds["xi_rho"].values - p["false_easting"]
    y = ds["eta_rho"].values - p["false_northing"]
    return x, y

def to_rho(ds, da):
    ## average u (v) points onto rho points along xi (eta), like ROMS u2rho/v2rho.
    ## land faces count as 0 (no flow through the coast); mask_u/mask_v are derived from mask_rho
    ## (a face is sea only if both rho cells are), so older files without them work too.
    ## the outermost rho column (row) has one face only and copies it.
    sea = ds["mask_rho"].values == 1
    if set(da.dims) == {"eta_u", "xi_u"}:
        a = np.where(sea[:, :-1] & sea[:, 1:], da.transpose("eta_u", "xi_u").values, 0)
        out = np.empty((a.shape[0], a.shape[1] + 1), dtype=a.dtype)
        out[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
        out[:, 0], out[:, -1] = a[:, 0], a[:, -1]
    elif set(da.dims) == {"eta_v", "xi_v"}:
        a = np.where(sea[:-1] & sea[1:], da.transpose("eta_v", "xi_v").values, 0)
        out = np.empty((a.shape[0] + 1, a.shape[1]), dtype=a.dtype)
        out[1:-1] = 0.5 * (a[:-1] + a[1:])
        out[0], out[-1] = a[0], a[-1]
    else:
        return da
    return xr.DataArray(out, dims=("eta_rho", "xi_rho"),
                        coords={"eta_rho": ds["eta_rho"].values, "xi_rho": ds["xi_rho"].values}, name=da.name)

def extract_variable(ds, res, name, level=-1):
    x, y = get_true_coords(ds, res)
    ds = ds.assign_coords(xi_rho=x, eta_rho=y)
    da = ds[name]
    if "ocean_time" in da.dims:
        da = da.isel(ocean_time=0)
    for vdim in ("s_rho", "s_w"):
        if vdim in da.dims:
            da = da.isel({vdim: level})
    da = to_rho(ds, da)       # u/v/ubar/vbar -> rho points; rho variables pass through
    if set(da.dims) != {"eta_rho", "xi_rho"}:
        raise ValueError(f"{name}: expected a 2D rho-grid field after selection, got dims {da.dims}")
    return da.where(ds["mask_rho"] == 1)

def fill_nearest(da):
    ## replace NaN (land) with the nearest valid value so spline interpolation has finite input
    nan = np.isnan(da.values)
    idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
    return da.copy(data=da.values[tuple(idx)])

def reproject_800m_to_160m(ds_800m, ds_160m, name, level=-1, method="cubic", margin=16):
    x_tgt, y_tgt = get_true_coords(ds_160m, "160m")
    src = extract_variable(ds_800m, "800m", name, level)

    ## crop 800m to the 160m domain + a margin of `margin` cells before filling/interpolating;
    ## much faster than the full domain, and the margin keeps cubic stable at the edges
    m = margin * float(src["xi_rho"][1] - src["xi_rho"][0])
    src = src.sel(
        xi_rho=slice(x_tgt.min() - m, x_tgt.max() + m),
        eta_rho=slice(y_tgt.min() - m, y_tgt.max() + m),
    )
    src = fill_nearest(src)

    out = src.interp(xi_rho=x_tgt, eta_rho=y_tgt, method=method)
    out = out.where(ds_160m["mask_rho"].values == 1)
    return out.values.astype(np.float32)

def set_rho_variables(ds, res):
    ## xi_rho and eta_rho are missing in the older data.
    ## set from the reference data
    if res == "800m":
        ref_ds = xr.open_dataset(REFERENCE_FILE_800M)
    elif res == "160m":
        ref_ds = xr.open_dataset(REFERENCE_FILE_160M)
    else:
        return ds
    ds["xi_rho"] = ref_ds["xi_rho"]
    ds["eta_rho"] = ref_ds["eta_rho"]
    return ds
