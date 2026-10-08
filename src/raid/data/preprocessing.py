import sys
import numpy as np
import xarray as xr
from scipy import ndimage

from .file_utils import daily_filename

GRID_PARAMS = {
    "800m": {"false_easting": 3369600, "false_northing": 1844800},
    "160m": {"false_easting": 3016640, "false_northing": 1438720},
}


REFERENCE_FILE_160M = "/data/osea/s3-osea/Norfjords/A04_v3/avg/2025/norfjords_160m_avg.nc4_2025010112"
REFERENCE_FILE_800M = "/data/osea/s3-osea/Norkyst/avg/2025/norkyst_800m_avg.nc4_2025010112"


## static input name -> variable in the 160m reference file
STATIC_FIELDS = {
    "bathymetry": ("h", np.float32),
    "land_mask": ("mask_rho", np.uint8),
}


def save_static_fields(out_dir, names):
    ## bathymetry and land mask are time-invariant, so take them from the reference grid
    ds = xr.open_dataset(REFERENCE_FILE_160M)
    for name in names:
        var, dtype = STATIC_FIELDS[name]
        np.save(out_dir / f"{name}.npy", ds[var].values.astype(dtype))


def compute_stats(data_dir, names, years):
    ## normalisation stats per variable over sea cells of the training days only
    ## accumulated day by day in float64 to avoid loading every day into memory
    sea = xr.open_dataset(REFERENCE_FILE_160M)["mask_rho"].values == 1
    stats = {}
    for name in names:
        s = ss = n = 0.0
        for year in years:
            for path in sorted(data_dir.glob(daily_filename(name, f"{year}[0-9][0-9][0-9][0-9]"))):
                v = np.load(path)[sea].astype(np.float64)
                v = v[~np.isnan(v)]
                s += v.sum()
                ss += (v**2).sum()
                n += v.size
        if n == 0:
            raise RuntimeError(f"no training files for {name} in {data_dir} for years {years}")
        mean = s / n
        stats[f"{name}_mean"] = mean
        stats[f"{name}_std"] = np.sqrt(ss / n - mean**2)
    return stats

def get_true_coords(ds, res):
    p = GRID_PARAMS[res]
    x = ds["xi_rho"].values - p["false_easting"]
    y = ds["eta_rho"].values - p["false_northing"]
    return x, y

def extract_variable(ds, res, name, level=-1):
    x, y = get_true_coords(ds, res)
    ds = ds.assign_coords(xi_rho=x, eta_rho=y)
    da = ds[name]
    if "ocean_time" in da.dims:
        da = da.isel(ocean_time=0)
    for vdim in ("s_rho", "s_w"):
        if vdim in da.dims:
            da = da.isel({vdim: level})
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
