import json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NORM_FILE = "/media/airlab/ROCSTOR/ai_land_model/normalization.json"
RUNS_DIR = "/media/airlab/ROCSTOR/ai_land_model/runs_nc"
PRED_DIR = "/media/airlab/ROCSTOR/ai_land_model/pred"
FIG_DIR = "/media/airlab/ROCSTOR/ai_land_model/paper/figures"
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

SOIL_VARS = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
]

SM_VARS = SOIL_VARS[:4]
ST_VARS = SOIL_VARS[4:]

SM_LABELS = [
    "Layer 1 (0-7 cm)",
    "Layer 2 (7-28 cm)",
    "Layer 3 (28-100 cm)",
    "Layer 4 (100-289 cm)",
]

CASE_STUDIES = {
    "2019_mississippi": {
        "label": "2019 Mississippi Flooding",
        "primary": "earthmind_ai_land_2019031500.nc",
        "inits": [
            "earthmind_ai_land_2019030100.nc",
            "earthmind_ai_land_2019031500.nc",
            "earthmind_ai_land_2019040100.nc",
            "earthmind_ai_land_2019050100.nc",
        ],
    },
    "2012_drought": {
        "label": "2012 Central US Drought",
        "primary": "earthmind_ai_land_2012051500.nc",
        "inits": [
            "earthmind_ai_land_2012051500.nc",
            "earthmind_ai_land_2012060100.nc",
            "earthmind_ai_land_2012061500.nc",
            "earthmind_ai_land_2012070100.nc",
        ],
    },
    "2021_heatdome": {
        "label": "2021 Pacific NW Heat Dome",
        "primary": "earthmind_ai_land_2021061000.nc",
        "inits": [
            "earthmind_ai_land_2021061000.nc",
            "earthmind_ai_land_2021061700.nc",
            "earthmind_ai_land_2021062400.nc",
            "earthmind_ai_land_2021062700.nc",
        ],
    },
    "2011_texas": {
        "label": "2011 Texas Drought",
        "primary": "earthmind_ai_land_2011021500.nc",
        "inits": [
            "earthmind_ai_land_2011021500.nc",
            "earthmind_ai_land_2011030100.nc",
            "earthmind_ai_land_2011040100.nc",
            "earthmind_ai_land_2011060100.nc",
        ],
    },
}

CASE_COLORS = {
    "2019_mississippi": "#0173b2",    # Blue
    "2012_drought": "#029e73",         # Green
    "2021_heatdome": "#de8f05",        # Orange
    "2011_texas": "#cc78bc",           # Purple
}


def set_pub_style():
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def load_norm():
    with open(NORM_FILE) as f:
        return json.load(f)


def denormalize(da, var, norm_stats):
    vmin = norm_stats[var]["min"]
    vmax = norm_stats[var]["max"]
    return da * (vmax - vmin) + vmin


def load_era5_daily(t0, t1, variables=None):
    import pandas as pd
    if variables is None:
        variables = SOIL_VARS
    ds = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})
    ds = ds[variables]
    ds_slice = ds.sel(time=slice(str(t0), str(t1)))
    ds_daily = ds_slice.groupby("time.date").mean("time").rename({"date": "time"})
    ds_daily = ds_daily.assign_coords(
        time=pd.to_datetime(ds_daily.time.values).values
    )
    return ds_daily


def load_forecast(fname):
    import os
    path = os.path.join(RUNS_DIR, fname)
    return xr.open_dataset(path)


def compute_rmse(ds_pred, ds_obs, var):
    diff = ds_pred[var] - ds_obs[var]
    return np.sqrt((diff ** 2).mean(dim=("latitude", "longitude")))


def compute_acc(f, o):
    f_anom = f - f.mean(dim=("latitude", "longitude"))
    o_anom = o - o.mean(dim=("latitude", "longitude"))
    num = (f_anom * o_anom).mean(dim=("latitude", "longitude"))
    den = (
        np.sqrt((f_anom ** 2).mean(dim=("latitude", "longitude")))
        * np.sqrt((o_anom ** 2).mean(dim=("latitude", "longitude")))
    )
    return num / den


def align_times(ds_pred, ds_era):
    import pandas as pd
    common = np.intersect1d(
        pd.to_datetime(ds_pred.time.values).values.astype("datetime64[ns]"),
        pd.to_datetime(ds_era.time.values).values.astype("datetime64[ns]"),
    )
    return ds_pred.sel(time=common), ds_era.sel(time=common)
