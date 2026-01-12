import argparse
import xarray as xr
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import earth2grid
import numpy as np
import os
from tqdm import tqdm

# ===================== ARGS =====================
parser = argparse.ArgumentParser()
parser.add_argument("--init_time", type=str, required=True,
                    help="e.g. 2019-01-01T00")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--weights", type=str, required=True,
                    help="Path to .pth weights saved from training")
parser.add_argument("--n_steps", type=int, default=10)
parser.add_argument("--out", type=str, default="soil_forecast.nc")
args = parser.parse_args()

# ===================== MODEL (MATCH TRAINING) =====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(up_channels, up_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size,
                                   padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True,
                 use_ds_conv=False, out_activation=None):
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)

        factor = 2 if trilinear else 1

        self.up1 = Up(self.channels[3], self.channels[2], self.channels[2] // factor, trilinear)
        self.up2 = Up(self.channels[2] // factor, self.channels[1], self.channels[1] // factor, trilinear)
        self.up3 = Up(self.channels[1] // factor, self.channels[0], self.channels[0], trilinear)

        self.outc = OutConv(self.channels[0], n_classes, activation=out_activation)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# ===================== SETTINGS =====================
zarr_path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
norm_file = "normalization.json"

soil_vars = [
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    "soil_temperature_level_1", "soil_temperature_level_2",
    "soil_temperature_level_3", "soil_temperature_level_4"
]

level = 6
nside = 2 ** level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===================== LOAD NORM =====================
with open(norm_file, "r") as f:
    norm_stats = json.load(f)
vars_to_use = list(norm_stats.keys())

# ===================== LOAD DATA =====================
ds = xr.open_zarr(zarr_path, consolidated=True, storage_options={"token": "anon"})
ds_year = ds.sel(time=str(args.year))
ds_daily = ds_year.groupby("time.dayofyear").mean("time")

# init day index from init_time
init_dt = np.datetime64(args.init_time)
year0 = np.datetime64(f"{args.year}-01-01T00")
init_day = int((init_dt - year0) / np.timedelta64(1, "D"))
print("init_day:", init_day)

# build regridders
sample_var = ds_daily[vars_to_use[0]].isel(dayofyear=0)
if "latitude" in sample_var.dims and not sample_var.latitude.values[0] < sample_var.latitude.values[-1]:
    sample_var = sample_var.sortby("latitude")

lat = sample_var.latitude.values
lon = sample_var.longitude.values

nlat, nlon = len(lat), len(lon)
src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
hpx_grid = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
regridder = earth2grid.get_regridder(src_grid, hpx_grid)
regridder_back = earth2grid.get_regridder(hpx_grid, src_grid)

# ===================== LOAD MODEL =====================
model = UNet(n_channels=len(vars_to_use), n_classes=len(soil_vars), out_activation=None).to(device)
state = torch.load(args.weights, map_location=device)
# support either raw state_dict or checkpoint dict
if isinstance(state, dict) and ("model_state_dict" in state or "state_dict" in state):
    state = state.get("model_state_dict", state.get("state_dict"))
model.load_state_dict(state)
model.eval()
print("Loaded weights:", args.weights)

# ===================== FORECAST LOOP =====================
all_fcst = []
fcst_times = []

forecast_np = None  # last predicted soil vars (normalized)

for step in tqdm(range(1, args.n_steps + 1)):
    day = init_day + (step - 1)

    input_vars = []
    for var in vars_to_use:
        if var not in ds_daily:
            continue

        v = ds_daily[var].isel(dayofyear=day)

        if "latitude" in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
            v = v.sortby("latitude")

        v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")

        v_norm = (v - norm_stats[var]["min"]) / (norm_stats[var]["max"] - norm_stats[var]["min"])
        input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

    x_tensor = torch.stack(input_vars).float()  # [C, lat, lon]

    input_healpix = torch.stack([
        regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
    ])  # [C, 12, nside, nside]

    # insert previous soil forecast into soil channels (autoregressive)
    if forecast_np is not None:
        input_healpix[-len(soil_vars):] = torch.tensor(forecast_np, dtype=torch.float32)

    X = input_healpix.unsqueeze(0).to(device)  # [1, C, 12, nside, nside]

    with torch.no_grad():
        y = model(X).squeeze(0).cpu().numpy()  # [8, 12, nside, nside] normalized

    forecast_np = y

    # regrid each soil var back to lat/lon, then flipud
    out_vars = []
    for i in range(len(soil_vars)):
        flat = y[i].reshape(-1)
        ll = regridder_back(torch.from_numpy(flat).double()).numpy()
        ll = np.flipud(ll)  # ✅ flipud fix
        out_vars.append(ll)

    all_fcst.append(np.stack(out_vars, axis=0))  # [8, lat, lon]
    fcst_times.append(init_dt + np.timedelta64(step, "D"))

# stack to [time, var, lat, lon]
all_fcst = np.stack(all_fcst, axis=0)

# ===================== WRITE SINGLE NETCDF =====================
# latitude was flipped -> coordinate should also be reversed
lat_out = lat[::-1]

ds_out = xr.Dataset(
    {
        soil_vars[i]: (("time", "latitude", "longitude"), all_fcst[:, i, :, :])
        for i in range(len(soil_vars))
    },
    coords={
        "time": np.array(fcst_times, dtype="datetime64[ns]"),
        "prediction_timedelta": ("time", np.arange(1, args.n_steps + 1, dtype=np.int32)),
        "latitude": lat_out,
        "longitude": lon,
    },
)

ds_out.to_netcdf(args.out)
print(f"✅ Wrote: {args.out}")
print(ds_out)
