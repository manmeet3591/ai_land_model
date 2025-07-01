import xarray as xr
import torch
import json

# --- Inputs ---
zarr_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
norm_file = 'normalization.json'
time_index = 0  # input at t, target will be t+1

# --- Variable groups ---
soil_vars = [
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    'soil_temperature_level_1', 'soil_temperature_level_2',
    'soil_temperature_level_3', 'soil_temperature_level_4'
]

# --- Load ---
ds = xr.open_zarr(zarr_path, consolidated=True, storage_options={"token": "anon"})

with open(norm_file, 'r') as f:
    norm_stats = json.load(f)

vars_to_use = list(norm_stats.keys())  # all normalized vars from file

# --- Input: vars at t ---
input_vars = []
for var in vars_to_use:
    if var not in ds:
        print(f"⚠️ Missing: {var}")
        continue
    v = ds[var].isel(time=time_index).compute()
    v_min = norm_stats[var]["min"]
    v_max = norm_stats[var]["max"]
    v_norm = (v - v_min) / (v_max - v_min)
    input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

input_tensor = torch.stack(input_vars)  # shape [C, lat, lon]

# --- Target: only soil vars at t+1 ---
target_vars = []
for var in soil_vars:
    if var not in ds:
        print(f"⚠️ Missing target: {var}")
        continue
    v = ds[var].isel(time=time_index + 1).compute()
    v_min = norm_stats[var]["min"]
    v_max = norm_stats[var]["max"]
    v_norm = (v - v_min) / (v_max - v_min)
    target_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

target_tensor = torch.stack(target_vars)  # shape [8, lat, lon]

# --- Done ---
print("✅ Input shape :", input_tensor.shape)
print("✅ Target shape:", target_tensor.shape)
