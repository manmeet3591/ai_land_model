import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import json

# --- Settings ---
zarr_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
var_name = 'volumetric_soil_water_layer_1'
time_index = 86400*10

# --- Load dataset ---
ds = xr.open_zarr(zarr_path, consolidated=True, storage_options={"token": "anon"})
v = ds[var_name].isel(time=time_index).compute()

# --- Ensure latitude increasing ---
if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
    v = v.sortby('latitude')

# --- Plot and save original with NaNs ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(v.values, origin='lower', cmap='viridis')
plt.title('Original (with NaNs)')
plt.colorbar()
plt.savefig("original.png")
print("✅ Saved: original.png")

# --- Fill NaNs ---
v_interp = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
v_interp = v_interp.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")

# --- Plot and save interpolated result ---
plt.subplot(1, 2, 2)
plt.imshow(v_interp.values, origin='lower', cmap='viridis')
plt.title('Interpolated (filled)')
plt.colorbar()
plt.tight_layout()
plt.savefig("interpolated.png")
print("✅ Saved: interpolated.png")

# --- Check NaNs ---
print(f"Original NaNs: {np.isnan(v.values).sum()}")
print(f"After Interpolation NaNs: {np.isnan(v_interp.values).sum()}")
