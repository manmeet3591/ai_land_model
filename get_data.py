import xarray as xr

ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    consolidated=True,
    storage_options={"token": "anon"}
)

print("Available variables:")
print(list(ds.data_vars))

forcing_vars = [
    '2m_temperature',
    'total_precipitation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards',
    'evaporation',
    'skin_temperature',
    'snowfall',
    'mean_sea_level_pressure'
]

soil_vars = [
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'soil_temperature_level_4'
]

all_vars = forcing_vars + soil_vars
existing_vars = [v for v in all_vars if v in ds.data_vars]
missing_vars = [v for v in all_vars if v not in ds.data_vars]

print("✅ Found:")
print(existing_vars)
print("❌ Missing:")
print(missing_vars)

ds_subset = ds[existing_vars]
norm_stats = {
    var: {
        "min": float(ds_subset[var].min().compute()),
        "max": float(ds_subset[var].max().compute())
    }
    for var in ds_subset.data_vars
}
import json
with open("normalization.json", "w") as f:
    json.dump(norm_stats, f, indent=2)
