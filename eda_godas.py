import xarray as xr

path = '/scratch/08105/ms86336/godas_pentad/'

ds = xr.open_dataset(path+'godas.P.20240209.nc')

print(ds)