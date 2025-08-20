import numpy as np
import xarray as xr
import torch
import earth2grid
from earth2grid import healpix, latlon

# Define the grid size
nlat, nlon = 33, 64

# Generate a lat-lon grid (for test purposes)
lat = np.linspace(-90, 90, nlat)
lon = np.linspace(0, 360, nlon)
field = np.cos(np.deg2rad(lat[:, None])) ** 2 + 0.5 * np.sin(np.deg2rad(lon))

# Create the Xarray dataset
ds = xr.Dataset(
    {
        "field": (["lat", "lon"], field)
    },
    coords={
        "lat": lat,
        "lon": lon
    }
)

print(ds)
############################################
# Convert lat-lon grid to HEALPix grid (again at level 6 for demonstration)
level = 6
nside = 2 ** level
llgrid = latlon.LatLonGrid(lat=lat, lon=lon)
hpxgrid = healpix.Grid(level=6)

# Create the regridder from Lat-Lon to HEALPix
regridder = llgrid.get_bilinear_regridder_to(hpxgrid.lon, hpxgrid.lat)

# Regrid the data from Lat-Lon to HEALPix
field_ll = ds["field"].values
field_tensor = torch.from_numpy(field_ll)
field_regridded_hpx = regridder(field_tensor)

field_regridded_hpx_ = field_regridded_hpx.reshape(12, nside, nside)
field_regridded_hpx__ = field_regridded_hpx_.flatten()

# Show the regridded data in HEALPix grid
print(field_regridded_hpx.shape)
print(field_regridded_hpx_.shape)
print(field_regridded_hpx__.shape)
print(field_regridded_hpx == field_regridded_hpx__)
print(field_regridded_hpx[:10], field_regridded_hpx__[:10])
######################################
# Create a HEALPix grid (level 6 for demonstration)
hpxgrid = healpix.Grid(level=6)

# Create the regridder from HEALPix to Lat-Lon grid
latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
regridder = earth2grid.get_regridder(hpxgrid, latlon_grid)


# Regrid the data from HEALPix to Lat-Lon
field_regridded = regridder(field_regridded_hpx)

# Show the regridded data
print(field_regridded.shape)
