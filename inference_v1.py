import xarray as xr
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import earth2grid
from copy import deepcopy
import numpy as np
import wandb

# Initialize wandb
wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")  # Replace with your actual key or use `wandb.login()` interactively
wandb.init(project="healpix-soil-model", config={
    "learning_rate": 1e-3,
    "architecture": "UNet",
    "epochs_per_day": 10,
    "optimizer": "Adam",
    "loss": "MSELoss"
})


# -------------------- Model --------------------
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
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False, out_activation=None):
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

# -------------------- Settings --------------------
year = 2019
zarr_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
norm_file = 'normalization.json'
soil_vars = [
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    'soil_temperature_level_1', 'soil_temperature_level_2',
    'soil_temperature_level_3', 'soil_temperature_level_4'
]
level = 6
nside = 2 ** level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used for training is', device)

# -------------------- Load Data --------------------
ds = xr.open_zarr(zarr_path, consolidated=True, storage_options={"token": "anon"})
with open(norm_file, 'r') as f:
    norm_stats = json.load(f)
vars_to_use = list(norm_stats.keys())

ds_year = ds.sel(time=str(year))
ds_daily = ds_year.groupby('time.dayofyear').mean('time')

sample_var = ds_daily[vars_to_use[0]].isel(dayofyear=0)
if 'latitude' in sample_var.dims and not sample_var.latitude.values[0] < sample_var.latitude.values[-1]:
    sample_var = sample_var.sortby('latitude')
nlat, nlon = sample_var.shape[-2:]
src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
hpx_grid = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
regridder = earth2grid.get_regridder(src_grid, hpx_grid)
regridder_back = earth2grid.get_regridder(hpx_grid, src_grid)

# -------------------- Dataset --------------------
class HealpixSampleDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Initialize model
model = UNet(n_channels=len(vars_to_use), n_classes=len(soil_vars), out_activation=None).to(device)

# Load the best model weights
best_model_path = "best_unet_model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set the model to evaluation mode

# Load the normalization statistics
with open('normalization.json', 'r') as f:
    norm_stats = json.load(f)

# Load first time step of all variables (atmospheric and soil variables)
ds_year = ds.sel(time=str(year))  # Select the year for inference (can be changed)
ds_daily = ds_year.groupby('time.dayofyear').mean('time')

input_vars = []

# Prepare the input data for all variables (atmospheric + soil)
for var in vars_to_use:
    if var not in ds_daily:
        continue
    v = ds_daily[var].isel(dayofyear=0)  # Use the first time step for prediction
    if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
        v = v.sortby('latitude')
    v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
    v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    
    # Normalize the variable
    v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
    input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

# Stack the input variables to create the input tensor
x_tensor = torch.stack(input_vars).float()

# Regrid the input variables using the regridder
input_healpix = torch.stack([
    regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
])

# Add an additional batch dimension for the model (1, C, D, H, W)
X = input_healpix.unsqueeze(0).to(device)

print(X.shape)

# Generate forecast for the first time step
with torch.no_grad():
    forecast = model(X)

# Convert the forecast to a numpy array for further processing
forecast_np = forecast.squeeze(0).cpu().numpy()


print(vars_to_use)

##############################################
####### First forecast #######################
##############################################

# Load the normalization statistics
with open('normalization.json', 'r') as f:
    norm_stats = json.load(f)

# Load the first time step of all variables (atmospheric + soil)
ds_year = ds.sel(time=str(year))  # Select the year for inference (can be changed)
ds_daily = ds_year.groupby('time.dayofyear').mean('time')

print(ds_daily)

input_vars = []

# Prepare the input data for all variables (atmospheric + soil)
for var in vars_to_use:
    if var not in ds_daily:
        continue
    v = ds_daily[var].isel(dayofyear=0)  # Use the first time step for prediction
    if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
        v = v.sortby('latitude')
    v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
    v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    
    # Normalize the variable
    v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
    input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

# Stack the input variables to create the input tensor
x_tensor = torch.stack(input_vars).float()

# Regrid the input variables using the regridder
input_healpix = torch.stack([
    regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
])

# Add an additional batch dimension for the model (1, C, D, H, W)
X = input_healpix.unsqueeze(0).to(device)

# Generate forecast for the first time step
with torch.no_grad():
    forecast = model(X)

# Convert the forecast to a numpy array for further processing
forecast_np = forecast.squeeze(0).cpu().numpy()

####################################################
########### Roll a forecast manually ###############
####################################################
input_vars = []
lead_time = 1
print('forecast_np.shape', forecast_np.shape)
# Prepare the input data for all variables (atmospheric + soil)
for var in vars_to_use:
    if var not in ds_daily:
        continue
    v = ds_daily[var].isel(dayofyear=lead_time)  # Use the first time step for prediction
    if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
        v = v.sortby('latitude')
    v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
    v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    
    # Normalize the variable
    v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
    input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

# Stack the input variables to create the input tensor
x_tensor = torch.stack(input_vars).float()

# Regrid the input variables using the regridder
input_healpix = torch.stack([
    regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
])
print('input_healpix.shape', input_healpix.shape)

input_healpix[-8:,:,:,:] = torch.tensor(forecast_np, dtype=torch.float32)

print('input_healpix.shape', input_healpix.shape)

# Add an additional batch dimension for the model (1, C, D, H, W)
X = input_healpix.unsqueeze(0).to(device)

# Generate forecast for the first time step
with torch.no_grad():
    forecast = model(X)

# Convert the forecast to a numpy array for further processing
forecast_np = forecast.squeeze(0).cpu().numpy()


print(forecast_np.shape)


forecast_flat = forecast_np[:1,:,:,:].flatten()
print(forecast_flat.shape, forecast_np[:1,:,:,:].shape)
out = regridder_back(torch.from_numpy(forecast_flat).double())

print(out.shape)

out_ = []

for i in range(forecast_np.shape[0]):  # Iterate over the first dimension (size 8)
    batch_data = forecast_np[i, :, :, :].flatten()  # Select data for one time step (12, 64, 64)
    out = regridder_back(torch.from_numpy(batch_data).double())
    out_.append(out)

# Concatenate all regridded batches along the first dimension (8)
out__ = torch.stack(out_, dim=0)  # Stack along the batch dimension (time steps)

print(out__.shape)
print(soil_vars)

ds_daily_ = ds_daily[soil_vars].isel(dayofyear=lead_time)

for i_soil_var in range(len(soil_vars)):
    ds_daily_[soil_vars[i_soil_var]+str(lead_time)] = (('latitude', 'longitude'), out__[i_soil_var, :, :])

ds_daily_.to_netcdf('pred_'+str(lead_time)+'.nc')


####################################################
######## Roll out forecast by for loop #############
####################################################

# Number of days (or steps) you want to roll forward
n_steps = 10  # change as needed

forecast_np = None  # will be updated at each step

for lead_time in range(1, n_steps + 1):
    print(f"\n===== Lead time {lead_time} =====")

    input_vars = []
    for var in vars_to_use:
        if var not in ds_daily:
            continue
        v = ds_daily[var].isel(dayofyear=lead_time)  # pick correct day
        if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
            v = v.sortby('latitude')
        v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
        
        # Normalize
        v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
        input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

    # Stack inputs
    x_tensor = torch.stack(input_vars).float()

    # Healpix regridding
    input_healpix = torch.stack([
        regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
    ])

    # Insert previous forecast into soil variables
    if forecast_np is not None:
        input_healpix[-len(soil_vars):, :, :, :] = torch.tensor(forecast_np, dtype=torch.float32)

    # Run model
    X = input_healpix.unsqueeze(0).to(device)
    with torch.no_grad():
        forecast = model(X)
    forecast_np = forecast.squeeze(0).cpu().numpy()

    # Regrid outputs back to lat/lon
    out_ = []
    for i in range(forecast_np.shape[0]):  
        batch_data = forecast_np[i, :, :, :].flatten()
        out = regridder_back(torch.from_numpy(batch_data).double())
        out_.append(out)
    out__ = torch.stack(out_, dim=0)

    # Attach outputs to dataset for inspection/saving
    ds_daily_ = ds_daily[soil_vars].isel(dayofyear=lead_time)
    for i_soil_var in range(len(soil_vars)):
        ds_daily_[soil_vars[i_soil_var] + f"_{lead_time}"] = (('latitude', 'longitude'), out__[i_soil_var, :, :])

    # Save each lead time
    out_file = f"pred_{lead_time}.nc"
    ds_daily_.to_netcdf(out_file)
    print(f"Saved forecast to {out_file}, shape={out__.shape}")
