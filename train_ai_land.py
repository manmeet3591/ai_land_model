import xarray as xr
import torch
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# --- Inputs ---
zarr_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
norm_file = 'normalization.json'
time_index = 86400*5  # input at t, target will be t+1

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
        # Ensure latitude is increasing for interpolation
    if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
        v = v.sortby('latitude')
    # Fill NaNs via bilinear interpolation (in lat-lon dims)
    # Fill NaNs bilinearly over lat/lon
    v_interp = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
    v_interp = v_interp.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    v_min = norm_stats[var]["min"]
    v_max = norm_stats[var]["max"]
    v_norm = (v_interp - v_min) / (v_max - v_min)
#    v_norm = (v - v_min) / (v_max - v_min)
    input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

input_tensor = torch.stack(input_vars)  # shape [C, lat, lon]

# --- Target: only soil vars at t+1 ---
target_vars = []
for var in soil_vars:
    if var not in ds:
        print(f"⚠️ Missing target: {var}")
        continue
    v = ds[var].isel(time=time_index + 1).compute()
        # Ensure latitude is increasing for interpolation
    if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
        v = v.sortby('latitude')
    v_min = norm_stats[var]["min"]
    v_max = norm_stats[var]["max"]
    v_interp = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
    v_interp = v_interp.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
    v_norm = (v_interp - v_min) / (v_max - v_min)

    #v_norm = (v - v_min) / (v_max - v_min)
    target_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

target_tensor = torch.stack(target_vars)  # shape [8, lat, lon]

# --- Done ---
print("✅ Input shape :", input_tensor.shape)
print("✅ Target shape:", target_tensor.shape)

print("✅ Checking NaNs in tensors:")
print("Input contains NaNs:", torch.isnan(input_tensor).any().item())
print("Target contains NaNs:", torch.isnan(target_tensor).any().item())



import earth2grid

# --- HEALPix settings ---
level = 6  # HEALPix resolution level; 2^level = nside
device = "cpu"
nside = 2 ** level

# --- Setup source and target grids ---
nlat, nlon = input_tensor.shape[1:]
src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
hpx_grid = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
regridder = earth2grid.get_regridder(src_grid, hpx_grid)

# Move regridder to match tensor device and dtype
regridder.to(input_tensor)

# --- Convert input tensor to HEALPix ---
input_healpix = torch.stack([
    regridder(var).reshape(12, nside, nside) for var in input_tensor
])
# input_healpix shape: [C, 12, nside, nside]

# --- Convert target tensor to HEALPix ---
target_healpix = torch.stack([
    regridder(var).reshape(12, nside, nside) for var in target_tensor
])
# target_healpix shape: [8, 12, nside, nside]

# --- Done ---
print("✅ HEALPix Input shape :", input_healpix.shape)   # e.g. [C, 12, 64, 64]
print("✅ HEALPix Target shape:", target_healpix.shape)  # e.g. [8, 12, 64, 64]

print("HEALPix Input NaNs:", torch.isnan(input_healpix).any().item())
print("HEALPix Target NaNs:", torch.isnan(target_healpix).any().item())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

# --- Simple Dataset Class ---
class HealpixSampleDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs  # torch.Tensor [C, 12, 64, 64]
        self.targets = targets  # torch.Tensor [8, 12, 64, 64]

    def __len__(self):
        return 1  # single example

    def __getitem__(self, idx):
        return self.inputs, self.targets

# --- Make sure input tensors are on CPU and float32 ---
input_healpix = input_healpix.float().cpu()
target_healpix = target_healpix.float().cpu()

# --- Loaders ---
dataset = HealpixSampleDataset(input_healpix, target_healpix)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Model ---
model = UNet(
    n_channels=input_healpix.shape[0],
    n_classes=target_healpix.shape[0],
    out_activation=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Loss and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)  # shape: [1, C, 12, 64, 64]
        y = y.to(device)  # shape: [1, 8, 12, 64, 64]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:02d} - Loss: {total_loss:.6f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    test_input = input_healpix.unsqueeze(0).to(device)
    test_output = model(test_input).squeeze(0).cpu()

print("✅ Done. Predicted shape:", test_output.shape)  # [8, 12, 64, 64]

