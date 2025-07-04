import xarray as xr
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import earth2grid
from copy import deepcopy

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
year = 2018
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



# -------------------- Dataset --------------------
class HealpixSampleDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# -------------------- Model Init --------------------
model = UNet(n_channels=len(vars_to_use), n_classes=len(soil_vars), out_activation=None).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -------------------- Training Loop --------------------
for t in range(len(ds_daily.dayofyear) - 1):
    print(f"\n📅 Training on day {t} → {t + 1}")
    input_vars = []
    target_vars = []

    for var in vars_to_use:
        if var not in ds_daily:
            continue
        v = ds_daily[var].isel(dayofyear=t)
        if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
            v = v.sortby('latitude')
        v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
        v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
        input_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

    for var in soil_vars:
        if var not in ds_daily:
            continue
        v = ds_daily[var].isel(dayofyear=t + 1)
        if 'latitude' in v.dims and not v.latitude.values[0] < v.latitude.values[-1]:
            v = v.sortby('latitude')
        v = v.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        v = v.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
        v_norm = (v - norm_stats[var]['min']) / (norm_stats[var]['max'] - norm_stats[var]['min'])
        target_vars.append(torch.tensor(v_norm.values, dtype=torch.float32))

    # x_tensor = torch.stack(input_vars)
    # y_tensor = torch.stack(target_vars)
    x_tensor = torch.stack(input_vars).float()
    y_tensor = torch.stack(target_vars).float()
    # input_healpix = torch.stack([regridder(x).reshape(12, nside, nside) for x in x_tensor])
    # target_healpix = torch.stack([regridder(y).reshape(12, nside, nside) for y in y_tensor])
    
    input_healpix = torch.stack([
    regridder(x.double()).reshape(12, nside, nside).float() for x in x_tensor
    ])
    target_healpix = torch.stack([
        regridder(y.double()).reshape(12, nside, nside).float() for y in y_tensor
    ])


    X = input_healpix.unsqueeze(0)
    Y = target_healpix.unsqueeze(0)

    dataset = HealpixSampleDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # -------------------- Training --------------------
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Day {t}] Epoch {epoch:02d} - Loss: {total_loss:.6f}")

    # -------------------- Evaluation --------------------
    model.eval()
    with torch.no_grad():
        sample_output = model(X.to(device)).squeeze(0).cpu()
    print("✅ Done. Predicted shape:", sample_output.shape)

    # Clear memory
    del input_vars, target_vars, x_tensor, y_tensor, input_healpix, target_healpix, X, Y, dataset, dataloader
    torch.cuda.empty_cache()
