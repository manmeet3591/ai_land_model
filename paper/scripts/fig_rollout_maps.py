"""Generate Figure 3 (soil moisture) and Figure 4 (soil temperature):
GT vs Prediction maps at days 1, 5, 9. Shows which initialization (IC) was used."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_pub_style, load_norm, denormalize, PRED_DIR, FIG_DIR, CASE_STUDIES

set_pub_style()
norm = load_norm()

LEADS = [1, 5, 9]

# Extract IC date from forecast filenames in runs_nc directory
def get_ic_date_from_forecasts():
    """Extract initialization date from the available forecast files."""
    runs_dir = "/media/airlab/ROCSTOR/ai_land_model/runs_nc"
    for case_key, case_info in CASE_STUDIES.items():
        primary_file = case_info['primary']
        if primary_file.startswith('earthmind_ai_land_'):
            date_str = primary_file.replace('earthmind_ai_land_', '').replace('.nc', '')
            # Format: YYYYMMDDDD -> YYYY-MM-DD HH
            if len(date_str) >= 8:
                ic_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                if len(date_str) > 8:
                    ic_date += f" {date_str[8:10]}:00 UTC"
                return ic_date, case_key
    return None, None


def flipud_da(da):
    flipped = np.flipud(da.values)
    return xr.DataArray(
        flipped, dims=da.dims,
        coords={d: da.coords[d] for d in da.dims},
        attrs=da.attrs, name=da.name,
    )


def make_panel(base_var, leads, out_name, cmap="viridis", label=None):
    gts, preds, avail = [], [], []
    for lead in leads:
        p = Path(PRED_DIR) / f"pred_{lead}.nc"
        if not p.exists():
            print(f"  Skipping lead {lead}: {p} not found")
            continue
        ds = xr.open_dataset(str(p))
        gt = ds[base_var]
        pred_name = f"{base_var}_{lead}"
        if pred_name not in ds:
            cands = [v for v in ds.data_vars if v.startswith(base_var + "_")]
            pred_name = cands[-1] if cands else None
        if pred_name is None:
            continue
        pred = ds[pred_name]
        for dim in ["time", "step", "member"]:
            if dim in gt.dims:
                gt = gt.isel({dim: 0})
            if dim in pred.dims:
                pred = pred.isel({dim: 0})
        pred = denormalize(pred, base_var, norm)
        pred = flipud_da(pred)
        gts.append(gt)
        preds.append(pred)
        avail.append(lead)

    if not avail:
        print(f"  No data for {base_var}")
        return

    vals = np.concatenate([np.ravel(d.values) for d in gts + preds])
    vmin = float(np.nanquantile(vals, 0.02))
    vmax = float(np.nanquantile(vals, 0.98))

    n = len(avail)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 5.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (lead, gt, pred) in enumerate(zip(avail, gts, preds)):
        im = gt.plot(ax=axes[0, j], vmin=vmin, vmax=vmax, cmap=cmap,
                     add_colorbar=False)
        axes[0, j].set_title(f"Ground Truth (Day {lead})", fontsize=10)
        axes[0, j].set_xlabel("")
        axes[0, j].set_ylabel("")

        pred.plot(ax=axes[1, j], vmin=vmin, vmax=vmax, cmap=cmap,
                  add_colorbar=False)
        axes[1, j].set_title(f"Forecast (Day {lead})", fontsize=10)
        axes[1, j].set_xlabel("")
        axes[1, j].set_ylabel("")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar_label = label or base_var.replace("_", " ")
    plt.colorbar(im, cax=cbar_ax, label=cbar_label)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    out = os.path.join(FIG_DIR, out_name.replace(".pdf", ".png"))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating soil moisture rollout maps...")
    make_panel("volumetric_soil_water_layer_1", LEADS, "fig_rollout_sm.pdf",
               cmap="YlGnBu", label="Volumetric soil water (m$^3$/m$^3$)")

    print("Generating soil temperature rollout maps...")
    make_panel("soil_temperature_level_1", LEADS, "fig_rollout_st.pdf",
               cmap="coolwarm", label="Soil temperature (K)")
