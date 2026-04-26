"""Generate Figure 7: Spatial bias maps at leads 7, 14, 30, 60 days.
Memory-efficient: loads only one variable from ERA5."""

import gc
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_pub_style, load_forecast, load_era5_daily, align_times,
                    FIG_DIR)

set_pub_style()

VAR = "volumetric_soil_water_layer_1"
LEADS_TO_SHOW = [7, 14, 30, 60]
CASE_FILE = "earthmind_ai_land_2019031500.nc"


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating bias maps...")

    ds_pred = load_forecast(CASE_FILE)
    t0 = str(ds_pred.time.values[0])[:10]
    t1 = str(ds_pred.time.values[-1])[:10]

    print(f"  Loading ERA5 daily means {t0} to {t1}...")
    ds_era = load_era5_daily(t0, t1, variables=[VAR])
    ds_pred_a, ds_era_a = align_times(ds_pred, ds_era)
    leads = ds_pred_a.prediction_timedelta.values

    valid_leads = [l for l in LEADS_TO_SHOW if l <= len(leads)]
    n = len(valid_leads)

    biases = []
    for lead in valid_leads:
        idx = lead - 1
        bias = ds_pred_a[VAR].isel(time=idx) - ds_era_a[VAR].isel(time=idx)
        biases.append(bias.compute())

    vmax_bias = max(float(np.nanquantile(np.abs(b.values), 0.98)) for b in biases)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for j, (lead, bias) in enumerate(zip(valid_leads, biases)):
        im = bias.plot(ax=axes[j], vmin=-vmax_bias, vmax=vmax_bias,
                       cmap="RdBu", add_colorbar=False)
        axes[j].set_title(f"Day {lead}")
        axes[j].set_xlabel("")
        axes[j].set_ylabel("")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="Bias (Pred $-$ ERA5)")
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    out = os.path.join(FIG_DIR, "fig_bias.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")

    del ds_pred, ds_era, ds_pred_a, ds_era_a, biases
    gc.collect()
