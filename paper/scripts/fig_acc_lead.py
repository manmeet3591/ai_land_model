"""Generate Figure 6: ACC vs lead time for soil moisture L1 across all 4 cases.
Memory-efficient: processes one case at a time."""

import gc
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_pub_style, load_forecast, load_era5_daily, align_times,
                    compute_acc, CASE_STUDIES, CASE_COLORS, FIG_DIR)

set_pub_style()

VAR = "volumetric_soil_water_layer_1"


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating ACC vs lead time figure...")

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, info in CASE_STUDIES.items():
        print(f"  Processing {info['label']}...")
        try:
            ds_pred = load_forecast(info["primary"])
            t0 = str(ds_pred.time.values[0])[:10]
            t1 = str(ds_pred.time.values[-1])[:10]

            ds_era = load_era5_daily(t0, t1, variables=[VAR])
            ds_pred_a, ds_era_a = align_times(ds_pred, ds_era)
            leads = ds_pred_a.prediction_timedelta.values

            acc = compute_acc(ds_pred_a[VAR], ds_era_a[VAR])
            ax.plot(leads, acc.values, label=info["label"],
                    color=CASE_COLORS[key], linewidth=1.8)

            del ds_pred, ds_era, ds_pred_a, ds_era_a, acc
            gc.collect()
        except Exception as e:
            print(f"    FAILED: {e}")

    ax.axhline(y=0.5, color="lightgray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Anomaly Correlation Coefficient")
    ax.set_title("Anomaly Correlation Coefficient: Soil Water Layer 1")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.1, 1.05)

    out = os.path.join(FIG_DIR, "fig_acc.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
