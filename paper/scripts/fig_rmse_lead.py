"""Generate Figure 5: RMSE vs lead time for 4 soil moisture layers across 4 cases.
Memory-efficient: processes one case at a time, loads only SM variables from ERA5."""

import gc
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_pub_style, load_forecast, load_era5_daily, align_times,
                    compute_rmse, SM_VARS, SM_LABELS, CASE_STUDIES, FIG_DIR)

set_pub_style()

COLORS = ["#0173b2", "#de8f05", "#029e73", "#cc78bc"]  # Colorblind-friendly palette


def process_one_case(key, info):
    print(f"  Loading forecast {info['primary']}...")
    ds_pred = load_forecast(info["primary"])
    t0 = str(ds_pred.time.values[0])[:10]
    t1 = str(ds_pred.time.values[-1])[:10]

    print(f"  Loading ERA5 daily means {t0} to {t1}...")
    ds_era = load_era5_daily(t0, t1, variables=SM_VARS)

    ds_pred_a, ds_era_a = align_times(ds_pred, ds_era)
    leads = ds_pred_a.prediction_timedelta.values

    results = {}
    for var in SM_VARS:
        rmse = compute_rmse(ds_pred_a, ds_era_a, var)
        results[var] = (leads, rmse.values)

    del ds_pred, ds_era, ds_pred_a, ds_era_a
    gc.collect()
    return results


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating RMSE vs lead time figure...")

    all_results = {}
    for key, info in CASE_STUDIES.items():
        print(f"\nProcessing {info['label']}...")
        try:
            all_results[key] = process_one_case(key, info)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[key] = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for idx, (key, info) in enumerate(CASE_STUDIES.items()):
        ax = axes[idx]
        results = all_results.get(key)
        if results is None:
            ax.set_title(info["label"])
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            continue

        for i, (var, lbl) in enumerate(zip(SM_VARS, SM_LABELS)):
            if var in results:
                leads, rmse_vals = results[var]
                ax.plot(leads, rmse_vals, label=lbl, color=COLORS[i], linewidth=1.5)

        ax.set_title(info["label"], fontweight='normal', fontsize=10)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_ylabel("RMSE (m³/m³)")
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_rmse.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")
