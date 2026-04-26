"""Generate Figure 8: Multi-initialization RMSE spread for each event.
Memory-efficient: processes one init at a time."""

import gc
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_pub_style, load_forecast, load_era5_daily, align_times,
                    compute_rmse, CASE_STUDIES, CASE_COLORS, FIG_DIR)

set_pub_style()

VAR = "volumetric_soil_water_layer_1"


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating multi-initialization RMSE figure...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for idx, (key, info) in enumerate(CASE_STUDIES.items()):
        ax = axes[idx]
        print(f"\n  Processing {info['label']} ({len(info['inits'])} inits)...")

        all_rmse = []
        all_leads = []

        for init_file in info["inits"]:
            try:
                ds_pred = load_forecast(init_file)
                t0 = str(ds_pred.time.values[0])[:10]
                t1 = str(ds_pred.time.values[-1])[:10]

                ds_era = load_era5_daily(t0, t1, variables=[VAR])
                ds_pred_a, ds_era_a = align_times(ds_pred, ds_era)
                leads = ds_pred_a.prediction_timedelta.values

                rmse = compute_rmse(ds_pred_a, ds_era_a, VAR)
                all_rmse.append(rmse.values)
                all_leads.append(leads)
                ax.plot(leads, rmse.values, color=CASE_COLORS[key],
                        alpha=0.25, linewidth=0.6, linestyle='-')

                del ds_pred, ds_era, ds_pred_a, ds_era_a, rmse
                gc.collect()
                print(f"    {init_file}: OK")
            except Exception as e:
                print(f"    {init_file}: FAILED ({e})")

        if all_rmse:
            min_len = min(len(r) for r in all_rmse)
            trimmed = np.array([r[:min_len] for r in all_rmse])
            leads_common = all_leads[0][:min_len]
            mean_rmse = trimmed.mean(axis=0)
            ax.plot(leads_common, mean_rmse, color=CASE_COLORS[key],
                    linewidth=2.2, label="Ensemble mean")
            ax.fill_between(leads_common, trimmed.min(axis=0), trimmed.max(axis=0),
                            color=CASE_COLORS[key], alpha=0.25, label="Min-max spread")

        ax.set_title(info["label"], fontweight='normal', fontsize=10)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_ylabel("RMSE (m³/m³)")
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_multi_init.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")
