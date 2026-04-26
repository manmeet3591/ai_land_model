# Nachiketa's Comments - Revision Summary for arXiv Version

## Status: COMPLETED (Text Edits) - PENDING (Figure Updates)

### Completed Text Revisions ✓

#### 1. **Title Simplified** ✓
- **Old:** "Earthmind-Land: A HEALPix-Based 3D UNet Land Surface Model for Subseasonal-to-Seasonal Prediction"
- **New:** "A Pioneering AI Land Surface Model for Subseasonal-to-Seasonal Prediction"
- Removed technical jargon (3D UNet, HEALPix) as requested

#### 2. **Abstract Enhanced** ✓
- Added emphasis on "first standalone AI land surface model"
- Clarified offline vs. coupled distinction
- Added statement: "offering the potential to be integrated into any physics-based or AI-based coupled model"
- Removed resolution ambiguity

#### 3. **Initial Conditions Clarified** ✓
- Added explicit section in Short-Range Forecast (Section 3.1) explaining:
  - ICs taken from ERA5 reanalysis at each evaluation date
  - Full 8-variable soil state specified for each case
  - Perfect initialization assumption for days 1-9 evaluation

#### 4. **ERA5 Soil Moisture Variables Explained** ✓
- Updated Table 1 caption to include ERA5 variable naming:
  - `volumetric_soil_water_layer_1` through `layer_4`
  - `soil_temperature_level_1` through `level_4`
- Clarified that ERA5 is deterministic (not ensemble); using single best-estimate product

#### 5. **Training Details Table Added** ✓
- Created Table 2 ("Training configuration and deep learning model details") with:
  - **Layer normalization:** Batch normalization after each conv layer
  - **Batch size:** 20
  - **Training years:** 2018-2020
  - **Epochs:** 100
  - **Loss function:** MSE
  - **Optimizer:** Adam (lr = 1e-3)
  - **Normalization strategy:** MinMax to [0,1]

#### 6. **Region Definitions Added** ✓
- Updated Section 3.2 (Extended-Range Forecast Skill) with lat/lon boxes:
  - **2019 Mississippi:** 30-50°N, 75-95°W
  - **2012 Central US Drought:** 35-45°N, 90-105°W
  - **2021 Pacific NW Heat Dome:** 45-50°N, 115-125°W
  - **2011 Texas Drought:** 25-35°N, 95-105°W
- Added statement: "All evaluations exclude ocean grid points; only land pixels are assessed"

#### 7. **RMSE Discussion Enhanced** ✓
- Added discussion of Texas drought anomaly:
  - "notably higher RMSE and steeper error growth compared to other cases"
  - Explanation: "semi-arid Texas climate, combined with ERA5's known representational challenges in this region"
  - "sparse observational networks" leading to larger ERA5 uncertainties
- Clarified error growth is physically consistent and stable

#### 8. **Offline Model Paradigm Clarified** ✓
- Expanded Discussion section (Section 4) with detailed explanation:
  - **Offline:** Land receives prescribed atmospheric forcing; evolves own state
  - **Coupled:** Land and atmosphere exchange feedbacks mutually
  - **Noah-MP context:** Validated offline before GCM coupling
  - **Earthmind strategy:** Following proven LSM development paradigm
- Explicit comparison to Noah-MP and CLM development workflows

#### 9. **Autoregressive Rollout Explained in Detail** ✓
- Rewrote Section 2.3.3 (Autoregressive Rollout Inference) with step-by-step:
  1. Day 1: Use ERA5 soil state IC + ERA5 atmospheric forcing
  2. Day 2+: Use predicted soil state + ERA5 atmospheric forcing
  3. Clarification: Atmosphere held fixed (offline), land state evolves
- Added numbering and clearer distinction between state variables and forcing
- Explained why this isolates land model skill from atmospheric error

#### 10. **Pioneering Nature Emphasized** ✓
- Updated Introduction (Section 1):
  - "No such AI-based land surface model existed prior to this work"
  - Added explicit statement of novelty
  - Clarified modularity: "can be integrated into any coupled AI or physics-based forecasting system"
- Revised key contributions to lead with: "The first standalone AI land surface model, which did not exist in the literature prior to this work"
- Added modularity as a core contribution

#### 11. **Figure Captions Updated (Removed "global")** ✓
- **Fig 1 (Rollout soil moisture):** "Maps of..." instead of "Global maps..."
- **Fig 3 (RMSE):** Removed "Global" prefix; added Texas discussion
- **Fig 4 (ACC):** Removed "Global" and added event names; noted ACC > 0.5 threshold
- **Fig 6 (Multi-init):** Enhanced description of initialization ensemble and robustness
- **Fig 7 (Bias):** Clarified red=wet bias, blue=dry bias; added land-only evaluation note

---

## Still TODO: Figure Updates (Nachiketa's requests)

### Figure 4 (ACC Plot) - Line Color & Formatting
- **Current:** Need to check current line colors and 0.5 threshold visualization
- **TODO:**
  - [ ] Change line colors (specify new color scheme)
  - [ ] Remove or de-emphasize the 0.5 horizontal dashed line (or make it subtle)
  - [ ] Regenerate `figures/fig_acc.pdf`

### Figure 5 (Multi-init envelope)
- **Current:** Shows RMSE envelope across initializations
- **TODO:**
  - [ ] Change colors for better visibility
  - [ ] Ensure shaded region is clear (remove horizontal 0.5 line if present)
  - [ ] Regenerate `figures/fig_multi_init.pdf`

### Pending Figure Generation Notes
- Python scripts in `paper/scripts/` likely generate these figures
- Will need to:
  1. Check current plotting code in `paper/scripts/`
  2. Identify color definitions
  3. Modify and regenerate figures
  4. Recompile PDF

---

## Additional Notes for arXiv Submission

### Positioning for Nature Scientific Reports / Nature Machine Intelligence
1. **Modularity** - Emphasized throughout as key innovation
2. **Pioneering status** - First AI land model in the literature
3. **Practical use** - Can be plugged into existing systems
4. **Physics alignment** - Follows established LSM development paradigm
5. **S2S relevance** - Demonstrates 30+ day skill for subseasonal applications

### Key Messaging Points
- "No AI land model existed before this work"
- Modularity enables integration into any coupled system
- Offline validation follows established physics-based LSM tradition
- Results span diverse extremes (flood, drought, heat)
- Maintains skill beyond 30 days (anomaly correlation > 0.5)

---

## Files Modified
- `/media/airlab/ROCSTOR/ai_land_model/paper/main.tex` - All text revisions completed
- `/media/airlab/ROCSTOR/ai_land_model/paper/main.pdf` - Recompiled (v3)

## Next Steps
1. Update figure colors in Python scripts
2. Regenerate figures (fig_acc.pdf, fig_multi_init.pdf)
3. Recompile final PDF
4. Review for arXiv readiness
5. Consider submission to Nature Scientific Reports in 2-3 weeks
