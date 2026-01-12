# ai_land_model

$ apptainer build apptainer_al_land.sif apptainer_al_land.def

$ apptainer exec --nv apptainer_al_land.sif python train_ai_land_v3.py

$ apptainer exec --nv apptainer_al_land.sif python infer_healpix_soil.py \
  --init_time 2019-01-01T00 \
  --year 2019 \
  --weights best_unet_model_entire_year.pth \
  --n_steps 10 \
  --out soil_forecast.nc
