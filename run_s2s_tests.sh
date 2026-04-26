#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# User settings
# ---------------------------
SIF="apptainer_al_land.sif"
PY="python"
SCRIPT="inference_v2.py"
WEIGHTS="best_unet_model_entire_year.pth"
N_STEPS=60
OUTDIR="runs_nc"
LOGDIR="logs"

# Max concurrent jobs you *want* (still gated by free VRAM check below)
MAX_JOBS=3

# GPU VRAM gate (MiB): only launch a new job if FREE >= this
MIN_FREE_MIB=9000

# How often to re-check VRAM when it's too full
POLL_SECONDS=20

mkdir -p "$OUTDIR" "$LOGDIR"

# ---------------------------
# Test init times
# ---------------------------
INIT_TIMES=(
  # 2012 Central U.S. flash drought
  "2012-05-15T00"
  "2012-06-01T00"
  "2012-06-15T00"
  "2012-07-01T00"

  # 2019 Midwest / Mississippi Basin flooding
  "2019-03-01T00"
  "2019-03-15T00"
  "2019-04-01T00"
  "2019-05-01T00"

  # 2021 Pacific Northwest heat dome
  "2021-06-10T00"
  "2021-06-17T00"
  "2021-06-24T00"
  "2021-06-27T00"

  # 2011 Texas drought onset
  "2011-02-15T00"
  "2011-03-01T00"
  "2011-04-01T00"
  "2011-06-01T00"
)

# ---------------------------
# Helpers
# ---------------------------
get_year() {
  # "YYYY-MM-DDT00" -> "YYYY"
  echo "${1:0:4}"
}

free_vram_mib() {
  # Returns free VRAM on GPU 0 (MiB). Assumes nvidia-smi exists.
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' '
}

wait_for_vram() {
  while true; do
    local free
    free="$(free_vram_mib || echo 0)"
    if [[ "$free" -ge "$MIN_FREE_MIB" ]]; then
      return 0
    fi
    echo "[gate] Free VRAM ${free} MiB < ${MIN_FREE_MIB} MiB; sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

run_one() {
  local init="$1"
  local year out log

  year="$(get_year "$init")"
  out="${OUTDIR}/earthmind_ai_land_${init//[:T-]/}.nc"   # sanitize into filename
  log="${LOGDIR}/run_${init//[:T-]/}.log"

  echo "[start] init=${init} year=${year} -> ${out}"
  (
    set -euo pipefail
    apptainer exec --nv "$SIF" "$PY" "$SCRIPT" \
      --init_time "$init" \
      --year "$year" \
      --weights "$WEIGHTS" \
      --n_steps "$N_STEPS" \
      --out "$out"
  ) >"$log" 2>&1
  echo "[done ] init=${init}"
}

# Retry wrapper so one failure doesn't kill the whole batch
run_one_with_retry() {
  local init="$1"
  local tries=2
  local i=1
  while true; do
    if run_one "$init"; then
      return 0
    fi
    if [[ "$i" -ge "$tries" ]]; then
      echo "[fail] init=${init} after ${tries} attempts (see logs)"
      return 1
    fi
    echo "[retry] init=${init} attempt $((i+1))/${tries} after 30s..."
    sleep 30
    i=$((i+1))
  done
}

# ---------------------------
# Main scheduling loop
# ---------------------------
echo "Launching up to ${MAX_JOBS} jobs concurrently, gated by >= ${MIN_FREE_MIB} MiB free VRAM."

pids=()

for init in "${INIT_TIMES[@]}"; do
  # Respect MAX_JOBS
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]]; do
    sleep 2
  done

  # Respect VRAM gate
  wait_for_vram

  # Launch in background
  run_one_with_retry "$init" &
done

# Wait for all to finish; return nonzero if any failed
wait
echo "All runs complete."
