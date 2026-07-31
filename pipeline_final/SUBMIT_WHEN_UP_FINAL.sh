#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/SUBMIT_WHEN_UP_FINAL.sh
#
# A small convenience for a specific annoyance: SSH to SCG needs Duo, so a submission cannot
# be fired from an unattended session. This polls until the cluster is reachable and then
# submits, so you can queue the intent and walk away.
#
# It is a scheduling nicety and nothing more. Everything about what actually runs lives in
# SUBMIT_ALL_FINAL.sh.
# ========================================================================================

# SUBMIT_WHEN_UP_FINAL.sh -- one-command submit for the _FINAL tSNE parameter sweep,
# once (a) the astro_sanity_FINAL checkpoint exists and (b) the SLURM controller is up.
# Submits the prep job (fixed 200k subsample) then the GPU array (24 combos, waits on
# prep). Retries around a flaky controller.
#
# *** STAGING NOTE: this is a HELPER the user runs by hand LATER. Nothing is submitted
# during staging. Run manually only after astro_sanity_FINAL has produced
# astro_isolation_FINAL/clustered_full.h5ad:   bash SUBMIT_WHEN_UP_FINAL.sh
#
# ADAPTED FROM /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/SUBMIT_WHEN_UP.sh.
set -uo pipefail
cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
submit(){ for i in $(seq 1 30); do OUT=$(sbatch --parsable "$@" 2>&1)
  if echo "$OUT" | grep -qE '^[0-9]+'; then echo "$OUT"; return 0; fi
  echo "  (controller busy, retry $i) $OUT" >&2; sleep 10; done; echo "FAILED"; return 1; }

CKPT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/astro_isolation_FINAL/clustered_full.h5ad
if [ ! -f "$CKPT" ]; then
  echo "REFUSING: checkpoint not found: $CKPT"
  echo "Run run_astro_sanity_FINAL.sh first (it writes clustered_full.h5ad)."; exit 1
fi

echo "Submitting prep job..."
PREP=$(submit run_tsne_sweep_prep_FINAL.sh) || { echo "prep submit failed"; exit 1; }
echo "PREP_JOB=$PREP"
echo "Submitting GPU array (depends on prep)..."
ARR=$(submit --dependency=afterok:$PREP run_tsne_sweep_FINAL.sh) || { echo "array submit failed"; exit 1; }
echo "ARRAY_JOB=$ARR"
sleep 3
squeue -u rodrigok -o "%.14i %.11j %.8T %.10M %R"
echo "Done. Outputs -> /oak/.../RK/Spatial/astro_isolation_FINAL/tsne_sweep/ (tsne_p*_late*.png + coords_*.npy)"
