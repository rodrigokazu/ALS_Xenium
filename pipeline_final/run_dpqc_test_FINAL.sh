#!/bin/bash
# run_dpqc_test_FINAL.sh -- single-sample end-to-end smoke test for the Dual-pass QC (_FINAL re-run).
# Recommended FIRST launch step once Novae_persample_INDEPENDENT_FINAL + CellTyping_coarse_FINAL exist:
# it exercises the full per-sample path (compute -> figures -> PDF) on one section and copies a few
# artifacts to the code logs dir for eyeballing. Do NOT run now (staging only).
#   sbatch run_dpqc_test_FINAL.sh
#SBATCH --job-name=dpqc_e2e_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:45:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_e2e_FINAL_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_e2e_FINAL_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_dpqc_test_FINAL.sh
#
# Single-sample smoke test of the whole dual-pass QC chain, hard-wired to --idx 8.
#
# Run this after touching anything in the dpqc set and before launching the array. It
# exercises compute, figures and report assembly end to end in about 45 minutes. Far cheaper
# than discovering a broken import twenty tasks into a real run.
# ========================================================================================

set -u
echo "host=$(hostname) start=$(date)"
source /home/rodrigok/.bashrc
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
OUT=/tmp/dpqc_e2e_FINAL_${SLURM_JOB_ID}
cd "$CODE"
# idx 8 is arbitrary; swap for --sample SD03614_BG to target a known-good section at launch.
python dpqc_persample_FINAL.py --idx 8 --outroot "$OUT"; rc=$?
echo "=== outputs ==="; ls -R "$OUT" 2>/dev/null
for f in "$OUT"/figures/*/*__master.png "$OUT"/figures/*/*__highlight.png "$OUT"/figures/*/*__markers.png "$OUT"/figures/*/*__qcpanels.png "$OUT"/pdf/*.pdf; do
  cp "$f" "$CODE/logs/" 2>/dev/null
done
echo "done end=$(date) rc=$rc"
exit $rc
