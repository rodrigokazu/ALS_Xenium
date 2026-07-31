#!/bin/bash
# run_dpqc_array_FINAL.sh -- per-sample Dual-pass QC (_FINAL re-run), OAK-quota-safe.
# Each array task computes ONE sample to node-local /tmp, then tars ONE file into oak pack/.
# %x_%A_%a is the array-correct specialisation of the mission's %x_%j log pattern (one log/task).
# SUBMIT (do NOT run now): A=$(sbatch --parsable run_dpqc_array_FINAL.sh)
#                          sbatch --dependency=afterany:$A run_dpqc_aggregate_FINAL.sh
#SBATCH --job-name=dpqc_arr_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-19%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:50:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_arr_FINAL_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_arr_FINAL_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_dpqc_array_FINAL.sh
#
# The per-sample dual-pass QC as an array: --array=0-19%6, 8 CPUs and 90G per task, 50 minutes
# each.
#
# Each task writes to node-local scratch and tars the result to OAK at the end. That is worth
# keeping. Twenty tasks writing many small figure files straight to shared storage is slow and
# antisocial.
#
# The aggregate job downstream is wired afterany rather than afterok, so one failed section
# does not block the cohort roll-up over the other nineteen.
# ========================================================================================

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} start=$(date)"
source /home/rodrigok/.bashrc
set -u
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export NUMBA_CACHE_DIR=/tmp/nb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MPLCONFIGDIR=/tmp/mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=/tmp/xdg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
DPQC=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/dualpass_qc
PACK=$DPQC/pack
mkdir -p "$PACK"
OUT=/tmp/dpqc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

cd "$CODE"
python dpqc_persample_FINAL.py --idx ${SLURM_ARRAY_TASK_ID} --outroot "$OUT"; rc=$?
SAMP=$(ls "$OUT"/stats/*__dpqc.json 2>/dev/null | head -1 | xargs -n1 basename 2>/dev/null | sed 's/__dpqc.json//')
if [ -n "$SAMP" ]; then
  tar czf "$PACK/${SAMP}.tar.gz" -C "$OUT" figures pdf stats tables && echo "packed $SAMP rc=$rc"
else
  echo "NO OUTPUT rc=$rc"
fi
echo "done task=${SLURM_ARRAY_TASK_ID} end=$(date) rc=$rc"
exit $rc
