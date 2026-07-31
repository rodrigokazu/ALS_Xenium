#!/bin/bash
# run_dpqc_aggregate_FINAL.sh -- cohort roll-up for the Dual-pass QC (_FINAL re-run).
# Extracts the per-sample packs (oak pack/) to node-local /tmp, builds Cohort/MeetingGuide/Audio/
# Strategy PDFs (+ audio .txt), then tars ONE final bundle to oak final/. OAK-quota-safe.
# SUBMIT with a dependency on the array (do NOT run now):
#   sbatch --dependency=afterany:<ARRAY_JOBID> run_dpqc_aggregate_FINAL.sh
#SBATCH --job-name=dpqc_agg_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_agg_FINAL_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/dpqc_agg_FINAL_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_dpqc_aggregate_FINAL.sh
#
# The roll-up job. Runs the cohort PDF, the meeting guide, the audio edition and the strategy
# document in sequence off the per-sample stats: 4 CPUs, 48G, 35 minutes.
#
# Each of the four is allowed to fail independently with its return code collected, so a crash
# in the audio narrative does not cost you the cohort PDF.
#
# Depends on the array via afterany, so it runs over whatever sections completed.
# ========================================================================================

echo "host=$(hostname) start=$(date)"
source /home/rodrigok/.bashrc
set -u
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export NUMBA_CACHE_DIR=/tmp/nb_agg_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/mpl_agg_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_agg_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
DPQC=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/dualpass_qc
PACK=$DPQC/pack
FINAL=$DPQC/final
mkdir -p "$FINAL"
ALL=/tmp/dpqc_all_${SLURM_JOB_ID}
mkdir -p "$ALL/cohort"

n=0; for t in "$PACK"/*.tar.gz; do [ -e "$t" ] || continue; tar xzf "$t" -C "$ALL" && n=$((n+1)); done
echo "extracted $n sample packs from $PACK"

cd "$CODE"
rc=0
python dpqc_cohort_FINAL.py   --statsdir "$ALL/stats" --figroot "$ALL/figures" --out "$ALL/cohort/DualPassQC_Cohort.pdf" || rc=$?
python dpqc_meeting_FINAL.py  --statsdir "$ALL/stats" --figroot "$ALL/figures" --out "$ALL/cohort/DualPassQC_MeetingGuide.pdf" || rc=$?
python dpqc_audio_FINAL.py    --statsdir "$ALL/stats" --out "$ALL/cohort/DualPassQC_Cohort_Audio_RKS.pdf" --txt "$ALL/cohort/DualPassQC_Cohort_Audio_RKS.txt" || rc=$?
python dpqc_strategy_FINAL.py --outdir "$ALL/cohort" --statsdir "$ALL/stats" || rc=$?

tar czf "$FINAL/DualPassQC_ALL.tar.gz" -C "$ALL" figures pdf stats tables cohort
echo "AGGREGATE DONE end=$(date) rc=$rc"; ls -la "$FINAL/"
exit $rc
