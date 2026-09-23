#!/bin/bash
#SBATCH --job-name=Novae_INDEP_AUG5_agg
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/1_novae/run_novae_INDEP_AUG5_agg.sh
#
# Aggregates the AUG5 per-sample rows into the cohort summary. It chains with afterany so a
# dead task shows up as a failed row in the summary instead of silently shrinking the cohort.
# ========================================================================================
# Stitches the 20 per-sample rows/manifests into the cohort summary + run_manifest.json.
# Chained with afterany so it still runs (and lists any missing sample as failed) if a task dies.

source /home/rodrigok/.bashrc
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
export NUMBA_CACHE_DIR="$TMPDIR/numba"; export MPLCONFIGDIR="$TMPDIR/mpl"; export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job started: $(date) on $(hostname)"
cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
python novae_INDEP_AUG5.py aggregate
rc=$?
echo "Job finished: $(date) (exit $rc)"
exit $rc
