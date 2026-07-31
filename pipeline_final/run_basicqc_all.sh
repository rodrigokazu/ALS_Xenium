#!/bin/bash
#SBATCH --job-name=basicqc_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/basicqc_FINAL_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/basicqc_FINAL_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_basicqc_all.sh
#
# Loops the basic QC page over all 20 samples in one modest job: 4 CPUs, 32G, 40 minutes.
#
# Not an array, because each page is cheap and twenty short array tasks would spend more time
# queueing than plotting.
#
# Accumulates a return code across the loop instead of exiting on the first failure, so one
# bad sample still leaves you nineteen pages. Check the log; a non-zero exit here means at
# least one sample produced nothing.
# ========================================================================================

echo "host=$(hostname) start=$(date)"
source /home/rodrigok/.bashrc
set -u
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/mpl_basicqc_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PERSAMPLE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/per_sample
OUT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/dualpass_qc/basicqc
mkdir -p "$OUT"
cd "$CODE"

rc=0
for h5 in "$PERSAMPLE"/*__niches_independent.h5ad; do
  base=$(basename "$h5")
  sample=${base%%__niches_independent.h5ad}
  python qc_basic_FINAL.py --sample "$sample" --h5 "$h5" --out "$OUT/${sample}__basicqc.pdf" || rc=$?
done
echo "done end=$(date) rc=$rc"
exit $rc
