#!/bin/bash
#SBATCH --job-name=cozi_ncv3
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/cozi_ncv3_%j.log
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/cozi_ncv3_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/run_cozi_ncv3.sbatch
#
# The current whole-section run on the canonical v3 object with niche_v2, 8 CPUs, 128G.
# Output cozi_als/20260915_022205_ncv3: 13 of 121 cell-type pairs and 11 of 49 niche pairs
# differ between ALS and control.
# ========================================================================================
set -euo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/NicheDE
B=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
OUT=$B/cozi_als/$(date +%Y%m%d_%H%M%S)_ncv3
mkdir -p "$OUT"
# v2 numbering, same COZI parameters as the v1 run (k=10, 300 permutations, seed 1)
# so the two are directly comparable.
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cozi_env/bin/python \
  cozi_als.py --outdir "$OUT" \
  --obj $B/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad \
  --niche-col niche_v2
echo "OUT=$OUT"
