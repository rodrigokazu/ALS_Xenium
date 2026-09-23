#!/bin/bash
#SBATCH --job-name=cozi_ncv3_vh
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/cozi_ncv3_vh_%j.log
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/cozi_ncv3_vh_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/run_cozi_ncv3_vh.sbatch
#
# The same run restricted to the ventral horn (73,020 cells, 9 ALS and 8 control sections).
# Output cozi_als_vh/20260915_081701_ncv3: 0 of 121 pairs differ.
# ========================================================================================
set -euo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/NicheDE
B=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
OUT=$B/cozi_als_vh/$(date +%Y%m%d_%H%M%S)_ncv3
mkdir -p "$OUT"
# ventral-horn scope on the RETAINED object (965,285 cells, v2 niches); same k=10 / 300 perms / seed 1
# as the whole-section ncv3 run 20260915_022205 so the two scopes are directly comparable.
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cozi_env/bin/python \
  cozi_als.py --outdir "$OUT" \
  --obj $B/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad \
  --niche-col niche_v2 --restrict-vh
echo "OUT=$OUT"
