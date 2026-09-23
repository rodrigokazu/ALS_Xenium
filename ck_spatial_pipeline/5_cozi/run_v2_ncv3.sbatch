#!/bin/bash
#SBATCH --job-name=niche_de_ncv3
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/niche_de_ncv3_%j.log
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/niche_de_ncv3_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/run_v2_ncv3.sbatch
#
# The current 7-niche DE run on the v3 object. Output niche_de_v2/20260915_021658_ncv3.
# ========================================================================================
set -euo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/NicheDE
B=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
OUT=$B/niche_de_v2/$(date +%Y%m%d_%H%M%S)_ncv3
mkdir -p "$OUT"
# v2 numbering. All SEVEN niches are used: unlike the v1 run (which took 0,1,2,4,5 and
# excluded non-cord niche 3, the one-section niche 6 and the 24/5-cell specks), every v2
# niche is cord tissue and passes the sample-spread gate at 20/20 sections.
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/lianaplus/bin/python \
  niche_de_v2.py --outdir "$OUT" \
  --obj $B/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad \
  --niche-col niche_v2 \
  --niches v2_0,v2_1,v2_2,v2_3,v2_4,v2_5,v2_6
echo "OUT=$OUT"
