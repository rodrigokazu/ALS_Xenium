#!/bin/bash
#SBATCH --job-name=rerun_ctm
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/rerun_ctm_%j.log
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/logs/rerun_ctm_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/rerun_all_ctm.sbatch
#
# The 2026-09-04 orchestrator on the v2 object (1,097,669 cells, latent_leiden_0.3). It runs
# niche DE and COZI for both WDR49 and MN focal modes, whole and VH, then make_figures.
# These are the runs the Niche DE and COZI Explorer shows.
# ========================================================================================
set -euo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/NicheDE
B=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
TS=$(date +%Y%m%d_%H%M%S)
LP=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/lianaplus/bin/python
CP=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cozi_env/bin/python
echo "ALL RUNS USE --celltype-col cell_type_mn (default). Coarse cell_type MN class is DEPRECATED."

echo "########## 1. WDR49+ astrocyte niche DE ##########"
O1=$B/niche_de_wdr49/$TS; mkdir -p "$O1"
$LP niche_de_v2.py --outdir "$O1"
echo "O1=$O1"

echo "########## 2. canonical-MN niche DE ##########"
O2=$B/niche_de_mn/$TS; mkdir -p "$O2"
$LP niche_de_v2.py --outdir "$O2" --focal-mode mn_canonical --focal-label MN_canonical \
    --niches 0,1,2 --min-cells 10 --min-counts 500
echo "O2=$O2"

echo "########## 3. COZI whole section ##########"
O3=$B/cozi_whole/$TS; mkdir -p "$O3"
$CP cozi_als.py --outdir "$O3"
echo "O3=$O3"

echo "########## 4. COZI within the ventral-horn mask ##########"
O4=$B/cozi_vh/$TS; mkdir -p "$O4"
$CP cozi_als.py --outdir "$O4" --restrict-vh
echo "O4=$O4"

echo "########## 5. figures ##########"
$LP make_figures.py --de "$O1" --cozi "$O3" --outdir "$O1/figures"
$LP make_figures.py --de "$O2" --cozi "$O4" --outdir "$O2/figures"
echo "TS=$TS"
