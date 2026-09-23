#!/bin/bash
#SBATCH --job-name=cellaudit
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/cellaudit_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/cellaudit_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/run_cellaudit.sh
#
# Wrapper for cellaudit.py, 8 CPUs, 180G, 4 hours, cozi_env. Job 52504830.
# ========================================================================================
set -euo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cozi_env/bin/activate
python3 cellaudit.py /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC/cell_audit
