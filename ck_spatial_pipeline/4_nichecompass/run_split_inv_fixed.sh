#!/bin/bash
#SBATCH --job-name=splitinv
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/splitinv_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/splitinv_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/run_split_inv_fixed.sh
#
# Wrapper for the split inventory, chained afterok on 52579392. Job 52579396.
# ========================================================================================
set -euo pipefail
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1
cd /home/rodrigok/SLURM_jobs/Spatial
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python nc3_split_inventory.py
