#!/bin/bash
#SBATCH --job-name=nc3_step1
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/nc3_step1_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/nc3_step1_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/run_nc3_step1.sh
#
# Wrapper for step 1, batch, 8 CPUs, 220G, 4 hours.
# ========================================================================================
set -euo pipefail
# The nichecompass venv rides on the python/3.11.1 module — without it the
# interpreter cannot find libpython3.11.so.1.0. gcc/13.3.0 pairs with it.
# This is exactly what NC_v2_integrated_whole.sbatch does.
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1
cd /home/rodrigok/SLURM_jobs/Spatial
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python nc3_step1.py
