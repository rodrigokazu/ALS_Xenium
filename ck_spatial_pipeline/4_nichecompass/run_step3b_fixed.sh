#!/bin/bash
#SBATCH --job-name=step3b_redo
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/step3b_redo_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/step3b_redo_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/run_step3b_fixed.sh
#
# The wrapper that produced the canonical object: job 52579392 (step3b_redo), batch, 16 CPUs,
# 250G, 8 hours. Earlier step 3 jobs 52579272, 52579296 and 52579367 are superseded.
# ========================================================================================
set -euo pipefail
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1
cd /home/rodrigok/SLURM_jobs/Spatial
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python nc3_step3b_subcluster.py
