#!/bin/bash

#SBATCH --job-name=xemb_swee          # Job name
#SBATCH --output=logs/%x_%j.out      # STDOUT
#SBATCH --error=logs/%x_%j.err       # STDERR
#SBATCH --partition=batch      # Partition
#SBATCH --cpus-per-task=16           # 16 CPU cores
#SBATCH --mem=512G                   # 128 GB system RAM
#SBATCH --time=48:00:00               # 6-hour wall-clock time
#SBATCH --account=mpsnyder           # your SCG account
# ========================================================================================
# ALS_Xenium repo commentary | qc_clustering/run_clustering.sh
#
# The sweep job: 16 CPUs, 512G, 48 hours in the rk_spatial_min environment.
#
# Both the memory and the walltime are genuinely needed here. This is the largest single
# allocation anywhere in the repo and it exists to settle embedding parameters once rather
# than re-litigating them.
# ========================================================================================


# 1) Initialize conda
source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh

# 2) Activate environment
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/rk_spatial_min

# 3) Harmless export
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4) Run the script using the environment's specific python

/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/rk_spatial_min/bin/python scanpy_xenium_workflow_v2.py