#!/bin/bash
#SBATCH --job-name=DOTventral
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | deconvolution/run_DOTpy_ventral_horn.sh
#
# Ventral horn deconvolution: 16 CPUs, 256G, 12 hours in the dotpy environment. Half the
# memory and half the time of the full-section run, because the subset is much smaller.
# ========================================================================================


source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/dotpy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/dotpy/bin/python DOTpy_ventral_horn.py
