#!/bin/bash
#SBATCH --job-name=DOTnonneuro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | deconvolution/run_DOTpy_nonneuronal.sh
#
# Deconvolution job: 16 CPUs, 512G, 24 hours, in the dedicated dotpy conda environment.
#
# The half-terabyte request is real. DOT holds the reference and the spatial matrix together
# and sweeps two HVG settings at 2,000 and 5,000 genes, narrowed from a wider sweep because
# eight cell types do not need more and the extra genes were adding noise.
# ========================================================================================


source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/dotpy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/dotpy/bin/python DOTpy_nonneuronal.py
