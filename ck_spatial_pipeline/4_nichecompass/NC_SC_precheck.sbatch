#!/bin/bash

#SBATCH --job-name=NC_SC_precheck                                                  # Job name (%x in log paths)
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso/logs/%x_%j.out    # STDOUT
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso/logs/%x_%j.err     # STDERR
#SBATCH --partition=batch                                                          # SCG batch partition
#SBATCH --cpus-per-task=4                                                          # CPU only, light job
#SBATCH --mem=64G                                                                  # GP dicts + one adata var check
#SBATCH --time=04:00:00                                                            # staging downloads + GP survival gate
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_SC_precheck.sbatch
#
# Wrapper for the gene-programme precheck. Job 52414419.
# ========================================================================================

# ------------------------------------------------------------------------------ #
# NicheCompass ALS spinal-cord Xenium — GP resource staging + GP-survival gate.   #
# Stages MEBOCOST TSVs / NicheNet CSVs / OmniPath onto oak, then checks how many  #
# gene programs survive the min-genes filters on the 480-gene Xenium panel.       #
# Exits non-zero on FAIL, which blocks the afterok dependency chain.              #
# Ported from Sanger LSF (nichecompass 0.2.3) to SCG SLURM (nichecompass 0.3.3).  #
# ------------------------------------------------------------------------------ #

set -eo pipefail

WORKDIR=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso
PYBIN=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python

# 1) Ensure the log dir exists (SLURM opens %x_%j files at job start; keep for reruns)
mkdir -p "${WORKDIR}/logs"

# 2) Environment init — the venv rides on the python/3.11.1 module; gcc/13.3.0 pairs with it
module load python/3.11.1 gcc/13.3.0

# 3) Harmless exports (no-op on CPU; OMP matches the allocation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# 4) Run from the pipeline dir, never /tmp (stray /tmp/struct.py shadows stdlib struct;
#    NicheNet extraction writes temp .rds files to the CWD, which must be writable)
cd "${WORKDIR}"

echo "== NC_SC_precheck | job ${SLURM_JOB_ID:-NA} | node $(hostname) | $(date) =="

# 5) Stage GP resources (MEBOCOST enzyme/sensor TSVs -> oak gene_programs/; verified URLs)
bash stage_gp_resources.sh

# 6) GP-survival precheck — builds the OmniPath/NicheNet/MEBOCOST GP dicts, applies the
#    Sanger min-genes filters (2/1/1) against the 480-gene panel, prints the verdict,
#    exits non-zero if too few GPs survive
"${PYBIN}" NC_SC_gp_precheck.py

echo "== NC_SC_precheck done | $(date) =="
