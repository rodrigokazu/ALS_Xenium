#!/bin/bash
#SBATCH --job-name=VHv2_attach
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VHv2/logs/%x_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VHv2/logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/motor_neurons/attach_vhv2.sbatch
#
# Wrapper for attach_vhv2.py, 4 CPUs, 96G, 6 hours, nichecompass_env.
# ========================================================================================

set -eo pipefail
WORKDIR=/home/rodrigok/SLURM_jobs/Spatial/VHv2
PYBIN=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python
mkdir -p "${WORKDIR}/logs"
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "${WORKDIR}"

echo "== VHv2_attach | job ${SLURM_JOB_ID:-NA} | node $(hostname) | $(date) =="
"${PYBIN}" "${WORKDIR}/attach_vhv2.py"
echo "== DONE $(date) =="
