#!/bin/bash
#SBATCH --job-name=ps_sweep_agg
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report/logs/ps_sweep_agg_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report/logs/ps_sweep_agg_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_sweep_aggregate.sh
#
# Small roll-up job running the cohort report and then the audio edition: 2 CPUs, 16G, 15
# minutes.
#
# Depends on the sweep array. Runs the two steps in sequence because the audio narrative reads
# the same stats the cohort pages do.
# ========================================================================================


echo "host=$(hostname) start=$(date)"
source /home/rodrigok/.bashrc
module load python/3.11.1 gcc/13.3.0
set -eo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report
python ps_sweep_cohort.py
python ps_sweep_audio.py
echo "done end=$(date)"
