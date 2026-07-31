#!/bin/bash
#SBATCH --job-name=Novae_replot_smallpts
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/replot_nature_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/replot_nature_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/run_replot_smallpts.sh
#
# Wrapper for the small-point re-plot: 4 CPUs, 32G, one hour.
#
# Note it writes its logs to the same replot_nature_%j filenames as the other re-plot job, so
# check the job id rather than the filename when reading logs for one specific run.
# ========================================================================================


source /home/rodrigok/.bashrc
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate
export PYTHONNOUSERSITE=1

echo "Job started: $(date) on $(hostname)"
which python
python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/replot_nature_smallpts.py
rc=$?
echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
