#!/bin/bash
#SBATCH --job-name=VH_subset_stage1
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage1_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage1_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/run_subset_stage1.sh
#
# Subsetting job: 8 CPUs, 128G, 2 hours. Ran in about 90 seconds.
#
# Chained into stage two with --dependency=afterok. Cheap enough to re-run freely while
# deciding which domains to keep.
# ========================================================================================

# STAGE 1 (CPU): load the 1.69M-cell joint Novae object, subset to domains D1014+D1018,
# write the combined + per-sample lean subsets, and per-sample Nature-tier spatial plots.
# No GPU, no model training -> CPU/RAM only. 128G holds the full object + the subset .copy().
#
# Scripts live in /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/ (this .sh is copied there
# alongside subset_domains_stage1.py and the shared nature_style.py).

source ~/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++ new
# enough for the compiled scanpy/pandas extensions (the CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv.
export PYTHONNOUSERSITE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== which python ==="
which python
python -c "import scanpy, anndata, matplotlib; print('scanpy', scanpy.__version__, '| anndata', anndata.__version__, '| matplotlib', matplotlib.__version__)"

# run from the script dir so 'import nature_style' resolves (cwd is on sys.path).
cd /home/rodrigok/SLURM_jobs/Spatial/VH_isolation
python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/subset_domains_stage1.py
rc=$?

echo "finished"
echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
