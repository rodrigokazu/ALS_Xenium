#!/bin/bash
#SBATCH --job-name=Novae_downstream
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/novae_dwn_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/novae_dwn_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/run_novae_downstream.sh
#
# Wrapper for the domain characterisation: 16 CPUs, 256G, 12 hours, CPU only.
#
# No GPU needed. Novae has already produced the domains at this stage and everything here is
# scanpy and seaborn over a large object, so the memory and not the accelerator is what
# matters.
#
# Prints the novae, scanpy and anndata versions before starting. That is how the DotPlot
# behaviour above got pinned to scanpy 1.9.8.
# ========================================================================================

# NO GPU: this is post-hoc analysis of an already-trained Novae model (batch correction,
# UMAP, DEG, plotting). batch_effect_correction / compute on 1.69M cells is CPU/RAM-bound;
# 256G holds the full object + a normalized .copy() for the DEG block. cpus-per-task=16
# speeds the scanpy neighbours/UMAP and wilcoxon DEG.

source ~/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++ new
# enough for the compiled pandas/scanpy extensions (the CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv (a broken ~/.local anndata
# has shadowed envs before on this account).
export PYTHONNOUSERSITE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== which python ==="
which python
python -c "import novae, scanpy, anndata; print('novae', novae.__version__, '| scanpy', scanpy.__version__, '| anndata', anndata.__version__)"

python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/novae_downstream.py
rc=$?

echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
