#!/bin/bash
#SBATCH --job-name=VH_downstream_stage3
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage3_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage3_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/run_downstream_subdomains_stage3.sh
#
# Final stage of the chain: 16 CPUs, 256G, 12 hours, CPU only. Ran in about 70 seconds.
#
# The generous allocation reflects the memory profile of the seaborn clustermaps and not the
# runtime. Chained afterok from stage two.
# ========================================================================================

# STAGE 3 (CPU): downstream characterisation of the Stage-2 Novae sub-domains (DEG,
# pathway scores, composition, clustermaps, per-sample facets). No GPU: post-hoc analysis
# of the already-trained sub-domain object. 256G holds the subset object + a normalized
# .copy() for the wilcoxon DEG; cpus-per-task=16 speeds neighbours/DEG.
#
# Scripts live in /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/ (this .sh is copied there
# alongside downstream_subdomains_stage3.py and the shared nature_style.py).

source ~/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++ new
# enough for the compiled scanpy/pandas/seaborn extensions (the CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv.
export PYTHONNOUSERSITE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== which python ==="
which python
python -c "import scanpy, anndata, seaborn, matplotlib; print('scanpy', scanpy.__version__, '| anndata', anndata.__version__, '| seaborn', seaborn.__version__, '| matplotlib', matplotlib.__version__)"

# run from the script dir so 'import nature_style' resolves (cwd is on sys.path).
cd /home/rodrigok/SLURM_jobs/Spatial/VH_isolation
python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/downstream_subdomains_stage3.py
rc=$?

echo "finished"
echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
