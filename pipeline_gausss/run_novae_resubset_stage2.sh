#!/bin/bash
#SBATCH --job-name=VH_novae_stage2
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage2_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/vh_stage2_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/run_novae_resubset_stage2.sh
#
# GPU job for the sub-domain model: one qrtx4000, 8 CPUs, 128G, 12 hours. Took about 12
# minutes on 537k cells.
#
# Chained afterok from stage one and feeds stage three the same way.
# ========================================================================================

# STAGE 2 (GPU): train a FRESH Novae model on the D1014+D1018 subset to find finer
# sub-domains. The 3 qrtx4000 GPUs are typically free (a100/h200 are booked/queued);
# Novae streams subgraph mini-batches so 8GB VRAM is sufficient. If CUDA OOM shows in
# .err, resubmit with a smaller Novae batch_size.
#
# Scripts live in /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/ (this .sh is copied there
# alongside novae_resubset_stage2.py and the shared nature_style.py).

source ~/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++ new
# enough for pandas/scanpy/torch compiled extensions (the CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv.
export PYTHONNOUSERSITE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== GPU ==="
nvidia-smi || echo "nvidia-smi unavailable"
echo "=== which python ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "import novae, scanpy, anndata; print('novae', novae.__version__, '| scanpy', scanpy.__version__, '| anndata', anndata.__version__)"

# run from the script dir so 'import nature_style' resolves (cwd is on sys.path).
cd /home/rodrigok/SLURM_jobs/Spatial/VH_isolation
python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/novae_resubset_stage2.py
rc=$?

echo "finished"
echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
