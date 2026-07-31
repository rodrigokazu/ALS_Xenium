#!/bin/bash
#SBATCH --job-name=Novae_persample_niches
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/novae_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/logs/novae_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
# ========================================================================================
# ALS_Xenium repo commentary | superseded/novae_v1/run_novae_persample_niches.sh
#
# SLURM wrapper for the first Novae fit, kept alongside the script it launched.
#
# Historical value is in the resource story. The original submission was killed at 24 minutes
# because a single-worker dataloader left the GPU idle; adding eight workers brought the same
# job home in about three hours fourteen on a qrtx4000. Every later Novae runner in this repo
# inherits that setting.
# ========================================================================================

# NOTE: a100 (batch) is ~2 weeks queued and h200 (gpu_normal/long) is fully booked;
# the 3 qrtx4000 GPUs are free now. Novae streams subgraph mini-batches so 8GB VRAM
# is sufficient. If CUDA OOM appears in .err, resubmit with a smaller Novae batch_size.

source /home/rodrigok/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++
# new enough for pandas' compiled extension (CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv
export PYTHONNOUSERSITE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== GPU ==="
nvidia-smi || echo "nvidia-smi unavailable"
echo "=== which python ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/novae_persample_niches.py
rc=$?

echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
