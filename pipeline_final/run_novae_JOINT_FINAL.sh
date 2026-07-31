#!/bin/bash
#SBATCH --job-name=Novae_JOINT_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_novae_JOINT_FINAL.sh
#
# GPU wrapper for the joint Novae fit. One qrtx4000, 8 CPUs, 128G, 12 hours.
#
# The small card is a deliberate choice rather than a compromise. Novae streams subgraph
# mini-batches, so 8GB is enough even for 1.8M cells, and on SCG the a100 queue has run to
# roughly two weeks while the three qrtx4000 nodes are often free and start immediately. The
# h200 partitions gpu_normal and gpu_long are open to mpsnyder but were fully booked.
#
# It prints torch, CUDA availability and the novae/scanpy/anndata versions before starting
# work. Costs nothing, and it has explained more than one strange result after the fact.
#
# Known cosmetic failure: novae.plot.loss_curve has a signature that does not match how it is
# called, so training_loss.png is never written and convergence is not visually confirmed.
# Early stopping is patience 3 / min_delta 0.1 within 50 epochs. Fix the call if you need the
# curve.
# ========================================================================================

# ONE joint Novae model over ALL 20 slides of the combined _FINAL concat (~1.8M cells).
# qrtx4000 (8GB): Novae streams subgraph mini-batches so 8GB VRAM suffices; a100/h200 are
# booked/queued. If CUDA OOM appears in .err, resubmit with a smaller Novae batch_size.
# cpus-per-task=8 matches NUM_WORKERS=8 (Novae starves the GPU with 0 DataLoader workers).
#
# *** STAGING ONLY -- DO NOT SUBMIT until the combined _FINAL h5ad exists AND (for real MN
#     positioning) Marcel's _final MN annotation is joined into it. is_MN degrades gracefully
#     (all-False -> placeholder MN outputs) so the niche map itself is valid even before then. ***

source /home/rodrigok/.bashrc

# novae_env is a venv on the python/3.11.1 module; gcc/13.3.0 supplies a libstdc++ new
# enough for the compiled pandas/scanpy/torch extensions (the CentOS-7 system one is too old).
module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

# Prevent ~/.local user-site packages from shadowing the venv; don't litter .pyc on OAK.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# Route all caches to node-local /tmp (HOME quota is full; keep OAK clean).
export TMPDIR="${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
export NUMBA_CACHE_DIR="$TMPDIR/numba"
export MPLCONFIGDIR="$TMPDIR/mpl"
export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=== GPU ==="
nvidia-smi || echo "nvidia-smi unavailable"
echo "=== which python ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "import novae, scanpy, anndata; print('novae', novae.__version__, '| scanpy', scanpy.__version__, '| anndata', anndata.__version__)"

# Run from the staged code dir so 'import final_config' / 'import mn_annotation_join' resolve
# (the .py also inserts this dir onto sys.path as belt-and-braces).
cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
python /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/novae_JOINT_FINAL.py
rc=$?

echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
