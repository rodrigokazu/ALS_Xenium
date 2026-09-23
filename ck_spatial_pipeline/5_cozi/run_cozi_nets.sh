#!/bin/bash
#SBATCH --job-name=cozi_nets
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/cozi_nets_%j.log
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/cozi_nets_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/run_cozi_nets.sh
#
# Wrapper for cozi_nets.py, 8 CPUs, 64G, 3 hours. --exclude-sections gives the noQC2
# sensitivity run.
# ========================================================================================
set -euo pipefail

# COZI result -> directed + undirected cell-type networks (whole cord + ventral horn),
# plus a COZI re-run on cell_type_v2 as a second granularity, plus the obs label audit.

RK=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK
PY=$RK/envs/cozi_env/bin/python
NC=$RK/Spatial/NicheCompass_SC
OBJ=$NC/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad
RUN_WHOLE=$NC/cozi_als/20260915_022205_ncv3
RUN_VH=$NC/cozi_als_vh/20260915_081701_ncv3
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$NC/cozi_nets/${STAMP}_ncv3
SCRIPT=/home/rodrigok/SLURM_jobs/Spatial/NicheDE/cozi_nets.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLBACKEND=Agg

echo "=== cozi_nets  $(date)  job ${SLURM_JOB_ID:-local}  host $(hostname) ==="
echo "OUT=$OUT"
for f in "$OBJ" "$RUN_WHOLE/tables/cozi_persection_celltype_wdr49.csv" \
         "$RUN_VH/tables/cozi_persection_celltype_wdr49.csv" "$SCRIPT"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 2; }
done
mkdir -p "$OUT/code"
cp "$SCRIPT" "$0" "$OUT/code/"
"$PY" -c "import cozipy, scipy, pandas, h5py; print('cozipy', cozipy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'h5py', h5py.__version__)"

cd /home/rodrigok/SLURM_jobs/Spatial/NicheDE   # never run from /tmp (stray struct.py shadows stdlib)
"$PY" "$SCRIPT" \
  --run whole="$RUN_WHOLE" \
  --run vh="$RUN_VH" \
  --label-set celltype_wdr49 \
  --obj "$OBJ" \
  --audit-cols cell_type,cell_type_mn,cell_type_v2 \
  --rerun-celltype-col cell_type_v2 \
  --n-neighbors 10 --n-permutations 300 --min-cell-count 10 --seed 1 \
  --outdir "$OUT"

echo
echo "=== output inventory ==="
find "$OUT" -type f | wc -l
du -sh "$OUT"
ls "$OUT"
echo "DONE $(date)"
echo "OUT=$OUT"
