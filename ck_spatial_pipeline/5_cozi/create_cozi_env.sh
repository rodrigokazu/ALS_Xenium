#!/bin/bash
#SBATCH --job-name=create_cozi_env
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/create_cozi_env_%j.log
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/create_cozi_env_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/create_cozi_env.sh
#
# First attempt at the cozipy venv, built with --copies from the cell_annotator_env Python.
# That base env has to survive because libpython still resolves from it. finish_cozi_env.sh
# is the step that completed.
# ========================================================================================
set -euo pipefail

ENVS=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs
TARGET=$ENVS/cozi_env
BASE=$ENVS/cell_annotator_env/bin/python      # Python 3.11.15, verified standalone-runnable

echo "=== create_cozi_env  $(date) ==="
echo "base python: $($BASE -V 2>&1)  ->  $BASE"

if [ -e "$TARGET" ]; then
  echo "REFUSING: $TARGET already exists. Remove it deliberately if you want a rebuild."
  exit 1
fi

# venv with --copies so the interpreter is copied, not symlinked
"$BASE" -m venv --copies "$TARGET"
echo "venv created at $TARGET"

# shellcheck disable=SC1091
source "$TARGET/bin/activate"
python -V
python -m pip install --upgrade pip wheel setuptools

# COZI itself is tiny (numpy/scipy/pandas). scanpy+anndata+h5py so the env can
# read our h5ads directly; matplotlib/seaborn for the dot plots FB_SPI draws.
python -m pip install \
  "cozipy==0.1.4" \
  "scanpy" "anndata" "h5py" \
  "numpy" "scipy" "pandas" \
  "matplotlib" "seaborn" \
  "numba"

echo
echo "=== versions ==="
python - <<'PY'
import cozipy, numpy, scipy, pandas, scanpy, anndata, h5py
print("cozipy    ", cozipy.__version__)
print("numpy     ", numpy.__version__)
print("scipy     ", scipy.__version__)
print("pandas    ", pandas.__version__)
print("scanpy    ", scanpy.__version__)
print("anndata   ", anndata.__version__)
print("h5py      ", h5py.__version__)
print("exports   ", cozipy.__all__)
PY

echo
echo "=== SMOKE TEST: two planted cell types that sit together ==="
python - <<'PY'
import numpy as np
from cozipy import run_cozi
rng = np.random.default_rng(0)
# 3 blobs. A and B share blob 0 (should show mutual preference); C sits alone in blob 1.
n = 400
A = rng.normal([0, 0], 0.6, size=(n, 2))
B = rng.normal([0, 0], 0.6, size=(n, 2))
C = rng.normal([8, 8], 0.6, size=(n, 2))
coords = np.vstack([A, B, C])
labels = np.array(["A"] * n + ["B"] * n + ["C"] * n)
df = run_cozi(coords, labels, nbh_def="knn", n_neighbors=10,
              n_permutations=300, random_state=1, normalize_zscore=True)
print(df.to_string(index=False))
ab = df[(df.index_cell_type == "A") & (df.neighbor_cell_type == "B")]
ac = df[(df.index_cell_type == "A") & (df.neighbor_cell_type == "C")]
assert float(ab.zscore.iloc[0]) > 0,  "A->B should be POSITIVE (they are co-located)"
assert float(ac.zscore.iloc[0]) < float(ab.zscore.iloc[0]), "A->C must be weaker than A->B"
print("\nSMOKE TEST PASSED: A->B z=%.2f (co-located) > A->C z=%.2f (separated)"
      % (float(ab.zscore.iloc[0]), float(ac.zscore.iloc[0])))
print("cond_ratio A->B = %.3f" % float(ab.cond_ratio.iloc[0]))
PY

echo
echo "=== freeze ==="
python -m pip freeze > "$ENVS/cozi_env_freeze.txt"
wc -l "$ENVS/cozi_env_freeze.txt"
echo "DONE $(date)"
