#!/bin/bash
#SBATCH --job-name=finish_cozi_env
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/finish_cozi_env_%j.log
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/finish_cozi_env_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/finish_cozi_env.sh
#
# Installs cozipy 0.1.4 and the scientific stack with --only-binary=:all: because the nodes
# carry GCC 4.8.5. It then runs a planted three-label smoke test and writes the freeze.
# ========================================================================================
set -euo pipefail
ENVS=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs
T=$ENVS/cozi_env
cd "$ENVS"                      # never run from /tmp: a stray /tmp/struct.py shadows stdlib

echo "=== finish_cozi_env $(date) ==="
"$T/bin/python" -V
# --only-binary=:all: is REQUIRED. Without it pip falls back to the numpy sdist and the
# compute nodes only have GCC 4.8.5, while numpy needs >= 9.3 (max gcc module here is 9.2.0).
"$T/bin/python" -m pip install --upgrade --quiet pip wheel setuptools
"$T/bin/python" -m pip install --only-binary=:all: \
  "cozipy==0.1.4" numpy scipy pandas h5py anndata scanpy matplotlib seaborn

echo; echo "=== versions ==="
"$T/bin/python" - <<'PY'
import cozipy, numpy, scipy, pandas, scanpy, anndata, h5py
print("cozipy  ", cozipy.__version__)
print("numpy   ", numpy.__version__)
print("scipy   ", scipy.__version__)
print("pandas  ", pandas.__version__)
print("scanpy  ", scanpy.__version__)
print("h5py    ", h5py.__version__)
print("exports ", cozipy.__all__)
PY

echo; echo "=== SMOKE TEST ==="
"$T/bin/python" - <<'PY'
import numpy as np
from cozipy import run_cozi
rng = np.random.default_rng(0)
n = 400
A = rng.normal([0, 0], 0.6, size=(n, 2))
B = rng.normal([0, 0], 0.6, size=(n, 2))   # co-located with A
C = rng.normal([8, 8], 0.6, size=(n, 2))   # far away
coords = np.vstack([A, B, C])
labels = np.array(["A"]*n + ["B"]*n + ["C"]*n)
df = run_cozi(coords, labels, nbh_def="knn", n_neighbors=10,
              n_permutations=300, random_state=1, normalize_zscore=True)
print(df.to_string(index=False))
ab = float(df[(df.index_cell_type=="A") & (df.neighbor_cell_type=="B")].zscore.iloc[0])
ac = float(df[(df.index_cell_type=="A") & (df.neighbor_cell_type=="C")].zscore.iloc[0])
cr = float(df[(df.index_cell_type=="A") & (df.neighbor_cell_type=="B")].cond_ratio.iloc[0])
assert ab > 0,   "A->B must be positive (planted co-location)"
assert ab > ac,  "A->B must exceed A->C (C is spatially separated)"
print(f"\nSMOKE TEST PASSED: A->B z={ab:.2f} > A->C z={ac:.2f}; cond_ratio(A->B)={cr:.3f}")
PY

echo; echo "=== freeze ==="
"$T/bin/python" -m pip freeze > "$ENVS/cozi_env_freeze.txt"
wc -l "$ENVS/cozi_env_freeze.txt"

cat > "$T/README.md" <<'MD'
# cozi_env

Python venv for COZI neighbour-preference analysis (the method the FB_SPI /
Burnside spinal-cord-injury manuscript uses for directional neighbour preference).

- **Package:** `cozipy` 0.1.4 (PyPI), MIT, Schapiro Lab.
  Repo <https://github.com/SchapiroLabor/COZIpy>, paper Schiller et al. 2026,
  Nat Commun, doi:10.1038/s41467-026-71699-z
- **Python 3.11.15**, seeded with `--copies` from `../cell_annotator_env/bin/python`.
  That base env must keep existing: the copied interpreter still resolves
  `libpython3.11.so` from it. (`nichecompass_env`'s python is broken for exactly
  this reason, so it was not used as the base.)
- **Install rule:** always `pip install --only-binary=:all:`. The compute nodes have
  GCC 4.8.5 and the newest gcc module is 9.2.0, but numpy needs >= 9.3, so any sdist
  fallback fails at build time.

## API
```python
from cozipy import run_cozi
df = run_cozi(coords, labels, nbh_def="knn",   # or "radius" / "delaunay"
              n_neighbors=10, n_permutations=300,
              random_state=1, normalize_zscore=True, min_cell_count=0)
```
Returns a DataFrame: `index_cell_type, neighbor_cell_type, n_index_cells,
n_neighbor_cells, n_index_cells_with_neighbor, n_index_neighbor_edges, zscore,
cond_ratio`.

`zscore` is the conditional z-score (directional: A->B is not B->A).
`cond_ratio` is the fraction of A cells that actually neighbour a B cell, which
separates local enrichment from tissue-wide infiltration.

## Matching the FB_SPI manuscript
They used cozipy **0.1.1** with `n_neighbors=10`, `n_permutations=300`, run per
slide, then divided z by sqrt(n cells per slide) by hand. In 0.1.4 that division is
built in as `normalize_zscore=True`. Two other differences to know:
- 0.1.3 changed the return type from dict to DataFrame, so code written against
  0.1.1 will not run unchanged here.
- 0.1.1 -> 0.1.4 are additive (extra count columns, a z normalisation option, a
  cell-count filter, an overflow fix). The z-score maths itself did not change,
  so results are comparable.
MD
echo "wrote $T/README.md"
echo "DONE $(date)"
