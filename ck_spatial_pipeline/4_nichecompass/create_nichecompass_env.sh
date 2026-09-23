#!/bin/bash
#SBATCH --job-name=create_nichecompass_env
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/create_nichecompass_env_%j.log
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/create_nichecompass_env.sh
#
# Builds the NicheCompass 0.3.3 venv on SCG (CentOS 7.9, glibc 2.17). Pins live in
# nichecompass_constraints.txt. The venv needs module python/3.11.1 at runtime for
# libpython, so every wrapper loads it together with gcc/13.3.0 and sets PYTHONNOUSERSITE=1.
# ========================================================================================
#
# NicheCompass 0.3.3 env for SCG (CentOS 7.9, glibc 2.17).  Attempt 4.
#
# TWO FAILED ATTEMPTS, ONE ROOT CAUSE: gcc 13.3.0 on the batch nodes dies with
# `internal compiler error: Illegal instruction` (at libstdc++ limits:1753) on ANY C++
# extension build. So no source build can ever succeed here -- the only safe policy is to
# refuse them outright.
#   * job 52380368 (83 s): `numpy<2.5` -> numpy 2.4.6, manylinux_2_28 only -> sdist -> ICE.
#   * job 52380382 (7 min): fixed numpy, but PIP_ONLY_BINARY listed package NAMES and
#     missed `contourpy` (a matplotlib dep) -> contourpy 1.3.3 is manylinux_2_28 -> ICE.
# Enumerating names is unwinnable. **PIP_ONLY_BINARY=:all: is the fix**: pip then treats an
# incompatible wheel as "no candidate" and BACKTRACKS to a version that has a glibc-2.17
# wheel (contourpy 1.3.2), instead of reaching for a compiler.
#
#   * job 52380389 (2:48): no compiler reached -- but ResolutionImpossible, because `docrep`
#     (a decoupler dep) is **sdist-only at every version** and :all: excluded it. Note that
#     `PIP_NO_BINARY=docrep` does NOT rescue this: when pip processes only_binary=:all: it
#     calls `other.clear()`, wiping the no-binary list. docrep is therefore installed FIRST,
#     before :all: is exported -- it is pure python, so no compiler is involved.
#
# The whole 148-package closure was then resolved with source allowed and every resolved
# version audited against PyPI for a cp311/glibc-2.17 wheel. Exactly 3 were unusable:
# docrep (sdist-only, handled above), greenlet 3.5.5 and tornado 6.5.8 (C extensions whose
# recent wheels are manylinux_2_28) -> capped at greenlet 3.2.5 / tornado 6.5.4 below.
#
# Steps are ordered cheapest-first so a resolution failure costs ~1 min, not the 4 min it
# takes to pull the 2.5 GB torch wheel. (PIP_CACHE_DIR is node-local scratch, so a retry on
# a different node re-downloads.)
#
# glibc-2.17 wheel ceilings, measured against PyPI on 2026-08-18 (newest cp311 x86_64 wheel
# whose manylinux tag is <= 2.17); novae_env's working versions in brackets:
#   numpy 2.2.6 [2.2.6]   pandas 2.3.2 [2.3.3]   scipy 1.16.3 [1.14.1]
#   scikit-learn 1.7.2 [1.8.0]   h5py 3.14.0 [3.14.0]
#   numba 0.67.0, llvmlite 0.49.0, matplotlib 3.11.1, pyreadr 0.5.6 -> latest is fine
#
# Other verified constraints:
#   * torch 2.6.0+cu124 is the ONLY viable torch: data.pyg.org has cp311 x86_64
#     torch-sparse 0.6.18 + torch-scatter 2.1.2 for it, and the torch-2.7.0 index ships
#     aarch64 only. torch_sparse is a HARD top-level import in nichecompass/data/utils.py
#     and nn/aggregators.py, and it pulls torch_scatter.
#   * Both extension wheels were installed into a throwaway --target dir and imported
#     against novae_env's torch on this cluster: SparseTensor round-trip OK, coexists with
#     PyG 2.7.0. So the ABI works on glibc 2.17 despite the bare linux_x86_64 tag.
#   * mlflow-skinny, not mlflow: skinny has no pyarrow dependency and nichecompass only
#     calls mlflow.log_param / log_metric / log_figure. (Full mlflow WOULD work if you ever
#     need it -- pyarrow<=20.0.0 has a glibc-2.17 wheel -- but skinny is smaller and enough.)
#     Because nichecompass's metadata asks for plain `mlflow`, it installs with --no-deps
#     and its real deps go in by hand.
#   * squidpy and scib-metrics are LEFT OUT: both are imported only by
#     nichecompass/benchmarking/*, which the top-level __init__ does not import (it imports
#     data, models, modules, nn, train, utils). squidpy drags in spatialdata, scib-metrics
#     drags in jax -- both CentOS-7 minefields. Add them later only if you benchmark.
#   * decoupler>=2 required: utils/gene_programs.py calls the 2.x API `dc.op.collectri`.
#   * No cuda module is loaded (unlike create_novae_env.sh): the torch wheels bundle their
#     own CUDA runtime, and the cuda module sets HDF5_DIR to system HDF5 1.8.12, which is
#     the documented cause of h5py source-building here.

set +u; source ~/.bashrc; set -u
set -eo pipefail

module load python/3.11.1 gcc/13.3.0     # BOTH also needed at RUNTIME (libpython + CXXABI)

PREFIX=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env
CONSTRAINTS=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_constraints.txt

export PIP_CACHE_DIR=/local/scratch/rodrigok/.pip_nc_cache   # never on oak: inode quota
mkdir -p "${PIP_CACHE_DIR}"

# PIP_ONLY_BINARY is exported LATER, after docrep is in (see header). Do not move it up.

cat > "${CONSTRAINTS}" <<'EOF'
numpy<=2.2.6
pandas<=2.3.2
scipy<=1.16.3
scikit-learn<=1.7.2
h5py<=3.14.0
pyarrow<=20.0.0
contourpy<=1.3.2
pyzmq<=26.4.0
pillow<=12.2.0
scikit-image<=0.25.2
leidenalg<=0.10.0
greenlet<=3.2.5
tornado<=6.5.4
EOF
echo "=== constraints ==="; cat "${CONSTRAINTS}"

echo "=== recreating ${PREFIX} ==="
rm -rf "${PREFIX}"
python -m venv "${PREFIX}"
source "${PREFIX}/bin/activate"
python -m pip install -U pip setuptools wheel

echo "=== 0/6 docrep from sdist (pure python; sdist-only, so it must precede :all:) ==="
pip install docrep==0.3.2
python -c "import docrep; print('docrep OK')"

# From here on: refuse EVERY source build. An incompatible wheel becomes 'no candidate' and
# pip backtracks to a version that has a glibc-2.17 wheel, instead of reaching for gcc.
export PIP_ONLY_BINARY=:all:

echo "=== 1/6 compiled scientific core (wheels only, at the glibc-2.17 ceiling) ==="
pip install -c "${CONSTRAINTS}" --only-binary :all: \
    numpy==2.2.6 pandas==2.3.2 scipy==1.16.3 scikit-learn==1.7.2 h5py==3.14.0

echo "=== 2/6 light stack FIRST (fails in ~1 min if any wheel is missing) ==="
pip install -c "${CONSTRAINTS}" \
    scanpy "decoupler>=2" omnipath pyreadr networkx seaborn mlflow-skinny ipykernel

echo "=== 3/6 torch 2.6.0+cu124 (--extra-index-url, NOT --index-url) ==="
pip install -c "${CONSTRAINTS}" torch==2.6.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

echo "=== 4/6 PyG + the two C++ extensions ==="
pip install -c "${CONSTRAINTS}" torch-geometric
pip install -c "${CONSTRAINTS}" -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
    torch-scatter==2.1.2 torch-sparse==0.6.18

echo "=== 5/6 nichecompass --no-deps ==="
pip install -c "${CONSTRAINTS}" nichecompass==0.3.3 --no-deps

echo "=== 6/6 verify ==="
python - <<'PY'
import torch, torch_geometric, torch_sparse, torch_scatter, mlflow, numpy, scanpy
import nichecompass
from nichecompass.models import NicheCompass
from nichecompass.utils import (extract_gp_dict_from_omnipath_lr_interactions,
                                extract_gp_dict_from_collectri_tf_network)
print("nichecompass", nichecompass.__version__)
print("torch", torch.__version__, "| PyG", torch_geometric.__version__,
      "| sparse", torch_sparse.__version__, "| scatter", torch_scatter.__version__)
print("numpy", numpy.__version__, "| scanpy", scanpy.__version__)
print("cuda available:", torch.cuda.is_available())
from torch_sparse import SparseTensor
import scipy.sparse as sp
m = sp.random(40, 40, density=0.1, format="coo", dtype="float32")
st = SparseTensor.from_edge_index(torch.LongTensor(numpy.vstack([m.row, m.col])),
                                  torch.FloatTensor(m.data), (40, 40))
print("SparseTensor round-trip OK, nnz", st.nnz())
print("NicheCompass + GP builders import OK")
PY

python -m ipykernel install --user --name nichecompass \
    --display-name "NicheCompass (Python 3.11 + CUDA)"
echo "DONE_RC=0 $(date)"
