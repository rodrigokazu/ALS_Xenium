#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/1_novae/novae_INDEP_AUG5.py
#
# A thin driver around novae_INDEPENDENT_FINAL.py for the AUG5 h5ad set. It repoints
# final_config at SC_MNcorrected_AUG5_h5ads and an AUG5 output folder, switches the mark
# column to is_neuron, then calls the unchanged per-sample trainer.
#
# Nothing downstream reads its output. The only consumers are the three ps_*_AUG5 report
# scripts. The verdicts and the dual-pass QC read Novae_persample_INDEPENDENT_FINAL. The
# import order matters: final_config must be patched before novae_INDEPENDENT_FINAL imports,
# because that module captures its paths at import time.
# ========================================================================================

"""
novae_INDEP_AUG5.py -- run the INDEPENDENT per-sample Novae stage on the AUG5 h5ad set.

Thin driver, same shape as rebuild_aug5_concat.py: it repoints final_config at the AUG5
inputs and an AUG5-tagged output dir, then calls the unchanged, already-reviewed
novae_INDEPENDENT_FINAL machinery. Discovery order, the Delaunay graph on true microns, the
level scan for [4, 6, 8, 10] domains, the temp-write-then-rename, NUM_WORKERS=2 and every
gate are exactly as validated on 2026-07-25. Only the input set and where results land change.

    python novae_INDEP_AUG5.py <task_id>   -> one sample (SLURM array 0-19)
    python novae_INDEP_AUG5.py aggregate   -> stitch the per-sample summaries

Two deliberate deltas, both about the neuron call:

1. MN_KEY -> 'is_neuron'. The AUG5 build carries neuron_prob + is_neuron from Marcel's Aug-5
   neuron_calls and NO is_MN; the MN column was stripped on 2026-08-06 because synthesising an
   MN label from a provisional formula would put a fabricated column under the name a real
   annotation would use. Left on the default the stage would find no is_MN, CREATE an all-False
   one, and write it into 20 new h5ads -- reintroducing exactly that column by the back door.
   Pointing MN_KEY at the real annotation avoids that and makes the overlay meaningful.

2. MN_LABEL -> 'neuron', so no figure legend or title says MN about a neuron call.

The Jul-22 output dir Novae_persample_INDEPENDENT_FINAL/ is untouched; the dual-pass QC and
the overlays still derive from it.

CAVEAT to carry into anything built on this output: is_neuron has a white-matter false-positive
mode. The grey-matter fraction of non-MN neurons was 47-71% in SD01923_BI, SD01320_BG,
SD01616_BI, SD02622_BI and SD04219_BI versus 90-98% elsewhere, and the call rose in 19/20
samples versus Jul-22, which is a one-sided shift and so model re-inference rather than
segmentation jitter.
"""
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

# This build has no MN source by design; do not let the is_MN provenance gate halt anything.
os.environ.setdefault("MN_REQUIRE_FINAL_PROVENANCE", "0")

import final_config as cfg  # noqa: E402

BUILD_TAG = "AUG5"
BUILD_DATE = "20260806"

# --- repoint INPUTS to the AUG5 h5ads and OUTPUT to an AUG5-tagged dir -----------------
cfg.H5AD_DIR = cfg.SPATIAL_ROOT / f"SC_MNcorrected_{BUILD_TAG}_h5ads"
cfg.PERSAMPLE_SUFFIX = f"__MNcorrected_{BUILD_TAG}.h5ad"
cfg.COMBINED_H5AD = cfg.H5AD_DIR / (
    f"ALS_SCXenium_MNcorrected_{BUILD_TAG}_concatenated_offset_allsamples_preQC_{BUILD_DATE}.h5ad")
cfg.NOVAE_INDEP_DIR = cfg.SPATIAL_ROOT / f"Novae_persample_INDEPENDENT_{BUILD_TAG}"
cfg.SEG_PIPELINE = "mw_final_aug5"

# Import AFTER the repoint: the module snapshots cfg paths into its own globals at import time
# and creates its output subdirs there.
import novae_INDEPENDENT_FINAL as nv  # noqa: E402

# --- the two neuron-call deltas (see module docstring) --------------------------------
nv.MN_KEY = "is_neuron"
nv.MN_LABEL = "neuron"


def _banner(mode):
    print("=" * 72)
    print(f"NOVAE INDEPENDENT on {BUILD_TAG}  |  mode={mode}")
    print("=" * 72)
    print(f"  input dir   : {cfg.H5AD_DIR}")
    print(f"  per-sample  : <LABEL>{cfg.PERSAMPLE_SUFFIX}")
    print(f"  output dir  : {cfg.NOVAE_INDEP_DIR}")
    print(f"  mark column : obs['{nv.MN_KEY}']  (labelled '{nv.MN_LABEL}')")
    print(f"  targets     : {nv.TARGET_N_DOMAINS}  primary n{nv.PRIMARY_TARGET}")
    print("  is_MN       : ABSENT on this build by design; not synthesised here")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        sys.exit("usage: novae_INDEP_AUG5.py <task_id|aggregate>")
    # Guard the input set before burning a GPU slot: the stage silently falls back to
    # subsetting the combined h5ad when the per-sample count is not exactly 20, and that
    # fallback loading 1.85M cells per task is not what we want here.
    n_found = len(nv._persample_files())
    if n_found != nv.EXPECTED_SAMPLE_COUNT:
        sys.exit(f"FATAL expected {nv.EXPECTED_SAMPLE_COUNT} per-sample AUG5 h5ads in "
                 f"{cfg.H5AD_DIR}, found {n_found}. Refusing the combined-subset fallback.")
    if arg == "aggregate":
        _banner("aggregate")
        nv.aggregate()
    else:
        _banner(f"task {arg}")
        nv.process_one(int(arg))
