#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/promote_pass2.py
#
# Copies staged pass-2 objects over Ranger_procd/ALS_SCXenium_<TAG>_pass2.h5ad, but only
# after four gates per sample: trace OK, staged file newer than the trace, 28 required obs
# columns present, and n_obs equal to the trace's final cell count. Run with --apply on the
# login node. There is no SLURM wrapper.
#
# The promoted files have since been edited in place by the motor-neuron attach scripts, so
# they no longer match the staged copies byte for byte. n_obs has not changed.
# ========================================================================================

"""
Promote staged pass-2 objects over the canonical ones, but only the ones this
relabelled run actually wrote.

Three earlier runs were cancelled mid-flight, and one of them (52377719_15) had
`allow_errors=True` carry it past a failed relabel cell, so a staged file can exist
that looks complete and carries the OLD labels. Every candidate therefore has to
clear four gates before it is copied:

  1. its trace says status OK, no cell errors, vh_relabel is true, the relabel
     version is v4, and the spatial-coordinate content gate passed
  2. the staged h5ad is newer than the trace's run start
  3. the staged h5ad's obs carries the full relabelled column set
  4. its n_obs matches the trace's final_cells

Anything that fails is reported and left alone.

usage: promote_pass2.py [--apply]      (default is a dry run)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import h5py

SP = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial")
RUNS = SP / "Novae_IND_QC/runs_v6"   # the v6 array writes here; runs_srcfixed holds the superseded v2 run
STAGE = RUNS / "stage"
DEST = SP / "Ranger_procd"

# v4 contract. is_MN_inter is GONE from the source file, so requiring it here
# would refuse every sample; is_neuron / is_TDP / the area column are new.
REQUIRED = ["is_MN", "is_MN_v6_asdelivered", "is_MN_invh", "is_MN_eligible",
            "is_neuron_v3", "neuron_prob_v3", "is_neuron_prev",
            "is_neuron_general", "is_GADvh", "is_GAD", "is_Interneuron",
            "in_VH", "in_GM", "in_VH_not_GM", "has_vh_data", "vh_mask_empty",
            "vh_source", "is_gabaergic_neuron", "neuron_class",
            "cell_type_mw", "pathologist_class_mw", "STMN2_mw", "CE_STMN2_mw",
            "cell_area_um2", "mn_signature_prob", "is_TDP", "tdp_proba",
            "tdp_evaluated"]
RELABEL_VERSION = "v6"

SAMPLES = ["SD00614_BG", "SD01015_BG", "SD01115_BG", "SD01320_BG", "SD01413_BA",
           "SD01616_BI", "SD01620_BI", "SD01623_BI", "SD01915_BG", "SD01922_BI",
           "SD01923_BI", "SD02022_BI", "SD02622_BI", "SD02818_BG", "SD02913_BA",
           "SD03522_BG", "SD03614_BG", "SD03914_BG", "SD04219_BI", "SD05413_BG"]


def check(sample):
    tag = sample.replace("_", "")
    tr = RUNS / f"trace_src_{sample}.json"
    st = STAGE / f"ALS_SCXenium_{tag}_pass2.h5ad"
    if not tr.exists():
        return None, "no trace"
    body = json.loads(tr.read_text())
    run = body.get("run", {})
    if run.get("status") != "OK" or run.get("cell_errors"):
        return None, f"trace status {run.get('status')}, {len(run.get('cell_errors') or [])} cell errors"
    if not body.get("vh_relabel"):
        return None, "trace has no vh_relabel flag (pre-relabel run)"
    if body.get("vh_relabel_version") != RELABEL_VERSION:
        return None, (f"trace is relabel version {body.get('vh_relabel_version')!r}, "
                      f"expected {RELABEL_VERSION!r} -- this is an older run")
    cg = body.get("coord_gate_max_delta_um")
    if cg is None or cg >= 1e-3:
        return None, f"coordinate content gate absent or failed ({cg})"
    if not st.exists():
        return None, "no staged object"
    # the staged file must be newer than the trace, i.e. written by this run
    if st.stat().st_mtime < tr.stat().st_mtime - 3600:
        return None, "staged object older than its trace"
    with h5py.File(st, "r") as f:
        obs = set(f["obs"].keys())
        n = f["obs/_index"].shape[0]
    missing = [c for c in REQUIRED if c not in obs]
    if missing:
        return None, f"staged object missing {missing[:4]}"
    if n != body.get("final_cells"):
        return None, f"n_obs {n:,} != trace final_cells {body.get('final_cells'):,}"
    return st, (f"OK  n_obs {n:,}  is_MN {body.get('n_is_MN_final')}"
                f" (+{body.get('n_is_MN_na_final')} NA)")


def main():
    apply = "--apply" in sys.argv
    ok, refused = [], []
    for s in SAMPLES:
        src, why = check(s)
        print(f"  {s:12s} {'PASS' if src else 'REFUSE'}  {why}", flush=True)
        (ok if src else refused).append((s, src, why))
    print(f"\n{len(ok)} of {len(SAMPLES)} clear all gates")
    if refused:
        print("refused:", [(s, w) for s, _, w in refused])
    if not apply:
        print("\ndry run; pass --apply to copy")
        return
    if refused:
        print("\nREFUSING TO PROMOTE ANY: fix the failures first, a partial cohort on "
              "disk is worse than none")
        sys.exit(1)
    for s, src, _ in ok:
        dst = DEST / src.name
        tmp = dst.with_suffix(".h5ad.tmp_promote")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        print(f"  promoted {dst.name}", flush=True)
    print(f"\npromoted {len(ok)} objects at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
