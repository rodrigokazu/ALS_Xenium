#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/run_dualpass_src.py
#
# Executes one patched notebook top to bottom through nbclient on the curio_spatial kernel.
# It injects only a headless matplotlib preamble and applies no analysis patches of its own.
# It refuses a notebook that lacks the SOURCE FIX marker. SD03522_BG exists on SCG only as
# the -Copy1 notebook and the runner falls back to it.
# ========================================================================================

"""
Execute one per-sample dual-pass QC notebook that has ALREADY been corrected at
source by fix_notebooks_at_source.py.

Unlike run_dualpass.py this applies no analysis patches: the counts restoration,
the single pass-1 path and the domain-verdict smear removal now live in the
.ipynb itself. The only injected cell is a headless-matplotlib preamble, so the
run is what a person pressing "Run All" in Jupyter would get.

Output paths come from the environment (DUALPASS_PASS1_H5AD, DUALPASS_PASS2_H5AD,
DUALPASS_FIGDIR, DUALPASS_TRACE_DIR) so a verification run cannot overwrite the
canonical objects until its numbers have been checked.

usage: run_dualpass_src.py <SAMPLE>
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NBDIR = Path("/home/rodrigok/Notebooks/Spatial_MN_correct")
MARKER = "SOURCE FIX 2026-08-17"

SAMPLE = sys.argv[1]
OUTDIR = Path(os.environ.get(
    "DUALPASS_RUN_DIR",
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_IND_QC/runs_srcfixed"))
TRACE_DIR = Path(os.environ.get("DUALPASS_TRACE_DIR", str(OUTDIR)))

NB_IN = NBDIR / f"(RK)QC_per_sample_dualpass_{SAMPLE}.ipynb"
if not NB_IN.exists():
    alt = NBDIR / f"(RK)QC_per_sample_dualpass_{SAMPLE}-Copy1.ipynb"
    if not alt.exists():
        raise SystemExit(f"no notebook for {SAMPLE}: tried {NB_IN.name} and {alt.name}")
    NB_IN = alt

OUTDIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)
NB_OUT = OUTDIR / f"executed_src_{SAMPLE}.ipynb"
TRACE = TRACE_DIR / f"trace_src_{SAMPLE}.json"

PREAMBLE = '''
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
__TRACE__ = {}
'''


def main():
    t0 = time.time()
    nb = nbformat.read(NB_IN, as_version=4)

    if not any(MARKER in "".join(c.get("source", "")) for c in nb.cells):
        raise SystemExit(f"{NB_IN.name} carries no '{MARKER}' cell -- run "
                         f"fix_notebooks_at_source.py first")

    nb.cells.insert(0, nbformat.v4.new_code_cell(PREAMBLE))

    client = NotebookClient(nb, timeout=-1, kernel_name="curio_spatial",
                            resources={"metadata": {"path": str(NBDIR)}},
                            allow_errors=True)
    status, err = "OK", None
    try:
        client.execute()
    except Exception as e:                                   # noqa: BLE001
        status, err = "CLIENT_ERROR", f"{type(e).__name__}: {e}"
        traceback.print_exc()

    cell_errors = []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                cell_errors.append({"cell": i, "ename": o.get("ename"),
                                    "evalue": (o.get("evalue") or "")[:400]})
    if cell_errors:
        status = "CELL_ERROR"

    nbformat.write(nb, NB_OUT)

    meta = {"sample": SAMPLE, "mode": "src", "status": status, "client_error": err,
            "cell_errors": cell_errors, "n_cells": len(nb.cells),
            "runtime_s": round(time.time() - t0, 1), "executed_nb": str(NB_OUT),
            "notebook_in": str(NB_IN),
            "env": {k: os.environ.get(k) for k in
                    ("DUALPASS_PASS1_H5AD", "DUALPASS_PASS2_H5AD",
                     "DUALPASS_FIGDIR", "DUALPASS_TRACE_DIR")}}
    if TRACE.exists():
        body = json.loads(TRACE.read_text())
        body["run"] = meta
    else:
        body = {"sample": SAMPLE, "run": meta, "INCOMPLETE": True}
    TRACE.write_text(json.dumps(body, indent=1, default=str))

    print(f"[{SAMPLE}] {status} in {meta['runtime_s']}s -> {NB_OUT}", flush=True)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
