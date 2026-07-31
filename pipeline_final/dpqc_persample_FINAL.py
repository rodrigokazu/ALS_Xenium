#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_persample_FINAL.py
#
# The one-sample driver. Compute the metrics, draw the figures, assemble the PDF, for a single
# section selected by --idx.
#
# Writes under --outroot. In the array job that points at node-local /tmp and the result is
# tarred to OAK afterwards. Twenty tasks hammering shared storage with small writes is the
# thing being avoided.
#
# Inputs are the INDEPENDENT per-sample niches and the coarse cell types, both resolved
# through final_config instead of being spelled out here.
# ========================================================================================

"""dpqc_persample_FINAL.py -- one-sample driver for the Dual-pass QC pipeline (_FINAL re-run):
compute per-domain metrics + verdicts -> Nature-tier figures -> assembled per-sample PDF.
Writes everything under --outroot (use node-local /tmp in the array job, then tar to oak).
Reads the INDEPENDENT per-sample Novae niches from Novae_persample_INDEPENDENT_FINAL/per_sample
and the coarse cell types from CellTyping_coarse_FINAL (both via final_config paths).
"""
import os, sys, glob, argparse
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]  # PYTHONNOUSERSITE belt-and-braces
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import final_config as cfg
import dpqc_compute_FINAL as C
import dpqc_figures_FINAL as F
import dpqc_report_FINAL as R
import dpqc_style_FINAL as st


def run(h5, outroot, sample=None):
    A, obs, df, meta, rg = C.compute(h5, sample=sample)
    sample = meta["sample"]
    A.obs["cell_type"] = pd.Categorical(obs["cell_type"].values)
    meta = C.write_outputs(outroot, df, meta, rg)

    figdir = os.path.join(outroot, "figures", sample)
    os.makedirs(figdir, exist_ok=True)
    base = os.path.join(figdir, sample)
    F.make_master(A, df, meta, base + "__master")
    F.make_highlight(A, df, meta, base + "__highlight")
    F.make_markers(A, df, meta, rg, base + "__markers")
    F.make_qcpanels(A, df, meta, base + "__qcpanels")
    F.make_atlas(A, df, meta, base + "__atlas")
    ck = os.path.join(figdir, "_colourkey.png")
    st.colour_key_figure().savefig(ck, dpi=150, bbox_inches="tight")

    out_pdf = os.path.join(outroot, "pdf", f"{sample}__DualPassQC.pdf")
    R.build_persample_pdf(sample, meta, df, figdir, ck, out_pdf)
    print(f"[{sample}] DONE -> {out_pdf}  "
          f"KEEP={meta.get('n_keep',0)} REVIEW={meta.get('n_review',0)} REMOVE={meta.get('n_remove',0)} "
          f"({meta.get('pct_cells_remove',0)}% cells)", flush=True)
    return out_pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=None); ap.add_argument("--sample", default=None)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--persample-dir", default=str(cfg.NOVAE_INDEP_DIR / "per_sample"))
    ap.add_argument("--outroot", required=True)
    a = ap.parse_args()
    h5 = a.h5
    if h5 is None and a.idx is not None:
        h5 = sorted(glob.glob(os.path.join(a.persample_dir, "*niches_independent.h5ad")))[a.idx]
    if h5 is None and a.sample:
        hits = glob.glob(os.path.join(a.persample_dir, f"{a.sample}*niches_independent.h5ad"))
        h5 = hits[0] if hits else None
    if h5 is None:
        sys.exit("need --h5, --sample, or --idx")
    run(h5, a.outroot, sample=a.sample)
