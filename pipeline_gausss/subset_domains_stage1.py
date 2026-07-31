# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/subset_domains_stage1.py
#
# Stage one of three. Takes the joint domain map and keeps only D1014 and D1018, the grey
# matter and reactive niches that contain the ventral horn, then writes a combined subset,
# lean per-sample subsets, and a spatial plot per section.
#
# The reason for the whole chain is resolution. Level-5 domains across a full cord are too
# coarse for ventral horn microanatomy, so the approach is to cut down to the relevant tissue
# and re-train instead of pushing the joint model to a finer level where it over-splits
# everything else.
#
# Output was 537,469 cells and the plots look like a textbook cross-section, with D1014 as the
# blue grey-matter butterfly and D1018 as the orange white-matter rim. That visual check is
# the point of the stage; if it does not look like a cord, stop.
#
# Domain ids here belong to the _gausss joint run and do not correspond to anything in the
# _FINAL output.
# ========================================================================================

"""
subset_domains_stage1.py
============================================================
STAGE 1 of the ventral-horn (VH) sub-domain pipeline (sheff_als_spatial).

Goal: from the joint Novae domain map of the full ALS spinal-cord Xenium cohort, KEEP
only the two domains of interest (D1014, D1018) and emit a clean combined subset + lean
per-sample subsets + a Nature-tier per-sample spatial plot, ready for Stage 2 (a fresh
Novae model re-trained on just these domains to find FINER sub-domains).

INPUT (read-only):
  /oak/.../RK/Spatial/Novae_persample_niches/novae_all_samples_domains.h5ad
  1,685,385 cells x 480 genes. Relevant obs: novae_domains_5
  (D1003/D1014/D1016/D1017/D1018/unassigned), run_id (slide), sample (donor),
  status (control/c9/sporadic), seg_version. obsm: novae_latent (64),
  spatial (TRUE microns), spatial_orig. layers['counts']=raw. obsp = the (now-stale-once-
  subset) Novae spatial graph.

WHY drop obsp: the Novae spatial graph (obsp) is built over the FULL cell set. After
subsetting to two domains the retained submatrix is a torn graph (neighbours of kept
cells that fell in dropped domains are gone) -> it is STALE and must NOT be reused.
Stage 2 rebuilds the graph from scratch on the subset. We therefore clear obsp here.

OUTPUTS (created under ALT_DIR):
  all_samples__D1014_D1018.h5ad                 combined subset (obsp cleared)
  per_sample/<sample>__D1014_D1018.h5ad          lean per-sample subset (obsp cleared)
  plots_subset/<sample>__domains.pdf/.png        per-sample spatial map (2-colour domains)
  summary/stage1_manifest.json                   provenance + per-sample/domain counts

Env: novae_env (Python 3.11 venv); module load python/3.11.1 gcc/13.3.0. CPU only.
See run_subset_stage1.sh.

ROBUSTNESS: matplotlib Agg; every plot/analysis block in try/except that logs+continues;
print(flush=True); preconditions asserted at load; raw counts kept canonical (no
normalisation in this stage).
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nature_style as ns

# ============================================================
# CONFIG
# ============================================================
INPUT_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/novae_all_samples_domains.h5ad"
ALT_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_D1014_D1018_subset"

DOMAIN_KEY = "novae_domains_5"      # the primary domain key on the full object
KEEP_DOMAINS = ["D1014", "D1018"]   # the two domains to isolate for sub-domain re-clustering
SLIDE_KEY = "run_id"                # one Xenium run = one slide
SAMPLE_KEY = "sample"               # donor / sample label
STATUS_KEY = "status"               # control / c9 / sporadic
SEG_KEY = "seg_version"             # segmentation-version annotation (confound aid)

SPATIAL_PLOT_COORDS = "spatial_orig"  # plot on TRUE microns
POINT_SIZE = 2.0
SCALEBAR_UM = 500

COMBINED_NAME = "all_samples__D1014_D1018.h5ad"
PER_SAMPLE_DIR = os.path.join(ALT_DIR, "per_sample")
PLOTS_DIR = os.path.join(ALT_DIR, "plots_subset")
SUMMARY_DIR = os.path.join(ALT_DIR, "summary")

# ============================================================
# SETUP
# ============================================================
for d in (ALT_DIR, PER_SAMPLE_DIR, PLOTS_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)


def log(m):
    print(m, flush=True)


MANIFEST = {
    "stage": 1,
    "input_h5ad": INPUT_H5AD,
    "alt_dir": ALT_DIR,
    "domain_key": DOMAIN_KEY,
    "keep_domains": KEEP_DOMAINS,
    "blocks": {},
    "warnings": [],
}


def record(block, status, **detail):
    MANIFEST["blocks"][block] = {"status": status, **detail}
    log(f"[manifest] {block}: {status}" + (f"  {detail}" if detail else ""))


def write_manifest():
    path = os.path.join(SUMMARY_DIR, "stage1_manifest.json")
    try:
        with open(path, "w") as f:
            json.dump(MANIFEST, f, indent=2, default=str)
        log(f"[manifest] wrote {path}")
    except Exception as e:
        log(f"[manifest] FAILED to write {path}: {type(e).__name__}: {e}")


# ============================================================
# LOAD + PRECONDITIONS
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obs cols include domain={DOMAIN_KEY in adata.obs}  "
    f"obsm={list(adata.obsm.keys())}  layers={list(adata.layers.keys())}")

# Hard preconditions — fail loudly; everything downstream depends on these.
assert DOMAIN_KEY in adata.obs, f"obs['{DOMAIN_KEY}'] missing"
assert SPATIAL_PLOT_COORDS in adata.obsm, f"obsm['{SPATIAL_PLOT_COORDS}'] missing"
assert "counts" in adata.layers, "layers['counts'] (raw) missing"
for k in (SLIDE_KEY, SAMPLE_KEY, STATUS_KEY):
    assert k in adata.obs, f"obs['{k}'] missing"

present_domains = set(adata.obs[DOMAIN_KEY].astype(str).unique())
missing_keep = [d for d in KEEP_DOMAINS if d not in present_domains]
assert not missing_keep, (
    f"requested KEEP_DOMAINS {missing_keep} not present in obs['{DOMAIN_KEY}'] "
    f"(present: {sorted(present_domains)})"
)

if SEG_KEY not in adata.obs:
    MANIFEST["warnings"].append(f"obs['{SEG_KEY}'] absent on input; seg-confound annotation unavailable.")
    log(f"[warn] obs['{SEG_KEY}'] absent — continuing without seg-version annotation")

MANIFEST["n_cells_input"] = int(adata.n_obs)
MANIFEST["n_genes"] = int(adata.n_vars)
MANIFEST["domain_labels_input"] = sorted(present_domains)

# ============================================================
# SUBSET to KEEP_DOMAINS
# ============================================================
keep_mask = adata.obs[DOMAIN_KEY].astype(str).isin(KEEP_DOMAINS).values
n_keep = int(keep_mask.sum())
log(f"[subset] keeping {n_keep}/{adata.n_obs} cells in domains {KEEP_DOMAINS}")
assert n_keep > 0, "no cells matched KEEP_DOMAINS — aborting"

# .copy() after boolean subset => a standalone object. We then clear obsp because the
# sliced graph is stale (see header). spatial/spatial_orig/novae_latent/counts are kept.
sub = adata[keep_mask].copy()
sub.obsp.clear()
sub.uns.pop("neighbors", None)          # drop any cached neighbour bookkeeping too
# fold the categorical down to only the kept domains (drops empty D1003/D1016/... levels)
sub.obs[DOMAIN_KEY] = sub.obs[DOMAIN_KEY].astype("category").cat.remove_unused_categories()
log(f"[subset] obsp cleared (stale after subsetting); kept obsm={list(sub.obsm.keys())} "
    f"layers={list(sub.layers.keys())}; domains now={list(sub.obs[DOMAIN_KEY].cat.categories)}")

MANIFEST["n_cells_kept"] = int(sub.n_obs)

# ------------------------------------------------------------
# per-sample x domain count report
# ------------------------------------------------------------
try:
    ct = pd.crosstab(sub.obs[SAMPLE_KEY], sub.obs[DOMAIN_KEY], dropna=False)
    ct["total"] = ct.sum(axis=1)
    ct_path = os.path.join(SUMMARY_DIR, "stage1_counts_by_sample_domain.csv")
    ct.to_csv(ct_path)
    MANIFEST["counts_by_sample_domain"] = ct.to_dict(orient="index")
    # also a by-status x domain rollup (kept cells)
    ct_status = pd.crosstab(sub.obs[STATUS_KEY], sub.obs[DOMAIN_KEY], dropna=False)
    ct_status.to_csv(os.path.join(SUMMARY_DIR, "stage1_counts_by_status_domain.csv"))
    MANIFEST["counts_by_status_domain"] = ct_status.to_dict(orient="index")
    log(f"[counts] per-sample x domain table -> {ct_path}\n{ct}")
    record("counts", "ok", table=ct_path)
except Exception as e:
    record("counts", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# SAVE combined subset
# ============================================================
try:
    combined_path = os.path.join(ALT_DIR, COMBINED_NAME)
    log(f"[save] combined subset -> {combined_path}")
    sub.write_h5ad(combined_path, compression="gzip")
    MANIFEST["combined_h5ad"] = combined_path
    record("save_combined", "ok", path=combined_path, n_cells=int(sub.n_obs))
except Exception as e:
    record("save_combined", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# PER-SAMPLE lean subsets
# ============================================================
def _safe(name):
    return str(name).replace("/", "_")


per_sample_written = {}
try:
    groups = sub.obs.groupby(SAMPLE_KEY, observed=True).groups
    log(f"[per-sample] writing {len(groups)} lean per-sample subsets")
    for sample_name, idx in groups.items():
        try:
            ss = sub[idx].copy()
            ss.obsp.clear()  # already cleared on parent, but keep per-sample files lean/explicit
            ss.uns.pop("neighbors", None)
            out = os.path.join(PER_SAMPLE_DIR, f"{_safe(sample_name)}__D1014_D1018.h5ad")
            ss.write_h5ad(out, compression="gzip")
            per_sample_written[str(sample_name)] = {"n_cells": int(ss.n_obs), "path": out}
            log(f"  [{sample_name}] n={ss.n_obs} -> {out}")
        except Exception as e:
            per_sample_written[str(sample_name)] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            log(f"  [{sample_name}] FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
    MANIFEST["per_sample"] = per_sample_written
    record("save_per_sample", "ok", n_samples=len(per_sample_written))
except Exception as e:
    record("save_per_sample", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# PER-SAMPLE Nature-tier spatial plots (2-colour domains)
# ============================================================
ns.apply_style()
DOMAIN_COLORS = ns.domain_colors(KEEP_DOMAINS)  # stable colorblind-safe 2-colour map
log(f"[plot] domain colours: {DOMAIN_COLORS}")


def per_sample_spatial_plot(ss, sample_name, basepath):
    xy = np.asarray(ss.obsm[SPATIAL_PLOT_COORDS], dtype=float)
    dvals = ss.obs[DOMAIN_KEY].astype(str).values
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for d in KEEP_DOMAINS:
        m = dvals == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[DOMAIN_COLORS[d]],
                       label=d, linewidths=0, rasterized=True)
    ns.despine_spatial(ax)
    ax.set_title(str(sample_name))
    # scale bar anchored near the lower-left of the point cloud (true microns)
    try:
        x0 = float(np.nanpercentile(xy[:, 0], 2))
        y0 = float(np.nanpercentile(xy[:, 1], 98))  # large data-y => bottom on inverted axis
        ns.add_scalebar(ax, (x0, y0), length_um=SCALEBAR_UM)
    except Exception as e:
        log(f"  [{sample_name}] scalebar skipped: {type(e).__name__}: {e}")
    # legend OUTSIDE the axes (right), with proxy handles so marker size is legible
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="none", label=d)
               for d in KEEP_DOMAINS]
    ax.legend(handles=handles, title="domain", loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    ns.save_both(fig, basepath)


plot_results = {}
try:
    for sample_name in [str(s) for s in sub.obs[SAMPLE_KEY].cat.categories] if hasattr(
            sub.obs[SAMPLE_KEY], "cat") else sorted(sub.obs[SAMPLE_KEY].astype(str).unique()):
        try:
            ss = sub[sub.obs[SAMPLE_KEY].astype(str) == sample_name]
            if ss.n_obs == 0:
                continue
            base = os.path.join(PLOTS_DIR, f"{_safe(sample_name)}__domains")
            per_sample_spatial_plot(ss, sample_name, base)
            plot_results[sample_name] = "ok"
            log(f"  [plot {sample_name}] -> {base}.pdf/.png")
        except Exception as e:
            plot_results[sample_name] = f"failed: {type(e).__name__}: {e}"
            log(f"  [plot {sample_name}] FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            plt.close("all")
    record("per_sample_plots", "ok" if any(v == "ok" for v in plot_results.values()) else "failed",
           per_sample=plot_results)
except Exception as e:
    record("per_sample_plots", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())
    plt.close("all")

# ============================================================
# MANIFEST
# ============================================================
write_manifest()
log("Stage 1 done.")
