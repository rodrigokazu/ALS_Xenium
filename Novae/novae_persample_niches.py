"""
novae_persample_niches.py
============================================================
Joint Novae niche identification across all ALS SC Xenium samples, with
per-sample outputs and plots.

Design (decided with the spatial crew review):
  * ONE Novae model trained jointly over all slides using slide_key='run_id'
    (per-slide spatial graphs + Novae's built-in batch correction) so the
    domain/niche labels are COMPARABLE across samples.
  * Novae's spatial graph is built on TRUE microns (obsm['spatial_orig']),
    NOT the grid-offset obsm['spatial']. We overwrite obsm['spatial'] <-
    spatial_orig and build a Delaunay graph (coord_type='generic',
    delaunay=True, percentile pruning of long edges).
    NOTE: we deliberately do NOT pass technology='xenium' — in novae 1.0.3
    that REBUILDS obsm['spatial'] from x_centroid/y_centroid and would discard
    the spatial_orig swap (verified in the novae source).
  * Minimal pre-Novae filtering: drop 0-count cells (min_counts>=1) so Novae's
    internal normalisation cannot divide by zero. This is the ONLY filtering
    (input object is intentionally pre-QC).
  * Domains assigned at MULTIPLE hierarchy levels (Novae level=7 alone tends to
    over-split spinal cord); headline plots use a primary level, all levels are
    saved for inspection.

CAVEATS carried from the crew review (act on before trusting results):
  * Two samples (SD006/14, SD028/18 — both CONTROLS) were segmented with a
    different large-cells reseg run. A segmentation-version effect can appear as
    its own domain and contaminate the by-status comparison. obs['seg_version']
    is annotated and the by-status plot is labelled NOT-yet-confound-checked.
    Recommended follow-up: leave-2-out sensitivity run.
  * Graph length-scale: Delaunay is parameter-light for this first pass. The
    rigorous follow-up is an explicit radius sweep (~30/50/80/100 um) chosen by
    co-registration with DAPI/CHAT-defined ventral horn.

Outputs (under OUTPUT_DIR):
  - novae_all_samples_domains.h5ad            full object + latent + domain labels (all levels)
  - per_sample/<sample>__niches.h5ad          per-sample subset (lean)
  - plots/<sample>__niches_spatial.png        per-sample spatial niche map (true microns, primary level)
  - plots/domain_proportions_by_sample.png    composition per sample (primary level)
  - plots/domain_proportions_by_status.png    composition by disease status (primary level; confound-flagged)
  - plots/seg_version_by_sample.png           seg-version annotation per sample (confound check aid)
  - plots/training_loss.png / novae_*.png     Novae native plots (best-effort)
  - summary/domain_by_sample_counts.csv, domain_by_status_counts.csv
  - summary/run_manifest.json

Env: novae_env (Python 3.11 venv). Run with modules python/3.11.1 + gcc/13.3.0
loaded (see run_novae_persample_niches.sh). Needs a GPU (accelerator='cuda').
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import novae
import torch

# ============================================================
# CONFIG
# ============================================================
INPUT_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd/ALS_SCXenium_concatenated_offset_allsamples_reseg_20260617.h5ad"
OUTPUT_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches"

SLIDE_KEY = "run_id"          # one Xenium run = one slide (physical batch unit)
SAMPLE_KEY = "sample"         # donor/sample label for per-sample saves & plots
STATUS_KEY = "status"         # control / c9 / sporadic

# Graph: Delaunay on true microns (NO technology= ; that would overwrite spatial)
DELAUNAY = True
EDGE_PERCENTILE = 99          # prune the longest 1% of Delaunay edges (tissue-boundary artefacts)

LEVELS = [4, 5, 7]            # Novae hierarchical domain levels to assign
PRIMARY_LEVEL = 5             # headline plots/tables use this level
MAX_EPOCHS = 50               # early-stopping (patience=3, min_delta=0.1) should end before this
NUM_WORKERS = 8               # parallel DataLoader workers (Novae starves the GPU at 0; node has 8 cpus)
MIN_COUNTS = 1                # drop empty cells only (input is pre-QC by design)
UNASSIGNED = "unassigned"     # label for cells Novae marks neighbourhood-invalid (NaN domain)
POINT_SIZE = 2.0
DPI = 200

# Samples on the newer large-cells reseg run (segmentation-version confound; both controls)
RESEG_2026_TAG = "20260612"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ("per_sample", "plots", "summary"):
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def log(m): print(m, flush=True)

# ============================================================
# LOAD + MINIMAL PREP
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obsm={list(adata.obsm.keys())}")

assert "spatial_orig" in adata.obsm, "spatial_orig (true microns) missing — required for the Novae graph"
for k in (SLIDE_KEY, SAMPLE_KEY, STATUS_KEY):
    assert k in adata.obs, f"obs['{k}'] missing"

# Segmentation-version annotation (confound aid): tag the newer reseg slides.
adata.obs["seg_version"] = np.where(
    adata.obs[SLIDE_KEY].astype(str).str.contains(RESEG_2026_TAG),
    "largecells_reseg_2026-06-12", "largecells_reseg_2025-12",
)
adata.obs["seg_version"] = adata.obs["seg_version"].astype("category")
_reseg_samples = sorted(adata.obs.loc[adata.obs["seg_version"] == "largecells_reseg_2026-06-12", SAMPLE_KEY].unique())
log(f"[seg] newer-reseg samples (confound to watch, both controls): {_reseg_samples}")

# Use TRUE microns for the spatial graph (NOT the grid-offset coords).
adata.obsm["spatial_grid_offset"] = np.asarray(adata.obsm["spatial"], dtype=float)   # keep for reference (copy)
adata.obsm["spatial"] = np.asarray(adata.obsm["spatial_orig"], dtype=float)
log("[coords] obsm['spatial'] <- spatial_orig (true microns); grid-offset kept as 'spatial_grid_offset'")

# Drop empty cells so Novae normalisation cannot divide by zero (only filtering applied).
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
log(f"[filter] min_counts>={MIN_COUNTS}: {n0} -> {adata.n_obs} cells ({n0 - adata.n_obs} empty dropped)")

# Keep raw counts intact in a layer (Novae handles its own normalisation internally).
if "counts" not in adata.layers:
    adata.layers["counts"] = adata.X.copy()

# ============================================================
# NOVAE — joint model across all slides
# ============================================================
use_cuda = torch.cuda.is_available()
accelerator = "cuda" if use_cuda else "cpu"
log(f"[novae] torch={torch.__version__} cuda_available={use_cuda} -> accelerator='{accelerator}'")
if not use_cuda:
    log("[novae] WARNING: no GPU detected — training on CPU will be slow.")

log(f"[novae] building spatial neighbours (slide_key='{SLIDE_KEY}', delaunay={DELAUNAY}, percentile={EDGE_PERCENTILE}) on true microns")
novae.spatial_neighbors(
    adata, slide_key=SLIDE_KEY, coord_type="generic",
    delaunay=DELAUNAY, percentile=EDGE_PERCENTILE,
)

log(f"[novae] init + fit (max_epochs={MAX_EPOCHS})")
model = novae.Novae(adata)
model.fit(adata, max_epochs=MAX_EPOCHS, accelerator=accelerator, num_workers=NUM_WORKERS)

log("[novae] compute_representations")
model.compute_representations(adata, accelerator=accelerator, num_workers=NUM_WORKERS)

# Report Novae's neighbourhood-validity (cells it cannot place -> NaN domain)
n_invalid = None
if "neighborhood_valid" in adata.obs:
    n_invalid = int((~adata.obs["neighborhood_valid"].astype(bool)).sum())
    log(f"[novae] neighbourhood-invalid cells (will be '{UNASSIGNED}'): {n_invalid}")

# Assign domains at several levels; relabel NaN -> 'unassigned'; keep all.
domain_keys = {}
for lvl in LEVELS:
    key = model.assign_domains(adata, level=lvl)
    s = adata.obs[key].astype("object")
    s = s.where(adata.obs[key].notna(), UNASSIGNED).astype(str)
    adata.obs[key] = pd.Categorical(s)
    domain_keys[lvl] = key
    log(f"[novae] level={lvl} -> '{key}': {adata.obs[key].nunique()} labels")

primary_key = domain_keys[PRIMARY_LEVEL]
domains = sorted(adata.obs[primary_key].astype(str).unique())
log(f"[novae] PRIMARY level={PRIMARY_LEVEL} key='{primary_key}' domains={domains}")

# ============================================================
# SAVE FULL OBJECT
# ============================================================
full_out = os.path.join(OUTPUT_DIR, "novae_all_samples_domains.h5ad")
log(f"[save] full object -> {full_out}")
adata.write_h5ad(full_out, compression="gzip")

# ============================================================
# SUMMARY TABLES (primary level; include 'unassigned' for consistency)
# ============================================================
ct_sample = pd.crosstab(adata.obs[primary_key], adata.obs[SAMPLE_KEY], dropna=False)
ct_status = pd.crosstab(adata.obs[primary_key], adata.obs[STATUS_KEY], dropna=False)
ct_sample.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_sample_counts.csv"))
ct_status.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_status_counts.csv"))
log("[summary] wrote domain x sample and domain x status count tables")

# ============================================================
# PLOTS
# ============================================================
cmap = plt.get_cmap("tab20")
dom_to_color = {d: ("0.8" if d == UNASSIGNED else cmap(i % 20)) for i, d in enumerate(domains)}

def per_sample_spatial_plot(sub, sample_name, path):
    xy = np.asarray(sub.obsm["spatial_orig"], dtype=float)
    dvals = sub.obs[primary_key].astype(str).values
    fig, ax = plt.subplots(figsize=(7, 7))
    for d in domains:
        m = dvals == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[dom_to_color[d]], label=d, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{sample_name} — Novae niches (level {PRIMARY_LEVEL})")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(markerscale=4, fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="domain")
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)

log("[per-sample] writing lean subsets + spatial niche maps")
for sample_name, idx in adata.obs.groupby(SAMPLE_KEY, observed=True).groups.items():
    sub = adata[idx].copy()
    sub.obsp.clear()  # drop the sliced spatial-graph matrices -> lean per-sample file
    safe = str(sample_name).replace("/", "_")
    sub.write_h5ad(os.path.join(OUTPUT_DIR, "per_sample", f"{safe}__niches.h5ad"), compression="gzip")
    per_sample_spatial_plot(sub, sample_name, os.path.join(OUTPUT_DIR, "plots", f"{safe}__niches_spatial.png"))
    log(f"  [{sample_name}] n={sub.n_obs}")

def stacked_proportion_plot(crosstab, title, path, note=None):
    prop = crosstab.div(crosstab.sum(axis=0), axis=1)   # columns=groups, rows=domains
    groups = list(prop.columns)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(groups) + 3), 6))
    for d in prop.index.astype(str):
        vals = prop.loc[d].values
        ax.bar(x, vals, bottom=bottom, color=dom_to_color.get(d), label=d)
        bottom += vals
    ax.set_ylabel("proportion of cells"); ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="domain")
    if note:
        ax.text(0.0, -0.35, note, transform=ax.transAxes, fontsize=6, color="firebrick", wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)

stacked_proportion_plot(ct_sample, f"Novae domain composition by sample (level {PRIMARY_LEVEL})",
                        os.path.join(OUTPUT_DIR, "plots", "domain_proportions_by_sample.png"))
stacked_proportion_plot(
    ct_status, f"Novae domain composition by disease status (level {PRIMARY_LEVEL})",
    os.path.join(OUTPUT_DIR, "plots", "domain_proportions_by_status.png"),
    note=("CONFOUND-FLAGGED: 2 control samples (SD006/14, SD028/18) use a different segmentation "
          "run. Verify seg_version composition before reading this as biology (leave-2-out check)."),
)
log("[plots] wrote composition plots (by sample, by status)")

# seg_version composition per sample (confound check aid)
ct_seg = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs["seg_version"], dropna=False)
fig, ax = plt.subplots(figsize=(max(8, 0.5 * ct_seg.shape[0] + 3), 4))
x = np.arange(ct_seg.shape[0])
ax.bar(x, ct_seg.iloc[:, 0].values, label=ct_seg.columns[0])
if ct_seg.shape[1] > 1:
    ax.bar(x, ct_seg.iloc[:, 1].values, bottom=ct_seg.iloc[:, 0].values, label=ct_seg.columns[1])
ax.set_xticks(x); ax.set_xticklabels(list(ct_seg.index), rotation=90, fontsize=7)
ax.set_ylabel("cells"); ax.set_title("Segmentation version per sample"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "plots", "seg_version_by_sample.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

# Novae native plots (best-effort; never fail the run if signatures differ).
for fn, args, name in [
    (getattr(novae.plot, "loss_curve", None), (model,), "training_loss.png"),
    (getattr(novae.plot, "domains_proportions", None), (adata,), "novae_domains_proportions.png"),
]:
    if fn is None:
        continue
    try:
        fn(*args)
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", name), dpi=DPI, bbox_inches="tight")
        plt.close("all")
        log(f"[plots] novae.plot -> {name}")
    except Exception as e:
        log(f"[plots] skipped novae.plot {name}: {type(e).__name__}: {e}")
        plt.close("all")

# ============================================================
# MANIFEST
# ============================================================
manifest = {
    "input_h5ad": INPUT_H5AD,
    "output_dir": OUTPUT_DIR,
    "n_cells_after_filter": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "n_samples": int(adata.obs[SAMPLE_KEY].nunique()),
    "n_slides": int(adata.obs[SLIDE_KEY].nunique()),
    "slide_key": SLIDE_KEY,
    "graph": {"delaunay": DELAUNAY, "edge_percentile": EDGE_PERCENTILE, "coords": "spatial_orig (true microns)"},
    "domain_keys_by_level": domain_keys,
    "primary_level": PRIMARY_LEVEL,
    "primary_domain_key": primary_key,
    "n_domains_primary": len(domains),
    "domains_primary": domains,
    "max_epochs": MAX_EPOCHS,
    "accelerator": accelerator,
    "min_counts_filter": MIN_COUNTS,
    "n_neighbourhood_invalid": n_invalid,
    "reseg_2026_samples": _reseg_samples,
    "confound_note": "2 reseg-2026 samples are both controls; verify seg_version composition / leave-2-out before status comparison.",
    "status_domain_counts": ct_status.to_dict(),
}
with open(os.path.join(OUTPUT_DIR, "summary", "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)
log("[manifest] wrote summary/run_manifest.json")
log("Done.")
