# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/novae_resubset_stage2.py
#
# Stage two. Trains a completely fresh Novae model on just the D1014 and D1018 cells to find
# finer sub-domains inside them.
#
# The graph construction is the part that matters, and it differs from the full-cord runs for
# a reason. Delaunay edges are capped at a hard 100 um radius rather than pruned by
# percentile. On a subset the tissue is gappy, and a percentile cut keeps long edges that
# bridge across the gaps, wiring together cells that are nowhere near each other. The paper
# uses a micron ceiling for the same reason. A 50 um sensitivity run is still open.
#
# Targets 4, 6 and 8 sub-domains with 6 primary. Purity against the parent domains came out
# clean for five of six, with one sitting at roughly 40/60 across the grey and white matter
# boundary. It is flagged, not hidden. Whether that one is a real interface or a residual
# bridge artefact is unresolved.
#
# Do not pass technology='xenium' to spatial_neighbors; it rebuilds obsm['spatial'] from the
# centroid columns and discards the spatial_orig swap.
#
# One pandas trap that caused a failed run: adata.obs.pop(col, None) is invalid, because
# DataFrame.pop does not take a default the way dict.pop does. Use del adata.obs[col].
# ========================================================================================

"""
novae_resubset_stage2.py
============================================================
STAGE 2 of the ventral-horn (VH) sub-domain pipeline (sheff_als_spatial).

Goal: take the D1014+D1018 subset built in Stage 1 and train a FRESH Novae model on ONLY
those cells to resolve FINER sub-domains inside the two retained domains (the joint model's
level-5 domains are too coarse for ventral-horn microanatomy). Assign sub-domains at several
finer Novae levels; a PRIMARY level drives the headline plots.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DOCUMENTED CAVEAT (read before trusting Stage-2 sub-domains)
  Re-running spatial_neighbors on the SUBSET rebuilds the graph over a tissue with the
  non-D1014/D1018 cells REMOVED. The Delaunay graph then connects cells across the gaps
  left by the removed domains, so some neighbourhoods are DISTORTED relative to the intact
  tissue (a kept cell may be joined to a kept cell that was not its true physical neighbour).
  Sub-domains driven by such cross-gap edges are artefacts. Mitigations already in place:
  (i) percentile edge-pruning drops the longest edges (the worst cross-gap links);
  (ii) the CONFIG block exposes the graph params so the spatial reviewer can tighten them;
  (iii) Stage-3 reports composition with the seg-version confound flagged.
  The rigorous alternative (NOT done here) is to keep the ORIGINAL full-tissue graph and
  re-cluster only the latent of the kept cells; that is a larger change and is left as a
  reviewer-gated follow-up.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

INPUT (read-only): ALT_DIR/all_samples__D1014_D1018.h5ad  (Stage-1 output; obsp cleared).
  obs: novae_domains_5 (D1014/D1018), run_id, sample, status, seg_version.
  obsm: novae_latent(64), spatial(true µm), spatial_orig. layers['counts']=raw.

OUTPUTS (created under ALT_DIR):
  novae_subset_domains.h5ad                         full subset + new latent + sub-domain labels (all levels)
  per_sample/<sample>__subdomains.h5ad              lean per-sample subset (primary + all levels)
  plots_subdomains/<sample>__subdomains.pdf/.png    per-sample spatial map of PRIMARY sub-domains
  summary/stage2_manifest.json                      levels, n sub-domains/level, primary key, accelerator

Env: novae_env (Python 3.11 venv); module load python/3.11.1 gcc/13.3.0. NEEDS A GPU
(accelerator='cuda'). See run_novae_resubset_stage2.sh.

ROBUSTNESS: matplotlib Agg; plot/per-sample/native-plot blocks in try/except that log+continue;
print(flush=True); preconditions asserted at load; raw counts kept canonical (Novae does its own
internal normalisation; we never overwrite layers['counts']).
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
import torch
import novae

import nature_style as ns

# ============================================================
# ===================  TUNABLE CONFIG  =======================
#  Tune these from the spatial reviewer's recommendation. The graph rebuild on the subset
#  is the main correctness lever (see DOCUMENTED CAVEAT in the header).
# ============================================================
# --- graph (novae.spatial_neighbors) ---
# jarvis-spatial BLOCKER: on a gappy 2-domain subset, percentile pruning RETAINS long "bridge"
# edges spanning the deleted gaps (the 99th-pct edge is itself inflated by the gaps), smearing
# expression across tissue boundaries that don't exist. The Novae paper (Methods 4.13) drops
# edges >100 µm. So we use a HARD micron cap via `radius` (NOT delaunay+percentile).
COORD_TYPE = "generic"      # do NOT pass technology='xenium' (it rebuilds obsm['spatial'] from x/y)
# Paper Methods 4.13: a Delaunay graph with edges > RADIUS_UM µm DROPPED. novae applies the µm
# cap via radius=[0,RADIUS_UM] on the Delaunay edges — a HARD distance cap that removes the
# cross-gap "bridge" edges this 2-domain subset would otherwise create. (50 µm = sensitivity TODO.)
DELAUNAY = True             # Delaunay graph (its edges then pruned by the RADIUS_UM µm cap)
RADIUS_UM = 100.0           # hard max neighbour distance, µm (paper-matched ceiling)
EDGE_PERCENTILE = None      # optional extra percentile prune; None = rely on the µm cap alone

# --- sub-domain resolution: target sub-domain COUNTS (n_domains), not tree levels. jarvis-spatial:
#     a 2-compartment subset over-fragments at the parent's level integers; aim ~4/6/8, primary ~6. ---
N_DOMAINS_TARGETS = [4, 6, 8]   # finer sub-domain counts to assign on the subset
PRIMARY_N = 6                   # headline plots/tables use this many sub-domains

# --- training ---
MAX_EPOCHS = 50
NUM_WORKERS = 8             # DataLoader workers (Novae starves the GPU at 0; node has 8 cpus)
MIN_COUNTS = 1             # drop empty cells only (so Novae normalisation cannot divide by zero)
SEED = 0
# ============================================================
# =================  END TUNABLE CONFIG  =====================
# ============================================================

INPUT_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_D1014_D1018_subset/all_samples__D1014_D1018.h5ad"
ALT_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_D1014_D1018_subset"

SLIDE_KEY = "run_id"
SAMPLE_KEY = "sample"
STATUS_KEY = "status"
SEG_KEY = "seg_version"
PARENT_DOMAIN_KEY = "novae_domains_5"   # the Stage-1 domain label (D1014/D1018), kept for reference

SPATIAL_PLOT_COORDS = "spatial_orig"
POINT_SIZE = 2.0
SCALEBAR_UM = 500
UNASSIGNED = "unassigned"
SUBDOMAIN_PREFIX = "novae_subdomains_"  # obs keys: novae_subdomains_<level>

FULL_OUT_NAME = "novae_subset_domains.h5ad"
PER_SAMPLE_DIR = os.path.join(ALT_DIR, "per_sample")
PLOTS_DIR = os.path.join(ALT_DIR, "plots_subdomains")
SUMMARY_DIR = os.path.join(ALT_DIR, "summary")

# ============================================================
# SETUP
# ============================================================
for d in (ALT_DIR, PER_SAMPLE_DIR, PLOTS_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)


def log(m):
    print(m, flush=True)


MANIFEST = {
    "stage": 2,
    "input_h5ad": INPUT_H5AD,
    "alt_dir": ALT_DIR,
    "config": {
        "coord_type": COORD_TYPE, "delaunay": DELAUNAY, "edge_percentile": EDGE_PERCENTILE,
        "radius_um": RADIUS_UM, "n_domains_targets": N_DOMAINS_TARGETS, "primary_n": PRIMARY_N,
        "max_epochs": MAX_EPOCHS, "num_workers": NUM_WORKERS, "min_counts": MIN_COUNTS, "seed": SEED,
    },
    "caveat": ("graph rebuilt on the D1014/D1018 SUBSET => neighbourhoods could bridge the gaps "
               "left by removed domains; mitigated with a HARD radius<=100um cap (paper Methods 4.13); "
               "50um sensitivity is a documented follow-up; reviewer may tune CONFIG."),
    "blocks": {},
    "warnings": [],
}


def record(block, status, **detail):
    MANIFEST["blocks"][block] = {"status": status, **detail}
    log(f"[manifest] {block}: {status}" + (f"  {detail}" if detail else ""))


def write_manifest():
    path = os.path.join(SUMMARY_DIR, "stage2_manifest.json")
    try:
        with open(path, "w") as f:
            json.dump(MANIFEST, f, indent=2, default=str)
        log(f"[manifest] wrote {path}")
    except Exception as e:
        log(f"[manifest] FAILED to write {path}: {type(e).__name__}: {e}")


def _safe(name):
    return str(name).replace("/", "_")


# ============================================================
# LOAD + PRECONDITIONS
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obsm={list(adata.obsm.keys())}  layers={list(adata.layers.keys())}")

assert "spatial_orig" in adata.obsm, "obsm['spatial_orig'] (true microns) missing — required for the graph"
assert "counts" in adata.layers, "layers['counts'] (raw) missing"
for k in (SLIDE_KEY, SAMPLE_KEY, STATUS_KEY):
    assert k in adata.obs, f"obs['{k}'] missing"

MANIFEST["n_cells_input"] = int(adata.n_obs)
MANIFEST["n_genes"] = int(adata.n_vars)
MANIFEST["n_samples"] = int(adata.obs[SAMPLE_KEY].nunique())
MANIFEST["n_slides"] = int(adata.obs[SLIDE_KEY].nunique())

# ---- use TRUE microns for the spatial graph (defensive: re-assert spatial<-spatial_orig) ----
adata.obsm["spatial"] = np.asarray(adata.obsm["spatial_orig"], dtype=float)
log("[coords] obsm['spatial'] <- spatial_orig (true microns)")

# ---- drop empty cells so Novae internal normalisation cannot divide by zero ----
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
log(f"[filter] min_counts>={MIN_COUNTS}: {n0} -> {adata.n_obs} ({n0 - adata.n_obs} empty dropped)")
MANIFEST["n_cells_after_filter"] = int(adata.n_obs)
assert adata.n_obs > 0, "all cells dropped by min_counts filter — aborting"

# ============================================================
# NOVAE — fresh model on the subset
# ============================================================
use_cuda = torch.cuda.is_available()
accelerator = "cuda" if use_cuda else "cpu"
log(f"[novae] torch={torch.__version__} cuda_available={use_cuda} -> accelerator='{accelerator}'")
if not use_cuda:
    MANIFEST["warnings"].append("no GPU detected — Novae trained on CPU (slow).")
    log("[novae] WARNING: no GPU detected — training on CPU will be slow.")
MANIFEST["accelerator"] = accelerator

# ---- spatial graph (rebuilt on the subset; see DOCUMENTED CAVEAT) ----
try:
    sn_kwargs = dict(slide_key=SLIDE_KEY, coord_type=COORD_TYPE, delaunay=DELAUNAY, radius=RADIUS_UM)
    if EDGE_PERCENTILE is not None:
        sn_kwargs["percentile"] = EDGE_PERCENTILE
    log(f"[novae] spatial_neighbors({sn_kwargs}) on true microns "
        f"[subset graph; Delaunay edges pruned >{RADIUS_UM}um — paper Methods 4.13]")
    novae.spatial_neighbors(adata, **sn_kwargs)
    record("spatial_neighbors", "ok")
except Exception as e:
    record("spatial_neighbors", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())
    write_manifest()
    raise   # the graph is mandatory; nothing downstream can run without it

# ---- init + fit + representations ----
log(f"[novae] init Novae + fit(max_epochs={MAX_EPOCHS}, accelerator='{accelerator}', num_workers={NUM_WORKERS})")
model = novae.Novae(adata)
model.fit(adata, max_epochs=MAX_EPOCHS, accelerator=accelerator, num_workers=NUM_WORKERS)
log("[novae] compute_representations")
model.compute_representations(adata, accelerator=accelerator, num_workers=NUM_WORKERS)

n_invalid = None
if "neighborhood_valid" in adata.obs:
    n_invalid = int((~adata.obs["neighborhood_valid"].astype(bool)).sum())
    log(f"[novae] neighbourhood-invalid cells (-> '{UNASSIGNED}'): {n_invalid}")
MANIFEST["n_neighbourhood_invalid"] = n_invalid

# Protect the parent label first: assign_domains(level=L) writes obs['novae_domains_<L>'], which
# at L=5 would CLOBBER the parent novae_domains_5 (D1014/D1018). Copy it to a safe key for QC.
PARENT_SAFE = "parent_domain"
if PARENT_DOMAIN_KEY in adata.obs:
    adata.obs[PARENT_SAFE] = adata.obs[PARENT_DOMAIN_KEY].astype(str).values

# ---- assign sub-domains: novae assign_domains(n_domains=) EXACT-matches-or-raises (verified in
#      source) — on a 2-compartment subset the exact targets may be unreachable. So SCAN levels
#      (always valid; write to a temp key) to learn the achievable count per level, then pick the
#      level whose count is NEAREST each requested target. Never aborts on an unreachable count. ----
MAX_LEVEL_SCAN = 15
SCAN_KEY = "_scan_tmp"
level_to_count = {}
for L in range(1, MAX_LEVEL_SCAN + 1):
    try:
        model.assign_domains(adata, level=L, key_added=SCAN_KEY)
        level_to_count[L] = int(adata.obs[SCAN_KEY].nunique(dropna=True))
    except Exception as e:
        log(f"[novae] level scan stopped at L={L}: {type(e).__name__}: {e}")
        break
if SCAN_KEY in adata.obs.columns:
    del adata.obs[SCAN_KEY]
assert level_to_count, "no Novae level could be assigned — aborting"
log(f"[novae] achievable sub-domain counts by level: {level_to_count}")
MANIFEST["achievable_level_counts"] = {int(k): int(v) for k, v in level_to_count.items()}

# pick the level whose achieved count is nearest each requested target (unique levels)
chosen_levels = sorted({
    min(level_to_count, key=lambda L, t=nd: (abs(level_to_count[L] - t), L))
    for nd in N_DOMAINS_TARGETS
})
log(f"[novae] chosen levels (nearest targets {N_DOMAINS_TARGETS}): {chosen_levels}")

subdomain_keys = {}            # actual_count -> obs key
n_subdomains_by_count = {}
for L in chosen_levels:
    cnt = level_to_count[L]
    std_key = f"{SUBDOMAIN_PREFIX}{cnt}"
    model.assign_domains(adata, level=L, key_added=std_key)   # parent novae_domains_5 untouched
    s = adata.obs[std_key].astype("object")
    s = s.where(adata.obs[std_key].notna(), UNASSIGNED).astype(str)
    adata.obs[std_key] = pd.Categorical(s)
    subdomain_keys[cnt] = std_key
    n_subdomains_by_count[cnt] = int(adata.obs[std_key].nunique())
    log(f"[novae] level={L} -> {cnt} sub-domains -> obs['{std_key}']")

assert subdomain_keys, "no sub-domain assignment succeeded — aborting"

# PRIMARY = the achieved count nearest PRIMARY_N (default 6)
primary_count = min(subdomain_keys, key=lambda c: (abs(c - PRIMARY_N), c))
primary_key = subdomain_keys[primary_count]
subdomains = sorted(adata.obs[primary_key].astype(str).unique())
log(f"[novae] PRIMARY ~{PRIMARY_N} -> {primary_count} sub-domains, key='{primary_key}'")

MANIFEST["subdomain_keys_by_count"] = subdomain_keys
MANIFEST["n_subdomains_by_count"] = n_subdomains_by_count
MANIFEST["primary_n_requested"] = PRIMARY_N
MANIFEST["primary_n_actual"] = primary_count
MANIFEST["primary_subdomain_key"] = primary_key
MANIFEST["n_subdomains_primary"] = len(subdomains)
MANIFEST["subdomains_primary"] = subdomains
record("assign_subdomains", "ok", chosen_levels=list(chosen_levels),
       counts=list(subdomain_keys.keys()), primary_key=primary_key)

# ---- sub-domain QC (jarvis-spatial guards): per-slide n, sub-domain sizes, parent-purity ----
try:
    qc = {}
    # per-slide subset size (flag training-floor < 2000)
    slide_n = adata.obs[SLIDE_KEY].value_counts()
    small_slides = sorted(slide_n[slide_n < 2000].index.astype(str))
    qc["per_slide_n"] = {str(k): int(v) for k, v in slide_n.items()}
    qc["slides_below_2000"] = small_slides
    if small_slides:
        MANIFEST["warnings"].append(f"slides with <2000 subset cells (weak Novae training): {small_slides}")
    # sub-domain cohort size + n samples present (flag <500 cells or in <3 samples)
    sd_size = adata.obs[primary_key].value_counts()
    sd_nsamp = adata.obs.groupby(primary_key, observed=True)[SAMPLE_KEY].nunique()
    qc["subdomain_size"] = {str(k): int(v) for k, v in sd_size.items()}
    qc["subdomain_n_samples"] = {str(k): int(v) for k, v in sd_nsamp.items()}
    tiny = sorted(set(sd_size[sd_size < 500].index.astype(str)) | set(sd_nsamp[sd_nsamp < 3].index.astype(str)))
    qc["subdomains_low_confidence"] = [s for s in tiny if s != UNASSIGNED]
    if qc["subdomains_low_confidence"]:
        MANIFEST["warnings"].append(f"low-confidence sub-domains (<500 cells or <3 samples): {qc['subdomains_low_confidence']}")
    # parent-domain purity: each sub-domain should be >80% one parent (else a cross-gap bridge artefact)
    if PARENT_SAFE in adata.obs:
        pur = pd.crosstab(adata.obs[primary_key], adata.obs[PARENT_SAFE], normalize="index")
        pur.to_csv(os.path.join(SUMMARY_DIR, "stage2_subdomain_parent_purity.csv"))
        max_pur = pur.max(axis=1)
        impure = sorted(max_pur[(max_pur < 0.8) & (max_pur.index.astype(str) != UNASSIGNED)].index.astype(str))
        qc["subdomains_impure_parent_lt80pct"] = impure
        qc["parent_purity"] = {str(k): round(float(v), 3) for k, v in max_pur.items()}
        if impure:
            MANIFEST["warnings"].append(f"sub-domains <80% one parent (possible cross-gap bridge artefact): {impure}")
    MANIFEST["subdomain_qc"] = qc
    record("subdomain_qc", "ok", small_slides=len(small_slides),
           low_conf=len(qc.get("subdomains_low_confidence", [])),
           impure=len(qc.get("subdomains_impure_parent_lt80pct", [])))
except Exception as e:
    record("subdomain_qc", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# SAVE FULL SUBSET OBJECT
# ============================================================
try:
    full_out = os.path.join(ALT_DIR, FULL_OUT_NAME)
    log(f"[save] full subset object -> {full_out}")
    adata.write_h5ad(full_out, compression="gzip")
    MANIFEST["full_h5ad"] = full_out
    record("save_full", "ok", path=full_out, n_cells=int(adata.n_obs))
except Exception as e:
    record("save_full", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# SUMMARY TABLES (primary level)
# ============================================================
try:
    ct_sample = pd.crosstab(adata.obs[primary_key], adata.obs[SAMPLE_KEY], dropna=False)
    ct_status = pd.crosstab(adata.obs[primary_key], adata.obs[STATUS_KEY], dropna=False)
    ct_sample.to_csv(os.path.join(SUMMARY_DIR, "stage2_subdomain_by_sample_counts.csv"))
    ct_status.to_csv(os.path.join(SUMMARY_DIR, "stage2_subdomain_by_status_counts.csv"))
    record("summary_tables", "ok")
except Exception as e:
    record("summary_tables", "failed", error=f"{type(e).__name__}: {e}")
    log(traceback.format_exc())

# ============================================================
# PER-SAMPLE lean subdomain subsets
# ============================================================
per_sample_written = {}
try:
    groups = adata.obs.groupby(SAMPLE_KEY, observed=True).groups
    log(f"[per-sample] writing {len(groups)} lean per-sample subdomain subsets")
    for sample_name, idx in groups.items():
        try:
            ss = adata[idx].copy()
            ss.obsp.clear()  # drop the sliced graph -> lean file
            ss.uns.pop("neighbors", None)
            out = os.path.join(PER_SAMPLE_DIR, f"{_safe(sample_name)}__subdomains.h5ad")
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
# PER-SAMPLE Nature-tier spatial plots of PRIMARY sub-domains
# ============================================================
ns.apply_style()
SUBDOMAIN_COLORS = ns.domain_colors(subdomains, unassigned=UNASSIGNED)
log(f"[plot] sub-domain colours: {SUBDOMAIN_COLORS}")


def per_sample_subdomain_plot(ss, sample_name, basepath):
    xy = np.asarray(ss.obsm[SPATIAL_PLOT_COORDS], dtype=float)
    dvals = ss.obs[primary_key].astype(str).values
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    present = [d for d in subdomains if (dvals == d).any()]
    for d in present:
        m = dvals == d
        ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[SUBDOMAIN_COLORS[d]],
                   label=d, linewidths=0, rasterized=True)
    ns.despine_spatial(ax)
    ax.set_title(f"{sample_name} — sub-domains (n={primary_count})")
    try:
        x0 = float(np.nanpercentile(xy[:, 0], 2))
        y0 = float(np.nanpercentile(xy[:, 1], 98))
        ns.add_scalebar(ax, (x0, y0), length_um=SCALEBAR_UM)
    except Exception as e:
        log(f"  [{sample_name}] scalebar skipped: {type(e).__name__}: {e}")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=SUBDOMAIN_COLORS[d], markeredgecolor="none", label=d)
               for d in present]
    ncol = 2 if len(present) > 8 else 1
    ax.legend(handles=handles, title="sub-domain", loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0, ncol=ncol)
    ns.save_both(fig, basepath)


plot_results = {}
try:
    for sample_name in sorted(adata.obs[SAMPLE_KEY].astype(str).unique()):
        try:
            ss = adata[adata.obs[SAMPLE_KEY].astype(str) == sample_name]
            if ss.n_obs == 0:
                continue
            base = os.path.join(PLOTS_DIR, f"{_safe(sample_name)}__subdomains")
            per_sample_subdomain_plot(ss, sample_name, base)
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

# ---- Novae native plots (best-effort) ----
for fn_name, args, name in [
    ("loss_curve", (model,), "stage2_training_loss"),
    ("domains_proportions", (adata,), "stage2_novae_subdomains_proportions"),
]:
    fn = getattr(novae.plot, fn_name, None)
    if fn is None:
        continue
    try:
        fn(*args)
        fig = plt.gcf()
        ns.save_both(fig, os.path.join(PLOTS_DIR, name))
        log(f"[plots] novae.plot.{fn_name} -> {name}.pdf/.png")
    except Exception as e:
        log(f"[plots] skipped novae.plot.{fn_name}: {type(e).__name__}: {e}")
        plt.close("all")

# ============================================================
# MANIFEST
# ============================================================
write_manifest()
log("Stage 2 done.")
