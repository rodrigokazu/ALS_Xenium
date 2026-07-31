# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/novae_JOINT_FINAL.py
#
# One Novae model across all 20 slides at once. That is what makes domain labels mean the same
# thing in every section. If you want to compare composition between samples or between
# disease groups, this is the run to use.
#
# slide_key is run_id, one Xenium run per slide, and the sweep assigns domains at 4, 6, 8, 10
# and 12 targets, written as novae_domains_n4 through n12. The headline is the 6-domain
# solution. It also produces the motor-neuron-by-niche summaries. Where the motor neurons sit
# is the question the whole spinal cord effort is built around.
#
# Novae API traps, all of which have bitten us. Do not pass technology='xenium' to
# spatial_neighbors; in novae 1.0.3 it rebuilds obsm['spatial'] from the centroid columns and
# throws away the spatial_orig swap, so use coord_type='generic' with delaunay instead.
# assign_domains(n_domains=N) matches exactly or raises and never clamps, so the code scans
# hierarchy levels and picks the nearest. assign_domains(level=L) writes
# obs['novae_domains_<L>'] and will clobber a parent column of the same name, so pass
# key_added. Filter min_counts >= 1 first or you get divide-by-zero NaNs, and cells Novae
# cannot place come back NaN and are relabelled 'unassigned'.
#
# Set num_workers to match the CPU request. At 0 the dataloader starves the GPU and the job
# crawls. Eight workers took a 1.79M cell run from hopeless to about three hours on a
# qrtx4000.
#
# Read the confound note before drawing any biological conclusion. Spinal level is almost
# perfectly confounded with disease in this cohort: all 10 controls are cervical and 8 of 10
# ALS cases are lumbar. The cervical and lumbar sets in the code exist so the crosstabs can be
# honest about it. The only level-matched contrast available is SD03522 and SD01320 against
# the cervical controls. That is n=2.
#
# Domain ids are specific to this model. They are not comparable to any earlier run.
# ========================================================================================

"""
novae_JOINT_FINAL.py
============================================================
JOINT Novae niche identification on the MN-corrected _FINAL (Marcel's
Ranger_procd_mw_final segmentation) ALS SC Xenium object, with per-sample
outputs, plots, AND motor-neuron positioning (MN x niche) summaries.

Faithful port of the proven _gausss run
(/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/novae_persample_niches_MNcorrected.py).
ONLY the following change for _FINAL (everything else is preserved verbatim):
  * INPUT  -> the combined _FINAL concat (cfg.COMBINED_H5AD).
  * OUTPUT -> cfg.NOVAE_JOINT_DIR (Novae_persample_niches_FINAL).
  * Constants/paths are IMPORTED from final_config (single source of truth) instead
    of being re-derived here.
  * NICHE SWEEP widened: TARGET_N_DOMAINS = [4, 6, 8, 10, 12] (was [4, 6, 10]);
    writes novae_domains_n4/n6/n8/n10/n12. Headline still the ~6-domain solution.
  * seg_pipeline is now uniform 'mw_final' (was 'mw_gausss') -> no seg-version
    confound; annotated for provenance.
  * is_MN comes pre-baked in the combined _FINAL h5ad (the concat builder joins it
    via mn_annotation_join). This script READS it and DEGRADES GRACEFULLY when it is
    all-False (annotation still pending): MN overlay + MN-enrichment become honest
    placeholders instead of crashing.

Novae design unchanged (spatial-crew review; guardrails G11/G4/task-9):
  * ONE joint model, slide_key='run_id' (per-slide graphs + native batch correction)
    so niches are COMPARABLE across the 20 slides.
  * Graph on TRUE microns (obsm['spatial_orig']); overwrite obsm['spatial'] <-
    spatial_orig, Delaunay (coord_type='generic', delaunay=True, percentile edge
    pruning). NEVER pass technology='xenium' (it rebuilds obsm['spatial'] from
    x/y_centroid and discards the swap).
  * Only filtering: drop 0-count cells (min_counts>=1) so Novae normalisation can't
    divide by zero. Raw counts kept in layers['counts'] (guardrail G1: no library-
    size normalisation before domain ID).
  * Domains at TARGET COUNTS via a level-scan (novae assign_domains(n_domains=N)
    exact-matches-or-raises, so we scan levels and pick the nearest per target).
  * novae.plot.loss_curve is broken -> best-effort, never fails the run.

The primary confound is now SPINAL LEVEL x disease (ALS = 8 lumbar + 2 cervical;
controls = 10 cervical). spinal_level is annotated from an AUTHORITATIVE sd_code->level
map (cohort_ground_truth), NOT from obs['tissue_region']. Every by-status niche
contrast is confounded; the ONLY honest disease contrast is within-cervical (2 ALS vs
10 controls, underpowered). Treat this spinal Xenium as METHOD VALIDATION / positional
QC, not ALS-vs-control biology.

Env: novae_env (Python 3.11 venv), modules python/3.11.1 + gcc/13.3.0, GPU (accelerator='cuda').
"""

import os
import sys

# Belt-and-braces: drop user-site so a stale ~/.local anndata never shadows the env
# (the runner also exports PYTHONNOUSERSITE=1).
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
FINAL_CODE_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code"
if FINAL_CODE_DIR not in sys.path:
    sys.path.insert(0, FINAL_CODE_DIR)

import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import novae
import torch

import final_config as cfg
try:
    from mn_annotation_join import MN_LABEL_SPEC as _MN_LABEL_SPEC
except Exception:
    _MN_LABEL_SPEC = {"column": "MN_motorneuron", "positive_values": ["yes"]}

# ============================================================
# CONFIG  (paths imported from final_config; NOT re-derived)
# ============================================================
INPUT_H5AD = str(cfg.COMBINED_H5AD)
OUTPUT_DIR = str(cfg.NOVAE_JOINT_DIR)

SLIDE_KEY = "run_id"          # one Xenium run = one slide (expect 20 unique, 1:1 with sample)
SAMPLE_KEY = "sample"         # donor/sample label for per-sample saves & plots
STATUS_KEY = "status"         # control / c9 / sporadic
MN_KEY = "is_MN"              # annotated motor neurons (bool; pre-baked by the concat builder)

# Graph: Delaunay on true microns (NO technology= ; that would overwrite spatial)
DELAUNAY = True
EDGE_PERCENTILE = 99          # prune the longest 1% of Delaunay edges (tissue-boundary artefacts)

# Multi-resolution domains: hit sensible domain COUNTS (not raw hierarchy levels).
# novae assign_domains(n_domains=N) raises if no level matches exactly -> we SCAN levels and
# pick the level whose domain count is nearest each target (proven pattern from the _gausss run).
TARGET_N_DOMAINS = [4, 6, 8, 10, 12]   # widened sweep for _FINAL; write novae_domains_n{target}
PRIMARY_TARGET = 6             # headline plots/tables use the ~6-domain solution
SCAN_LEVELS = list(range(1, 13))  # candidate Novae hierarchy levels to scan for those counts
MAX_EPOCHS = 50               # early-stopping (patience=3, min_delta=0.1) should end before this
NUM_WORKERS = 8               # parallel DataLoader workers (Novae starves the GPU at 0; node has 8 cpus)
MIN_COUNTS = 1                # drop empty cells only (input is pre-QC by design)
UNASSIGNED = "unassigned"     # label for cells Novae marks neighbourhood-invalid (NaN domain)
POINT_SIZE = 2.0
DPI = 200
SEED = 0                      # reproducibility (best-effort; GPU training is not fully deterministic)

# Authoritative spinal level per donor (from souls/cohort_ground_truth.md; NOT obs['tissue_region']).
# ALS = 8 lumbar + 2 cervical (SD03522, SD01320); all 10 controls = cervical. 12 cervical / 8 lumbar.
CERVICAL_SD = {
    "SD03522", "SD01320",                                            # the 2 level-matched ALS cases
    "SD02818", "SD01915", "SD01115", "SD01015", "SD03914",           # controls
    "SD03614", "SD00614", "SD05413", "SD02913", "SD01413",
}
LUMBAR_SD = {"SD01923", "SD01623", "SD02622", "SD01922", "SD01616", "SD02022", "SD01620", "SD04219"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ("per_sample", "plots", "summary"):
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def log(m): print(m, flush=True)


def coerce_is_MN(series_or_none, n):
    """Return a REAL numpy bool array of length n, degrading safely.

    is_MN is written by the concat builder as a real numpy bool and preserved through the
    h5py write-sanitize (dtype==bool tested BEFORE is_numeric_dtype). Reading it back it
    should be bool. This helper is defensive against the classic 'False'->True trap if it
    ever comes back as a string/category, and degrades to all-False when is_MN is missing
    (annotation still pending).
    """
    if series_or_none is None:
        log("[MN] WARNING obs['is_MN'] MISSING -> is_MN all-False (annotation pending; "
            "re-run after the concat builder joins Marcel's _final annotation).")
        return np.zeros(n, dtype=bool)
    s = series_or_none
    dt = s.dtype
    if dt == bool:
        return np.asarray(s.to_numpy(), dtype=bool)
    if str(dt) in ("category", "object") or getattr(dt, "kind", "") in ("U", "S", "O"):
        # avoid str('False') -> True: only explicit truthy tokens are True
        vals = s.astype("object").astype(str).str.strip().str.lower()
        out = vals.isin({"true", "1", "yes", "t"}).to_numpy()
        log(f"[MN] NOTE obs['is_MN'] was {dt}; coerced truthy-token -> bool (True={int(out.sum())})")
        return np.asarray(out, dtype=bool)
    # numeric / nullable
    return np.asarray(pd.to_numeric(s, errors="coerce").fillna(0).to_numpy() != 0, dtype=bool)


# ============================================================
# LOAD + MINIMAL PREP
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obsm={list(adata.obsm.keys())}")

assert "spatial_orig" in adata.obsm, "spatial_orig (true microns) missing — required for the Novae graph"
for k in (SLIDE_KEY, SAMPLE_KEY, STATUS_KEY, "sd_code"):
    assert k in adata.obs, f"obs['{k}'] missing"

# is_MN: pre-baked by the concat builder; degrade gracefully if absent/all-False.
adata.obs[MN_KEY] = coerce_is_MN(adata.obs[MN_KEY] if MN_KEY in adata.obs else None, adata.n_obs)
n_mn_total = int(adata.obs[MN_KEY].sum())
is_MN_all_false = (n_mn_total == 0)
mn_source = str(adata.uns.get("is_MN_source", "unknown (not recorded in uns)"))
log(f"[MN] is_MN True: {n_mn_total} / {adata.n_obs}  | source: {mn_source}")
if is_MN_all_false:
    log("[MN] is_MN is ALL-FALSE (annotation pending) -> MN overlay + MN-enrichment outputs are "
        "PLACEHOLDERS; re-run this JOINT job once Marcel's _final MN annotation lands in the concat.")

# Provenance: segmentation is uniform (Marcel _final) -> no seg-version confound this time.
if "seg_pipeline" in adata.obs:
    _segs = sorted(adata.obs["seg_pipeline"].astype(str).unique())
    log(f"[seg] seg_pipeline (uniform '{cfg.SEG_PIPELINE}' expected): {_segs}")

# Authoritative spinal level from the cohort map (avoids the buggy tissue_region labels).
sd = adata.obs["sd_code"].astype(str)
adata.obs["spinal_level"] = np.where(
    sd.isin(CERVICAL_SD), "cervical",
    np.where(sd.isin(LUMBAR_SD), "lumbar", "unknown"),
)
adata.obs["spinal_level"] = adata.obs["spinal_level"].astype("category")
_unk = sorted(sd[adata.obs["spinal_level"] == "unknown"].unique())
assert not _unk, (f"spinal_level 'unknown' for sd_code(s) {_unk} — sd_code format changed? "
                  "The confound map (the whole reason for this re-run) would be void. Fix the map/keys.")
_lvl_samples = adata.obs.groupby("spinal_level", observed=True)[SAMPLE_KEY].nunique().to_dict()
assert _lvl_samples.get("cervical", 0) == 12 and _lvl_samples.get("lumbar", 0) == 8, \
    f"expected 12 cervical / 8 lumbar samples, got {_lvl_samples}"
assert adata.obs[SLIDE_KEY].nunique() == 20, f"expected 20 slides, got {adata.obs[SLIDE_KEY].nunique()}"
log(f"[level] spinal_level cells: {adata.obs['spinal_level'].value_counts().to_dict()} | samples: {_lvl_samples}")

# Use TRUE microns for the spatial graph (NOT the grid-offset coords).
adata.obsm["spatial_grid_offset"] = np.asarray(adata.obsm["spatial"], dtype=float)   # keep for reference (copy)
adata.obsm["spatial"] = np.asarray(adata.obsm["spatial_orig"], dtype=float)
log("[coords] obsm['spatial'] <- spatial_orig (true microns); grid-offset kept as 'spatial_grid_offset'")

# Drop empty cells so Novae normalisation cannot divide by zero (only filtering applied).
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
log(f"[filter] min_counts>={MIN_COUNTS}: {n0} -> {adata.n_obs} cells ({n0 - adata.n_obs} empty dropped)")
log(f"[MN] is_MN after filter: {int(adata.obs[MN_KEY].sum())}")

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

# Best-effort reproducibility (GPU training is not fully deterministic; the seed is recorded in the manifest).
import random as _random
np.random.seed(SEED); _random.seed(SEED); torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)
try:
    novae.utils.set_seed(SEED)
except Exception as _e:
    log(f"[seed] novae.utils.set_seed unavailable ({type(_e).__name__}); used numpy/torch/random seeds")
log(f"[seed] SEED={SEED}")

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

# --- Level scan: map each Novae hierarchy level to its domain count ---
level_keys, level_counts = {}, {}
for L in SCAN_LEVELS:
    try:
        k = model.assign_domains(adata, level=L)
    except Exception as e:
        log(f"[scan] level {L} unavailable: {type(e).__name__}: {e}")
        continue
    level_keys[L] = k
    level_counts[L] = int(adata.obs[k].dropna().astype(str).nunique())
    log(f"[scan] level {L} -> {level_counts[L]} domains (key '{k}')")
assert level_counts, "no Novae levels could be assigned"
log(f"[scan] level->count: {level_counts}")

# --- Pick the level nearest each target domain count (distinct levels) ---
chosen_level = {}
used = set()
for tgt in TARGET_N_DOMAINS:
    cand = sorted(level_counts.items(), key=lambda kv: (abs(kv[1] - tgt), kv[0]))
    lvl = next((L for L, _ in cand if L not in used), cand[0][0])
    used.add(lvl)
    chosen_level[tgt] = lvl
    log(f"[domains] target {tgt} -> level {lvl} ({level_counts[lvl]} domains)")

# --- Build clean alias columns novae_domains_n{target}; relabel NaN -> unassigned ---
domain_keys = {}
for tgt, lvl in chosen_level.items():
    src = level_keys[lvl]
    s = adata.obs[src].astype("object")
    s = s.where(adata.obs[src].notna(), UNASSIGNED).astype(str)
    alias = f"novae_domains_n{tgt}"
    adata.obs[alias] = pd.Categorical(s)
    domain_keys[tgt] = alias
    log(f"[domains] '{alias}': {adata.obs[alias].nunique()} labels (from level {lvl})")

# drop the raw scan columns to keep obs clean (aliases hold the copies).
# `del adata.obs[k]` NOT adata.obs.pop (pop takes no default here -> TypeError).
for L, k in level_keys.items():
    if (k in adata.obs) and (k not in set(domain_keys.values())):
        del adata.obs[k]

primary_key = domain_keys.get(PRIMARY_TARGET, domain_keys[TARGET_N_DOMAINS[0]])
domains = sorted(adata.obs[primary_key].astype(str).unique())
# actual Novae hierarchy level behind the primary target (for truthful plot titles that say "level N")
PRIMARY_LEVEL = chosen_level.get(PRIMARY_TARGET, chosen_level[TARGET_N_DOMAINS[0]])
log(f"[novae] PRIMARY target={PRIMARY_TARGET} level={PRIMARY_LEVEL} key='{primary_key}' domains={domains}")

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
ct_level = pd.crosstab(adata.obs[primary_key], adata.obs["spinal_level"], dropna=False)
ct_sample.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_sample_counts.csv"))
ct_status.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_status_counts.csv"))
ct_level.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_spinal_level_counts.csv"))

# Honest breakdown: domain x (status x spinal_level); the by-status contrast alone is level-confounded.
ct_status_level = pd.crosstab(adata.obs[primary_key], [adata.obs[STATUS_KEY], adata.obs["spinal_level"]], dropna=False)
ct_status_level.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_status_x_level_counts.csv"))
# The ONLY honest disease contrast: WITHIN cervical (2 level-matched ALS vs 10 cervical controls; underpowered).
_cerv = adata.obs["spinal_level"] == "cervical"
ct_cerv = pd.crosstab(adata.obs.loc[_cerv, primary_key], adata.obs.loc[_cerv, STATUS_KEY], dropna=False)
ct_cerv.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_within_cervical_by_status_counts.csv"))
with open(os.path.join(OUTPUT_DIR, "summary", "CONFOUND_README.txt"), "w") as _fh:
    _fh.write(
        "SPINAL-LEVEL x DISEASE CONFOUND — read before any ALS-vs-control niche claim.\n"
        "ALS = 8 lumbar + 2 cervical (SD03522, SD01320); all 10 controls = cervical.\n"
        "=> the by-status comparison is confounded with spinal level (lumbar is 100% ALS).\n"
        "The ONLY honest disease contrast is WITHIN CERVICAL: the 2 cervical ALS cases vs the 10 cervical\n"
        "controls (domain_within_cervical_by_status_counts.csv) — and it is UNDERPOWERED (n=2 ALS).\n"
        "slide_key='run_id' is 1:1 with sample, so Novae's batch unit == the biological unit.\n"
        "Segmentation is uniform (mw_final) -> no seg-version confound; spinal level is the primary one.\n"
        "Treat this spinal Xenium as METHOD VALIDATION / positional QC, NOT ALS-vs-control biology.\n"
    )
log("[summary] wrote domain x {sample,status,level,status_x_level,within_cervical} tables + CONFOUND_README")

# ---- MOTOR-NEURON POSITIONING: is_MN x domain (degrades to placeholders if all-False) ----
mn = adata.obs[MN_KEY].values.astype(bool)
ct_mn = pd.crosstab(adata.obs[primary_key], adata.obs[MN_KEY], dropna=False)
ct_mn.to_csv(os.path.join(OUTPUT_DIR, "summary", "domain_by_isMN_counts.csv"))

# MN enrichment per domain: fraction of MNs vs fraction of all cells (log2), + within-domain MN rate.
frac_mn = adata.obs.loc[mn, primary_key].astype(str).value_counts(normalize=True)
frac_all = adata.obs[primary_key].astype(str).value_counts(normalize=True)
dom_size = adata.obs[primary_key].astype(str).value_counts()
mn_in_dom = adata.obs.loc[mn, primary_key].astype(str).value_counts()
enr = pd.DataFrame({
    "n_cells": dom_size,
    "n_MN": mn_in_dom,
    "frac_of_MNs": frac_mn,
    "frac_of_all_cells": frac_all,
}).fillna(0.0)
enr["within_domain_MN_rate"] = enr["n_MN"] / enr["n_cells"].replace(0, np.nan)
enr["log2_MN_enrichment"] = np.log2((enr["frac_of_MNs"] + 1e-9) / (enr["frac_of_all_cells"] + 1e-9))
enr = enr.sort_values("log2_MN_enrichment", ascending=False)
enr.to_csv(os.path.join(OUTPUT_DIR, "summary", "MN_enrichment_by_domain.csv"))
if is_MN_all_false:
    log("[MN] enrichment table is a PLACEHOLDER (0 MNs; annotation pending).")
else:
    log("[MN] domain enrichment:\n" + enr.to_string())

# MN domain composition per sample (which niche the MNs of each donor fall in).
ct_mn_sample = pd.crosstab(adata.obs.loc[mn, SAMPLE_KEY], adata.obs.loc[mn, primary_key], dropna=False)
ct_mn_sample.to_csv(os.path.join(OUTPUT_DIR, "summary", "MN_domain_by_sample.csv"))
log("[MN] wrote is_MN x domain, MN enrichment, and MN-domain-by-sample tables")

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
    # Overlay the annotated motor neurons (black rings) so their niche placement is visible.
    mmn = sub.obs[MN_KEY].values.astype(bool)
    if mmn.any():
        ax.scatter(xy[mmn, 0], xy[mmn, 1], s=POINT_SIZE * 6, facecolors="none",
                   edgecolors="k", linewidths=0.4, label=f"MN (n={int(mmn.sum())})", zorder=5)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{sample_name} — Novae niches (level {PRIMARY_LEVEL}) + MNs")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(markerscale=4, fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="domain")
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)

log("[per-sample] writing lean subsets + spatial niche maps (with MN overlay)")
for sample_name, idx in adata.obs.groupby(SAMPLE_KEY, observed=True).groups.items():
    sub = adata[idx].copy()
    sub.obsp.clear()  # drop the sliced spatial-graph matrices -> lean per-sample file
    safe = str(sample_name).replace("/", "_")
    sub.write_h5ad(os.path.join(OUTPUT_DIR, "per_sample", f"{safe}__niches.h5ad"), compression="gzip")
    per_sample_spatial_plot(sub, sample_name, os.path.join(OUTPUT_DIR, "plots", f"{safe}__niches_spatial.png"))
    log(f"  [{sample_name}] n={sub.n_obs}  MN={int(sub.obs[MN_KEY].sum())}")

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
    note=("CONFOUND-FLAGGED: spinal level is confounded with disease (ALS = 8 lumbar + 2 cervical; "
          "controls = 10 cervical). Read any by-status domain difference against domain_proportions_by_"
          "spinal_level.png and the cervical-only ALS cases (SD03522, SD01320) before calling it biology."),
)
stacked_proportion_plot(ct_level, f"Novae domain composition by spinal level (level {PRIMARY_LEVEL})",
                        os.path.join(OUTPUT_DIR, "plots", "domain_proportions_by_spinal_level.png"))
log("[plots] wrote composition plots (by sample, by status, by spinal level)")

# MN enrichment bar (log2) per domain — where the motor neurons sit.
fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(enr) + 2), 4))
colors = [dom_to_color.get(d, "0.5") for d in enr.index.astype(str)]
ax.bar(np.arange(len(enr)), enr["log2_MN_enrichment"].values, color=colors)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(np.arange(len(enr))); ax.set_xticklabels(list(enr.index.astype(str)), rotation=45, ha="right", fontsize=8)
ax.set_ylabel("log2(MN fraction / all-cell fraction)")
_mn_title = "Motor-neuron enrichment per Novae niche"
if is_MN_all_false:
    _mn_title += " [PLACEHOLDER — is_MN all-False, annotation pending]"
ax.set_title(f"{_mn_title} (level {PRIMARY_LEVEL})")
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "plots", "MN_enrichment_by_domain.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)
log("[plots] wrote MN_enrichment_by_domain.png")

# Novae native plots (best-effort; never fail the run if signatures differ).
# NOTE: novae.plot.loss_curve is known-broken on this version -> the try/except skips it.
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
    "run_kind": "JOINT (one Novae model, slide_key='run_id', comparable domains)",
    "input_h5ad": INPUT_H5AD,
    "output_dir": OUTPUT_DIR,
    "n_cells_after_filter": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "n_samples": int(adata.obs[SAMPLE_KEY].nunique()),
    "n_slides": int(adata.obs[SLIDE_KEY].nunique()),
    "n_MN": int(adata.obs[MN_KEY].sum()),
    "is_MN_all_false": bool(is_MN_all_false),
    "is_MN_source": mn_source,
    "MN_LABEL_SPEC": _MN_LABEL_SPEC,
    "slide_key": SLIDE_KEY,
    "graph": {"delaunay": DELAUNAY, "edge_percentile": EDGE_PERCENTILE, "coords": "spatial_orig (true microns)"},
    "target_n_domains": TARGET_N_DOMAINS,
    "chosen_level_per_target": chosen_level,
    "level_scan_counts": level_counts,
    "domain_keys_by_target": domain_keys,
    "primary_target": PRIMARY_TARGET,
    "primary_domain_key": primary_key,
    "n_domains_primary": len(domains),
    "domains_primary": domains,
    "max_epochs": MAX_EPOCHS,
    "accelerator": accelerator,
    "min_counts_filter": MIN_COUNTS,
    "seed": SEED,
    "n_neighbourhood_invalid": n_invalid,
    "MN_enrichment_caveat": ("is_MN uses STMN2 + markers that ARE in the 480-gene panel Novae embeds, so "
                             "MN-in-niche enrichment is semi-circular (expected by construction). It is a "
                             "POSITIONAL QC signal (do MNs land in one coherent ventral-horn niche?), NOT a "
                             "novel MN-niche discovery — any claim needs per-sample consistency + a "
                             "label-permutation null + the panel/STMN2 circularity stated."),
    "disease_biology_caveat": ("Spinal level is confounded with disease (ALS 8 lumbar + 2 cervical; controls "
                               "10 cervical). Use as METHOD VALIDATION, not ALS-vs-control biology. Honest "
                               "contrast = within-cervical (n=2 ALS, underpowered). See summary/CONFOUND_README.txt."),
    "seg_pipeline": sorted(adata.obs["seg_pipeline"].astype(str).unique()) if "seg_pipeline" in adata.obs else None,
    "seg_pipeline_expected": cfg.SEG_PIPELINE,
    "spinal_level_source": "authoritative cohort map (sd_code); NOT obs['tissue_region'] (mislabels BA-block controls)",
    "spinal_level_counts": adata.obs["spinal_level"].value_counts().to_dict(),
    "confound_note": "Uniform mw_final segmentation (no seg-version confound). PRIMARY confound now = spinal level x disease (ALS 8 lumbar + 2 cervical; controls 10 cervical). Level-matched ALS = SD03522, SD01320.",
    "MN_enrichment_by_domain": enr.reset_index().rename(columns={"index": "domain"}).to_dict(orient="records"),
    "status_domain_counts": ct_status.to_dict(),
}
with open(os.path.join(OUTPUT_DIR, "summary", "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)
log("[manifest] wrote summary/run_manifest.json")
log("Done.")
