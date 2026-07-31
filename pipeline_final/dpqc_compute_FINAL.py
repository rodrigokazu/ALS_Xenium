#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_compute_FINAL.py
#
# The engine of the dual-pass QC and the only file here that decides anything. Everything else
# in the dpqc set draws or narrates what this computes.
#
# After Novae runs independently on each sample, some domains are real tissue compartments and
# some are segmentation smear, tissue edge or debris. This scores every domain in a section
# and returns a KEEP / REVIEW / REMOVE verdict.
#
# Every metric is computed WITHIN a sample. That is not a stylistic choice. Spinal level,
# disease group and section quality all vary across the cohort, so a cross-sample threshold
# would mostly measure the confounds. Within-section scoring survives them.
#
# Two families of evidence. Spatial coherence asks whether the domain is a place: Moran's I on
# domain membership, the fraction of cells in the largest connected component, and the SpaGCN
# PAS statistic over a k=10 neighbourhood. Identity and QC ask whether it is a biological
# thing: how many strong specific markers it has, its negative-control fraction against the
# section median, its median transcript count against the section median, and how much of it
# is untyped relative to the sample baseline.
#
# The gates as set: Moran's I below 0.10, largest component below 0.20, PAS above 0.50 on the
# spatial side; at most one strong marker on the identity side; negative controls above three
# times the section median, median transcripts below 0.6 times, untyped above 1.5 times the
# baseline on the QC side.
#
# A domain is only removed when BOTH gates fail. One gate alone caps at REVIEW. There are
# three protections on top: three or more strong markers can never be removed, a motor-neuron
# fraction above twice the sample rate downgrades a REMOVE, and a domain under 50 cells is
# capped at REVIEW because there is no power to judge it.
#
# The motor-neuron protection reports itself as INACTIVE while annotation is pending, which it
# currently is. Do not read a clean QC pass as evidence that motor neurons were protected.
#
# On _final the segmentation is uniform, so the _gausss-era confound where the only two
# re-segmented samples were both controls is gone. seg_version now comes from
# obs['seg_pipeline'].
#
# Thresholds live at the top as named constants on purpose. explain_report_FINAL.py reads this
# file to build the glossary, so a threshold changed here propagates into the documentation
# instead of leaving it quietly wrong.
# ========================================================================================

"""dpqc_compute_FINAL.py -- Dual-pass QC (_FINAL re-run): deep per-domain characterisation of the
INDEPENDENT per-sample Novae niches with quantitative SMEAR detection (the field-standard spatial-
continuity triad + expression-identity + QC gates, all computed WITHIN sample to survive the
region/disease confounds). Scoring + two-gate verdict logic is IDENTICAL to the proven _gausss
dpqc_compute; the _FINAL deltas are: (1) input from Novae_persample_INDEPENDENT_FINAL +
CellTyping_coarse_FINAL, sourced via final_config; (2) a UNIFORM Marcel _final segmentation, so the
_gausss-era 2-control reseg-batch confound is GONE (seg_version now read from obs['seg_pipeline']);
(3) is_MN surfaced as an explicit MN-protection status that degrades to INACTIVE while Marcel's
_final annotation is still pending.

For ONE sample at the primary resolution novae_domains_n6, for EACH domain:
  spatial coherence : Moran's I of the domain indicator; kNN neighbourhood purity + enrichment
                      over prevalence; PAS (percentage of abnormal spots, SpaGCN); CHAOS (within-
                      domain 1-NN micron distance) + ratio to section; largest connected-component
                      fraction + n_components; convex-hull spread (hull-area frac / cell frac)
  expression id     : n strong specific positive markers (logFC>1 & padj<1e-10); top-10 marker
                      sign asymmetry (the D1007 'one gene propping it up' test); novae_latent
                      silhouette vs rest (subsampled)
  transcript QC     : median/IQR transcript_counts,total_counts,n_counts; ratio to section median;
                      low-count fraction; transcript density (tx/cell_area)
  morphology        : cell_area, nucleus_area, nucleus-free fraction, nuclear fraction
  negative control  : per-cell neg fraction (probe+codeword+genomic+deprecated) + ratio to section
  graph             : neighborhood_valid fraction
  composition       : cell_type_coarse dominant type + fraction + Shannon entropy; % untyped vs
                      sample baseline; is_MN count/fraction (UNVALIDATED, protective prior only)
  reproducibility   : cross-resolution nesting entropy of the n6 domain across n4 and n10

Verdict = TWO-GATE rule: REMOVE only if spatially incoherent AND identity/QC-failing (protective
override keeps rare-but-real strong-identity or MN-enriched domains at REVIEW). KEEP / REVIEW / REMOVE.

Writes stats/<sample>__dpqc.json, tables/<sample>__domain_metrics.csv, tables/<sample>__decisions.csv.
Reads the h5ad via ps_io (h5py build; anndata 0.10.8 cannot read /uns/log1p). DATA IS PRE-QC.
Runs in module python/3.11.1 (scanpy 1.9.8, sklearn, scipy).
"""
import os, sys, json, glob, argparse
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]  # PYTHONNOUSERSITE belt-and-braces
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
VH = "/home/rodrigok/SLURM_jobs/Spatial/VH_isolation"
PSR = os.path.join(VH, "persample_report")
for p in (HERE, VH, PSR):
    if p not in sys.path:
        sys.path.insert(0, p)
# NOT the shared ps_io.py (PSR, /home quota is full so it can't be patched in place): the shared
# module's _read_csr() assumes csr_matrix, but the _FINAL niches h5ads write layers/counts as
# csc_matrix -> silent shape mismatch ("index pointer size 481 should be 78640"). ps_io_FINAL_fix
# is a local copy (in HERE, already first on sys.path) with the encoding-type dispatch fixed;
# everything else is byte-identical to ps_io.py.
import ps_io_FINAL_fix as ps_io
import final_config as cfg   # single source of truth for _FINAL paths / SEG_PIPELINE

RES = "novae_domains_n6"
KNN = 15
PAS_K = 10                                              # SpaGCN PAS neighbourhood
SIL_SUB = 8000                                          # subsample cap for latent silhouette
# Coarse cell types from the _FINAL typing run. obs_celltype.csv: index = '<LABEL>:<cell_id>'
# (the _final integer obs_names), column 'cell_type_coarse'. CROSS-CHECK at launch: the
# CellTyping_coarse_FINAL adapter MUST key obs_celltype.csv on the SAME obs_names as the Novae
# per-sample h5ads (load_celltypes logs the matched fraction; expect ~1.0).
CT_CSV = str(cfg.COARSE_TYPING_DIR / "obs_celltype.csv")
OBS_COLS = ["novae_domains_n4", "novae_domains_n6", "novae_domains_n10", "is_MN",
            "status", "sample", "sd_code", "transcript_counts", "total_counts", "n_counts",
            "cell_area", "nucleus_area", "nucleus_count", "control_probe_counts",
            "control_codeword_counts", "genomic_control_counts", "deprecated_codeword_counts",
            "neighborhood_valid", "xenium_cell_id", "seg_pipeline"]

# _FINAL DELTA: Marcel's _final segmentation is applied UNIFORMLY to all 20 sections, so the
# _gausss-era "2 controls from a newer reseg batch" segmentation confound is RESOLVED. seg_version
# is now read per sample from obs['seg_pipeline'] (== SEG_PIPELINE 'mw_final' for the whole cohort);
# RESEG_2026 is kept (empty) only so any residual reference resolves to the uniform-batch path.
RESEG_2026 = set()

# ---- gate thresholds (within-sample-safe; intrinsic metrics, self-calibrated where noted) ----
G_MORAN = 0.10          # spatial gate: below = incoherent
G_LARGEST_CC = 0.20     # spatial gate
G_PAS = 0.50            # spatial gate: >half of cells abnormal
ID_STRONG = 1           # identity gate: <=1 strong specific positive marker
ID_NEG_RATIO = 3.0      # QC gate: neg-control fraction > 3x section median
ID_COUNT_RATIO = 0.6    # QC gate: median tx < 0.6x section median
ID_UNTYPED_MULT = 1.5   # QC gate: untyped fraction > 1.5x sample baseline
PROTECT_STRONG = 3      # protective: >=3 strong markers -> never REMOVE (downgrade to REVIEW)
PROTECT_MN_MULT = 2.0   # protective: MN fraction > 2x sample -> downgrade REMOVE to REVIEW
SMALL_N = 50            # low-power domain -> cap at REVIEW


def _to_bool(s):
    return pd.Series(s).map(lambda v: str(v).lower() in ("true", "1", "1.0")).values


def _read_uns_str(h5, key):
    """Best-effort read of a scalar string uns key straight from the h5 (avoids anndata's
    /uns/log1p issue). Returns None if absent/unreadable. Used to surface the is_MN provenance
    (uns['is_MN_source']) that mn_annotation_join wrote upstream, so the report can state whether
    MN protection is active or the annotation is still pending."""
    try:
        import h5py
        with h5py.File(h5, "r") as f:
            if "uns" in f and key in f["uns"]:
                v = f["uns"][key][()]
                if isinstance(v, (bytes, bytearray)):
                    return v.decode()
                if isinstance(v, np.ndarray) and v.size == 1:
                    v = v.reshape(-1)[0]
                    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                return str(v)
    except Exception:
        pass
    return None


def spinal_level(sample):
    b = str(sample).split("_")[-1].upper()
    return "cervical" if b in ("BG", "BA") else ("lumbar" if b == "BI" else "unknown")


def load_celltypes(sample, obs_names):
    if not os.path.exists(CT_CSV):
        print(f"[{sample}] WARNING coarse cell-type CSV not found: {CT_CSV} -> all Untyped "
              f"(CellTyping_coarse_FINAL not built yet? composition axis inactive for this sample).",
              flush=True)
        return pd.Series(["Untyped"] * len(obs_names), index=obs_names)
    ct = pd.read_csv(CT_CSV, index_col=0)
    ct.index = ct.index.astype(str)
    if "cell_type_coarse" not in ct.columns:
        print(f"[{sample}] WARNING '{CT_CSV}' has no 'cell_type_coarse' column "
              f"(cols={list(ct.columns)[:8]}) -> all Untyped.", flush=True)
        return pd.Series(["Untyped"] * len(obs_names), index=obs_names)
    sub = ct[ct.index.str.startswith(f"{sample}:")]
    lab = sub["cell_type_coarse"].reindex(obs_names)
    matched = int(lab.notna().sum())
    overlap = matched / len(obs_names) if len(obs_names) else 0.0
    print(f"[{sample}] cell-type join: {matched}/{len(obs_names)} = {overlap:.4f} obs matched "
          f"CellTyping_coarse_FINAL (expect ~1.0; a low value => obs_names/label-key mismatch with "
          f"the coarse-typing adapter -> composition axis silently degrades to Untyped).", flush=True)
    return lab.astype(object).where(lab.notna(), "Untyped")


def shannon(counts, norm=True):
    p = np.asarray(counts, float); p = p[p > 0]
    if p.sum() <= 0 or len(p) < 2:
        return 0.0
    p = p / p.sum()
    H = float(-(p * np.log(p)).sum())
    return H / np.log(len(p)) if norm else H


def morans_I(z, W, W_sum):
    n = len(z); den = float(z @ z)
    if den <= 0 or W_sum <= 0:
        return np.nan
    return (n / W_sum) * (float(z @ (W @ z)) / den)


def _hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except (QhullError, Exception):
        return 0.0


def compute(h5, sample=None):
    A = ps_io.load_indep(h5, obs_cols=OBS_COLS)
    sample = sample or str(A.obs["sample"].iloc[0])
    status = str(A.obs["status"].iloc[0])
    # _FINAL: uniform segmentation -> read the provenance tag from obs (fallback SEG_PIPELINE).
    seg_version = (str(A.obs["seg_pipeline"].iloc[0])
                   if "seg_pipeline" in A.obs.columns and len(A.obs["seg_pipeline"])
                   else cfg.SEG_PIPELINE)
    n = A.n_obs
    obs = A.obs
    xy = np.asarray(A.obsm["spatial_orig"], float)
    latent = np.asarray(A.obsm["novae_latent"], float) if "novae_latent" in A.obsm else None

    ctype = load_celltypes(sample, list(A.obs_names))
    obs = obs.assign(cell_type=pd.Categorical(ctype.values))
    typed_frac_sample = float((ctype.values != "Untyped").mean())
    sample_untyped_pct = 100 * (1 - typed_frac_sample)

    ismn = _to_bool(obs["is_MN"]) if "is_MN" in obs else np.zeros(n, bool)
    sample_mn_frac = float(ismn.mean())
    # is_MN is a REMOVE-protector only; when Marcel's _final annotation is still pending it is
    # all-False and MN protection is simply inactive (every other verdict is unaffected). Surface
    # that state explicitly (log + meta + PDFs) rather than letting it pass silently.
    mn_protection_active = bool(ismn.any())
    is_MN_source = _read_uns_str(h5, "is_MN_source") or (
        "present" if mn_protection_active else "none (annotation pending)")
    if mn_protection_active:
        print(f"[{sample}] is_MN present: {int(ismn.sum())} MNs (source: {is_MN_source}) "
              f"-> MN-protection ACTIVE (MN-enriched domains protected from REMOVE).", flush=True)
    else:
        print(f"[{sample}] is_MN all-False (source: {is_MN_source}) -> MN-protection INACTIVE: "
              f"no domain is MN-protection-saved; ALL OTHER verdicts unaffected. Re-run after "
              f"Marcel's _final is_MN annotation lands to activate MN protection.", flush=True)

    negcols = ["control_probe_counts", "control_codeword_counts",
               "genomic_control_counts", "deprecated_codeword_counts"]
    negsum = np.zeros(n, float)
    for c in negcols:
        if c in obs:
            negsum += np.asarray(obs[c], float)
    tc = np.asarray(obs["transcript_counts"], float)
    tot = np.asarray(obs["total_counts"], float)
    negfrac = negsum / np.maximum(tot + negsum, 1.0)
    section_neg_median = float(np.median(negfrac)) or 1e-9
    section_tx_median = float(np.median(tc)) or 1e-9
    carea = np.asarray(obs["cell_area"], float)
    density = tc / np.maximum(carea, 1e-6)
    section_density_median = float(np.median(density)) or 1e-9

    # ---- shared spatial k-NN graph ----
    k = min(KNN, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    section_1nn = float(np.median(dist[:, 1])) or 1e-9
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T); W_sum = float(W.sum())
    dom = obs[RES].astype(str).values
    same = (dom[idx] == dom[:, None])
    purity_cell = same[:, :PAS_K].mean(1)               # use first PAS_K neighbours
    pas_cell = purity_cell < 0.5                         # SpaGCN abnormal spot

    # ---- latent silhouette (subsample) ----
    latent_sil = {}
    if latent is not None:
        keep = dom != "unassigned"
        gsub = np.where(keep)[0]
        if len(np.unique(dom[keep])) >= 2 and len(gsub) > 10:
            rng = np.random.default_rng(0)
            sel = gsub if len(gsub) <= SIL_SUB else rng.choice(gsub, SIL_SUB, replace=False)
            try:
                ss = silhouette_samples(latent[sel], dom[sel])
                sdf = pd.DataFrame({"d": dom[sel], "s": ss})
                latent_sil = sdf.groupby("d")["s"].mean().to_dict()
            except Exception as e:
                print(f"[{sample}] silhouette failed: {e}", flush=True)

    # ---- marker identity ----
    ident = {}
    rg = None                                           # ranked-genes table (reused by figures)
    keep = dom != "unassigned"
    groups = sorted(set(dom[keep]))
    if len(groups) >= 2:
        sub = A[keep].copy()
        sub.obs["_d"] = pd.Categorical(dom[keep], categories=groups)
        try:
            sc.tl.rank_genes_groups(sub, "_d", method="wilcoxon", n_genes=sub.n_vars, use_raw=False)
            rg = sc.get.rank_genes_groups_df(sub, group=None)
            for g in groups:
                gg = rg[rg.group == g].sort_values("scores", ascending=False)
                pos = gg[(gg.logfoldchanges > 1) & (gg.pvals_adj < 1e-10)]
                top10 = gg.head(10)
                ident[g] = {
                    "top_marker": str(gg.iloc[0]["names"]),
                    "top_logFC": round(float(gg.iloc[0]["logfoldchanges"]), 2),
                    "n_strong_markers": int(len(pos)),
                    "strong_markers": list(pos["names"].head(8).astype(str)),
                    "top10_pos": int((top10.logfoldchanges > 0.5).sum()),
                    "top10_neg": int((top10.logfoldchanges < -0.5).sum()),
                }
        except Exception as e:
            print(f"[{sample}] rank_genes failed: {e}", flush=True)

    section_hull = _hull_area(xy)
    recs = []
    for d in pd.unique(dom):
        m = dom == d
        nc = int(m.sum()); frac = nc / n
        pdd = obs.loc[m]
        tcd = tc[m]
        rec = {
            "sample": sample, "status": status, "level": spinal_level(sample),
            "seg_version": seg_version,
            "resolution": RES, "domain": d, "n_cells": nc, "pct_cells": round(100 * frac, 3),
            # transcript QC
            "median_transcripts": float(np.median(tcd)), "mean_transcripts": float(tcd.mean()),
            "iqr_transcripts": float(np.subtract(*np.percentile(tcd, [75, 25]))),
            "count_ratio": round(float(np.median(tcd) / section_tx_median), 3),
            "pct_lt20_tx": round(100 * float((tcd < 20).mean()), 2),
            "median_total_counts": float(np.median(tot[m])),
            "median_n_counts": float(np.median(np.asarray(pdd["n_counts"], float))) if "n_counts" in pdd else np.nan,
            "median_tx_density": round(float(np.median(density[m])), 4),
            "density_ratio": round(float(np.median(density[m]) / section_density_median), 3),
            # morphology
            "median_cell_area": float(np.median(carea[m])),
            "median_nucleus_area": float(np.median(np.asarray(pdd["nucleus_area"], float))),
            "pct_no_nucleus": round(100 * float((np.asarray(pdd["nucleus_count"], float) == 0).mean()), 2),
            "median_nuclear_frac": round(float(np.median(
                np.asarray(pdd["nucleus_area"], float) / np.maximum(carea[m], 1e-6))), 3),
            # negative control
            "mean_negctrl_frac": round(float(negfrac[m].mean()), 6),
            "negctrl_ratio": round(float(negfrac[m].mean() / section_neg_median), 2),
            # graph
            "neighborhood_valid_frac": round(float(_to_bool(pdd["neighborhood_valid"]).mean()), 4)
                if "neighborhood_valid" in pdd else np.nan,
            # MN (unvalidated protective prior)
            "n_MN": int(ismn[m].sum()), "pct_MN": round(100 * float(ismn[m].mean()), 3),
        }
        # composition
        typed = obs.loc[m, "cell_type"].astype(str).values
        nonun = typed[typed != "Untyped"]
        rec["pct_untyped"] = round(100 * float((typed == "Untyped").mean()), 2)
        rec["untyped_ratio"] = round(float(rec["pct_untyped"] / sample_untyped_pct), 2) if sample_untyped_pct > 0 else np.nan
        if len(nonun):
            vc = pd.Series(nonun).value_counts()
            rec["dominant_celltype"] = str(vc.index[0])
            rec["dominant_celltype_frac"] = round(float(vc.iloc[0] / len(nonun)), 3)
            rec["celltype_entropy"] = round(shannon(vc.values), 3)
        else:
            rec["dominant_celltype"] = "Untyped"; rec["dominant_celltype_frac"] = 0.0; rec["celltype_entropy"] = np.nan
        rec["celltype_fracs"] = {kk: round(float(vv), 4) for kk, vv in
                                 (obs.loc[m, "cell_type"].value_counts() / nc).to_dict().items()}
        # identity
        idd = ident.get(d, {})
        rec["top_marker"] = idd.get("top_marker"); rec["top_logFC"] = idd.get("top_logFC")
        rec["n_strong_markers"] = idd.get("n_strong_markers", 0)
        rec["strong_markers"] = idd.get("strong_markers", [])
        rec["top10_pos"] = idd.get("top10_pos", 0); rec["top10_neg"] = idd.get("top10_neg", 0)
        rec["latent_silhouette"] = round(float(latent_sil.get(d, np.nan)), 4) if d in latent_sil else np.nan
        # ---- spatial coherence ----
        rec["nbhd_purity"] = round(float(purity_cell[m].mean()), 4)
        rec["purity_enrichment"] = round(float(purity_cell[m].mean() / frac), 2) if frac > 0 else np.nan
        rec["pas"] = round(float(pas_cell[m].mean()), 4)
        ind = m.astype(float); z = ind - ind.mean()
        rec["morans_I"] = round(float(morans_I(z, W, W_sum)), 4)
        if nc >= 3:
            sub_adj = W[m][:, m]
            ncomp, lab = connected_components(sub_adj, directed=False)
            sizes = np.bincount(lab)
            rec["largest_cc_frac"] = round(float(sizes.max() / nc), 4)
            rec["n_components"] = int(ncomp); rec["comp_per_1000"] = round(float(ncomp / nc * 1000), 2)
            # CHAOS: mean within-domain 1-NN micron distance
            dm = xy[m]
            nnd = NearestNeighbors(n_neighbors=2).fit(dm).kneighbors(dm)[0][:, 1]
            rec["chaos"] = round(float(nnd.mean()), 2)
            rec["chaos_ratio"] = round(float(nnd.mean() / section_1nn), 2)
        else:
            rec.update({"largest_cc_frac": np.nan, "n_components": nc, "comp_per_1000": np.nan,
                        "chaos": np.nan, "chaos_ratio": np.nan})
        hull = _hull_area(xy[m])
        rec["hull_area_frac"] = round(float(hull / section_hull), 4) if section_hull > 0 else np.nan
        rec["spread_ratio"] = round(float(rec["hull_area_frac"] / frac), 3) if frac > 0 and hull > 0 else np.nan
        # cross-resolution nesting entropy (reproducibility)
        for other in ("novae_domains_n4", "novae_domains_n10"):
            if other in obs:
                oc = obs.loc[m, other].astype(str).value_counts()
                rec[f"nest_entropy_{other.split('_')[-1]}"] = round(shannon(oc.values), 3)
        recs.append(rec)

    df = pd.DataFrame(recs)
    df = add_score_and_verdict(df, sample_untyped_pct, sample_mn_frac)
    meta = {"sample": sample, "status": status, "level": spinal_level(sample),
            "seg_version": seg_version,
            "n_cells": n, "n_genes": int(A.n_vars), "typed_frac_sample": round(typed_frac_sample, 4),
            "sample_untyped_pct": round(sample_untyped_pct, 2), "sample_mn_frac": round(sample_mn_frac, 5),
            "mn_protection_active": mn_protection_active, "is_MN_source": is_MN_source,
            "section_tx_median": round(section_tx_median, 1), "resolution": RES, "knn": k}
    return A, obs, df, meta, rg


def add_score_and_verdict(df, sample_untyped_pct, sample_mn_frac):
    d = df.copy()
    # absolute-scaled 0..1 sub-scores oriented higher = more smear-like (for the ranked bar)
    s_purity = 1 - np.clip(d["nbhd_purity"].values, 0, 1)
    s_moran = 1 - np.clip(d["morans_I"].values, 0, 1)
    s_cc = 1 - np.clip(d["largest_cc_frac"].values.astype(float), 0, 1)
    s_pas = np.clip(d["pas"].values, 0, 1)
    s_chaos = np.clip((d["chaos_ratio"].values.astype(float) - 1.0) / 3.0, 0, 1)
    s_spread = np.clip((d["spread_ratio"].values.astype(float) - 1.0) / 4.0, 0, 1)
    s_count = np.clip((1.0 - d["count_ratio"].values) / 0.6, 0, 1)
    s_neg = np.clip((d["negctrl_ratio"].values - 1.0) / 4.0, 0, 1)
    s_untyped = np.clip(d["untyped_ratio"].values.astype(float) / 2.0, 0, 1)
    s_ident = np.clip((2 - d["n_strong_markers"].values) / 2.0, 0, 1)
    s_entropy = np.clip(d["celltype_entropy"].values.astype(float), 0, 1)
    s_sil = np.clip((0.1 - d["latent_silhouette"].values.astype(float)) / 0.3, 0, 1)
    coherence = np.nanmean(np.vstack([s_purity, s_moran, s_cc, s_pas, s_chaos, s_spread]), 0)
    identity = np.nanmean(np.vstack([s_ident, s_sil]), 0)
    qc = np.nanmean(np.vstack([s_count, s_neg]), 0)
    comp = np.nanmean(np.vstack([s_untyped, s_entropy]), 0)
    score = 0.45 * coherence + 0.25 * identity + 0.15 * qc + 0.15 * comp
    d["smear_score"] = np.round(np.nan_to_num(score, nan=0.0), 3)
    d["sub_coherence"] = np.round(coherence, 3); d["sub_identity"] = np.round(identity, 3)
    d["sub_qc"] = np.round(qc, 3); d["sub_composition"] = np.round(comp, 3)

    def verdict(r):
        dom = str(r["domain"])
        if dom == "unassigned":
            return "REMOVE", "Novae under-connected leftovers (graph-invalid)"
        moran = r["morans_I"] if pd.notna(r["morans_I"]) else 0.0
        cc = r["largest_cc_frac"] if pd.notna(r["largest_cc_frac"]) else 0.0
        pas = r["pas"] if pd.notna(r["pas"]) else 1.0
        spatial_incoherent = (moran < G_MORAN) or (cc < G_LARGEST_CC) or (pas > G_PAS)
        identity_fail = ((int(r["n_strong_markers"]) <= ID_STRONG) or
                         (pd.notna(r["negctrl_ratio"]) and r["negctrl_ratio"] > ID_NEG_RATIO) or
                         (r["count_ratio"] < ID_COUNT_RATIO) or
                         (pd.notna(r["untyped_ratio"]) and r["untyped_ratio"] > ID_UNTYPED_MULT))
        protected = (int(r["n_strong_markers"]) >= PROTECT_STRONG) or \
                    (sample_mn_frac > 0 and r["pct_MN"] / 100.0 > PROTECT_MN_MULT * sample_mn_frac and r["pct_MN"] > 1)
        # reason bits
        bits = []
        if moran < G_MORAN: bits.append(f"Moran I {moran:.2f}")
        if cc < G_LARGEST_CC: bits.append(f"largest-CC {cc:.2f}")
        if pas > G_PAS: bits.append(f"PAS {pas:.2f}")
        if int(r["n_strong_markers"]) <= ID_STRONG: bits.append(f"{int(r['n_strong_markers'])} strong markers")
        if pd.notna(r["negctrl_ratio"]) and r["negctrl_ratio"] > ID_NEG_RATIO: bits.append(f"neg-ctrl x{r['negctrl_ratio']:.1f}")
        if r["count_ratio"] < ID_COUNT_RATIO: bits.append(f"tx x{r['count_ratio']:.2f}")
        if pd.notna(r["untyped_ratio"]) and r["untyped_ratio"] > ID_UNTYPED_MULT: bits.append(f"untyped x{r['untyped_ratio']:.1f}")
        if spatial_incoherent and identity_fail:
            v = "REVIEW" if protected else "REMOVE"
            if protected:
                bits.append(f"PROTECTED ({int(r['n_strong_markers'])} strong markers / MN-enriched)")
        elif spatial_incoherent or identity_fail:
            v = "REVIEW"
        else:
            v = "KEEP"
            bits = ["coherent territory with a positive identity"]
        if int(r["n_cells"]) < SMALL_N and v == "REMOVE":
            v = "REVIEW"; bits.append(f"n<{SMALL_N} low-power")
        return v, "; ".join(bits)

    vr = d.apply(lambda r: verdict(r), axis=1)
    d["verdict"] = [x[0] for x in vr]
    d["reason"] = [x[1] for x in vr]
    return d


def write_outputs(outroot, df, meta, rg):
    """Write stats JSON + domain-metrics/decisions/markers CSVs; enrich + return meta.
    Shared by the standalone main() and the dpqc_persample driver (so both agree)."""
    sample = meta["sample"]
    for sd in ("stats", "tables"):
        os.makedirs(os.path.join(outroot, sd), exist_ok=True)
    df.drop(columns=["celltype_fracs", "strong_markers"]).to_csv(
        os.path.join(outroot, "tables", f"{sample}__domain_metrics.csv"), index=False)
    if rg is not None:
        rg.to_csv(os.path.join(outroot, "tables", f"{sample}__markers_n6.csv"), index=False)
    df[["sample", "domain", "n_cells", "pct_cells", "verdict", "smear_score", "reason",
        "dominant_celltype", "top_marker", "n_strong_markers"]].to_csv(
        os.path.join(outroot, "tables", f"{sample}__decisions.csv"), index=False)
    meta["domains_detail"] = json.loads(df.to_json(orient="records"))
    meta["verdicts"] = {str(r["domain"]): r["verdict"] for _, r in df.iterrows()}
    for v in ("KEEP", "REVIEW", "REMOVE"):
        meta[f"n_{v.lower()}"] = int((df["verdict"] == v).sum())
    meta["pct_cells_remove"] = round(float(100 * df.loc[df.verdict == "REMOVE", "n_cells"].sum() / meta["n_cells"]), 2)
    with open(os.path.join(outroot, "stats", f"{sample}__dpqc.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(h5, outroot, sample=None):
    A, obs, df, meta, rg = compute(h5, sample=sample)
    sample = meta["sample"]
    meta = write_outputs(outroot, df, meta, rg)
    print(f"[{sample}] {meta['status']}/{meta['level']}/{meta['seg_version']} "
          f"domains={len(df)} KEEP={meta['n_keep']} REVIEW={meta['n_review']} REMOVE={meta['n_remove']} "
          f"cells_removed={meta['pct_cells_remove']}% typed={meta['typed_frac_sample']:.2f}", flush=True)
    cols = ["domain", "n_cells", "pct_cells", "median_transcripts", "count_ratio", "nbhd_purity",
            "morans_I", "pas", "largest_cc_frac", "chaos_ratio", "spread_ratio", "pct_untyped",
            "n_strong_markers", "latent_silhouette", "dominant_celltype", "smear_score", "verdict"]
    print(df[cols].to_string(index=False), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=None); ap.add_argument("--sample", default=None)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--persample-dir", default=str(cfg.NOVAE_INDEP_DIR / "per_sample"))
    ap.add_argument("--outroot", default=str(cfg.DUALPASS_QC_DIR))
    a = ap.parse_args()
    h5 = a.h5
    if h5 is None and a.idx is not None:
        h5 = sorted(glob.glob(os.path.join(a.persample_dir, "*niches_independent.h5ad")))[a.idx]
    if h5 is None and a.sample:
        hits = glob.glob(os.path.join(a.persample_dir, f"{a.sample}*niches_independent.h5ad"))
        h5 = hits[0] if hits else None
    if h5 is None:
        sys.exit("need --h5, --sample, or --idx")
    main(h5, a.outroot, sample=a.sample)
