# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/smear_gate.py
#
# A diagnostic that ranks pass-1 Leiden clusters as smear or debris candidates. It deletes
# nothing: DATA_DRIVEN_SMEAR is empty. Notebook cell 43 imports it by an absolute
# sys.path to /home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun, so the file has to live at
# that path on SCG or all 20 notebooks fail at cell 43.
# ========================================================================================

"""
Smear/debris DIAGNOSTIC for pass-1 Leiden clusters. Ranks candidates; deletes nothing.

Why it exists: the per-sample dual-pass notebooks ship with

    suspect         = ['4', '5']
    SMEAR_CLUSTERS  = ['4', '5']

carried over from SD03522_BG, and both cells say in their own comments that the
ids "almost certainly mean something else here". Leiden ids are not stable across
samples, so running the notebooks as-written deletes two arbitrary clusters per
sample.

What replaces it, and what does NOT:

  * Actual smear REMOVAL is done at the Novae-domain level, using the lab's
    already-validated, human-reviewed verdicts in
    dualpass_qc/extracted/tables/<SAMPLE>__decisions.csv. That is where the
    two-gate rule of dpqc_compute_FINAL.py is defined and where it is valid,
    because a Novae domain is a spatial territory.

  * This module only RANKS Leiden clusters as smear candidates for human review.
    The two-gate rule cannot simply be ported down to cluster level: it treats
    spatial incoherence as the smear signal, but a transcriptional cluster of
    astrocytes or oligodendrocytes is dispersed across the entire section by
    definition, so that test flags every real cell type. Ported naively it
    marked 44 of 46 clusters in SD01015_BG for removal.

  * The orientation is therefore inverted here: debris is a contiguous patch
    with no transcriptional identity, so high spatial CONCENTRATION plus a
    failing identity/QC axis is what raises a candidate. Clusters with >=3 strong
    markers or MN enrichment (>2x the sample fraction) are marked protected.

Gate constants are still those of dpqc_compute_FINAL.py so the two levels of QC
speak the same language.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# ---- gate constants, copied verbatim from dpqc_compute_FINAL.py -------------
G_MORAN         = 0.10   # spatial gate: below = incoherent
G_LARGEST_CC    = 0.20   # spatial gate
G_PAS           = 0.50   # spatial gate: >half of cells abnormal
ID_STRONG       = 1      # identity gate: <=1 strong specific positive marker
ID_NEG_RATIO    = 3.0    # QC gate: neg-control fraction > 3x sample median
ID_COUNT_RATIO  = 0.6    # QC gate: median tx < 0.6x sample median
PROTECT_STRONG  = 3      # protective: >=3 strong markers -> never REMOVE
PROTECT_MN_MULT = 2.0    # protective: MN fraction > 2x sample -> downgrade
SMALL_N         = 50     # low-power cluster -> cap at REVIEW

KNN    = 15
PAS_K  = 10
LFC_MIN, PADJ_MAX = 1.0, 1e-10


def _knn_graph(xy, k=KNN):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)
    return nn.kneighbors(xy, return_distance=False)[:, 1:]


def _morans_I(ind, idx):
    """Moran's I of the 0/1 cluster indicator over the spatial kNN graph."""
    x = ind.astype(float)
    xb = x - x.mean()
    if np.allclose(xb, 0):
        return np.nan
    num = (xb[:, None] * xb[idx]).sum()
    W = idx.size
    return float((len(x) / W) * num / (xb ** 2).sum())


def _largest_cc_frac(ind, idx):
    """Fraction of the cluster's cells sitting in its single largest spatial blob."""
    n = int(ind.sum())
    if n == 0:
        return np.nan
    pos = np.where(ind)[0]
    remap = -np.ones(len(ind), dtype=np.int64)
    remap[pos] = np.arange(n)
    rows, cols = [], []
    for i, p in enumerate(pos):
        for nb in idx[p]:
            if ind[nb]:
                rows.append(i); cols.append(remap[nb])
    if not rows:
        return 1.0 / n
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncc, lab = connected_components(A, directed=False)
    return float(np.bincount(lab).max() / n)


def _pas(ind, idx, k=PAS_K):
    """Proportion of abnormal spots: cells whose kNN majority disagrees with them."""
    n = int(ind.sum())
    if n == 0:
        return np.nan
    sub = idx[ind][:, :k]
    return float((ind[sub].mean(axis=1) < 0.5).mean())


def cluster_gate_table(sample, cluster_key="leiden_1.0", rank_key=None):
    """
    Build the per-cluster gate table and verdicts.

    sample    : AnnData, post transcript+density filter, pass-1 clustered
    rank_key  : key in sample.uns holding a rank_genes_groups run over cluster_key
    returns   : DataFrame indexed by cluster id
    """
    obs = sample.obs
    xy = np.column_stack([obs["x_centroid"].values, obs["y_centroid"].values])
    idx = _knn_graph(xy)

    # sample-level baselines the ratios are measured against
    med_counts = float(obs["total_counts"].median())
    neg_cols = [c for c in ["control_probe_counts", "control_codeword_counts",
                            "genomic_control_counts", "unassigned_codeword_counts",
                            "deprecated_codeword_counts"] if c in obs.columns]
    if neg_cols:
        neg_tot = obs[neg_cols].sum(axis=1)
        neg_frac = neg_tot / (obs["total_counts"] + neg_tot).clip(lower=1)
        med_neg = float(neg_frac.median())
    else:
        neg_frac, med_neg = None, np.nan

    has_mn = "is_MN" in obs.columns
    sample_mn_frac = float(pd.Series(obs["is_MN"]).astype("boolean").fillna(False).mean()) if has_mn else 0.0

    # strong specific positive markers per cluster
    strong = {}
    if rank_key and rank_key in sample.uns:
        import scanpy as sc
        for cl in obs[cluster_key].cat.categories:
            try:
                df = sc.get.rank_genes_groups_df(sample, group=str(cl), key=rank_key)
                strong[str(cl)] = int(((df["logfoldchanges"] > LFC_MIN) &
                                       (df["pvals_adj"] < PADJ_MAX)).sum())
            except Exception:
                strong[str(cl)] = 0

    rows = []
    for cl in obs[cluster_key].cat.categories:
        ind = (obs[cluster_key] == cl).values
        n = int(ind.sum())
        r = {
            "cluster": str(cl),
            "n_cells": n,
            "pct_of_total": round(100 * n / len(obs), 2),
            "median_counts": float(obs.loc[ind, "total_counts"].median()),
            "median_genes": float(obs.loc[ind, "n_genes_by_counts"].median()),
            "morans_I": _morans_I(ind, idx),
            "largest_cc_frac": _largest_cc_frac(ind, idx),
            "pas": _pas(ind, idx),
            "n_strong_markers": strong.get(str(cl), 0),
        }
        r["count_ratio"] = r["median_counts"] / med_counts if med_counts else np.nan
        r["negctrl_ratio"] = (float(neg_frac[ind].median()) / med_neg
                              if neg_frac is not None and med_neg > 0 else np.nan)
        r["pct_MN"] = (100 * float(pd.Series(obs.loc[ind, "is_MN"]).astype("boolean").fillna(False).mean())
                       if has_mn else 0.0)
        rows.append(r)

    df = pd.DataFrame(rows).set_index("cluster", drop=False)
    # ---- Leiden-level scoring -------------------------------------------------
    # NOTE the orientation change against dpqc_compute_FINAL.py. There the unit is
    # a Novae DOMAIN -- a spatial territory -- so spatial INCOHERENCE is the smear
    # signal. Here the unit is a transcriptional cluster, and real cell types
    # (astrocytes, oligodendrocytes, microglia) are dispersed across the whole
    # section by definition; scoring those on spatial coherence flags every one of
    # them. Debris is the opposite: a physically contiguous patch, band or edge
    # carrying no transcriptional identity. So the spatial term is inverted here --
    # high CONCENTRATION together with absent identity is what marks a candidate.
    # Nothing is deleted on this score; it ranks clusters for human review.
    def score(r):
        cc = r["largest_cc_frac"] if pd.notna(r["largest_cc_frac"]) else 0.0
        s_conc  = float(np.clip(cc, 0, 1))
        s_count = float(np.clip((1.0 - (r["count_ratio"] or 0)) / 0.6, 0, 1))
        s_ident = float(np.clip((2 - int(r["n_strong_markers"])) / 2.0, 0, 1))
        s_neg   = 0.0
        if pd.notna(r["negctrl_ratio"]):
            s_neg = float(np.clip((r["negctrl_ratio"] - 1.0) / 4.0, 0, 1))
        return round(0.35*s_conc + 0.30*s_count + 0.25*s_ident + 0.10*s_neg, 3)

    def flag(r):
        identity_fail = ((int(r["n_strong_markers"]) <= ID_STRONG) or
                         (pd.notna(r["negctrl_ratio"]) and r["negctrl_ratio"] > ID_NEG_RATIO) or
                         ((r["count_ratio"] or 0) < ID_COUNT_RATIO))
        cc = r["largest_cc_frac"] if pd.notna(r["largest_cc_frac"]) else 0.0
        concentrated = cc > 0.50
        protected = (int(r["n_strong_markers"]) >= PROTECT_STRONG) or \
                    (sample_mn_frac > 0 and r["pct_MN"]/100.0 > PROTECT_MN_MULT*sample_mn_frac
                     and r["pct_MN"] > 1)
        if int(r["n_cells"]) < SMALL_N:
            return "low-power"
        if identity_fail and concentrated:
            return "protected" if protected else "CANDIDATE"
        return "ok"

    def why(r):
        bits = []
        if (r["count_ratio"] or 0) < ID_COUNT_RATIO:
            bits.append(f"tx x{r['count_ratio']:.2f}")
        if int(r["n_strong_markers"]) <= ID_STRONG:
            bits.append(f"{int(r['n_strong_markers'])} strong markers")
        if pd.notna(r["negctrl_ratio"]) and r["negctrl_ratio"] > ID_NEG_RATIO:
            bits.append(f"neg-ctrl x{r['negctrl_ratio']:.1f}")
        cc = r["largest_cc_frac"] if pd.notna(r["largest_cc_frac"]) else 0.0
        if cc > 0.50:
            bits.append(f"contiguous patch (largest-CC {cc:.2f})")
        if r["pct_MN"] and sample_mn_frac > 0 and r["pct_MN"]/100.0 > PROTECT_MN_MULT*sample_mn_frac:
            bits.append("MN-enriched (protected)")
        return "; ".join(bits) or "unremarkable"

    df["smear_rank_score"] = df.apply(score, axis=1)
    df["flag"]   = df.apply(flag, axis=1)
    df["reason"] = df.apply(why, axis=1)
    df.attrs["sample_mn_frac"] = sample_mn_frac
    df.attrs["median_counts"] = med_counts
    return df
