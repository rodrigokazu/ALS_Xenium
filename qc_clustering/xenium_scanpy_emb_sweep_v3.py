#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | qc_clustering/xenium_scanpy_emb_sweep_v3.py
#
# An atlas-scale parameter sweep for anyone about to argue over embedding settings. Three
# parallel PCA branches (unscaled, scaled at max_value 10, and low-variance-filtered), each
# with its own elbow plot, silhouette scores, embeddings and object.
#
# The sweep crosses three neighbour counts, four PC counts, seven Leiden resolutions, four
# tSNE perplexities with two exaggeration settings, and four UMAP minimum distances with three
# spreads. It ends in a comparison CSV and a heatmap ranking every branch and combination.
#
# It lets you answer "why these parameters" with a table instead of an opinion. The cost is a
# very long job; the runner asks for 48 hours.
#
# Silhouette is computed on a subsample. Treat it as a ranking aid rather than a truth; the
# embeddings still need looking at.
# ========================================================================================

"""
t-SNE + UMAP + Leiden  —  ATLAS-LEVEL PARAMETER SWEEP  v3
==========================================================
Two parallel PCA branches:
  A) UNSCALED  — PCA on log-normalized data directly
  B) SCALED    — sc.pp.scale(max_value=10) then PCA
  C) FILTERED  — drop low-variance genes (std < threshold), no scaling, then PCA

Each branch gets its own elbow plot, silhouette scores, embeddings, and h5ad.
A grand comparison CSV + heatmap ranks all branches × neighbor combos × Leiden
resolutions so you can see definitively which approach works best.

Directory layout
────────────────
<out_root>/
  diagnostics/
    scaling_check_pre.png/.svg
    elbow__unscaled.png/.svg
    elbow__scaled.png/.svg
    elbow__filtered.png/.svg
    branch_comparison.png/.svg
  metrics/
    silhouette_summary.csv              ← all branches combined, ranked
    silhouette_heatmaps/
      silhouette__{branch}__{combo}.png/.svg
  plots/
    {branch}/
      {combo_tag}/
        tsne/  umap/
  h5ad/
    sweep__{branch}__{combo_tag}.h5ad
"""

import time
import inspect
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import sparse
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================
in_path = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
    "Hippocampal_APP/data/Hippocampal_APP_postQC.h5ad"
)

out_root = in_path.parent / "sweep_atlas_v3"
plots_root = out_root / "plots"
h5ad_root = out_root / "h5ad"
diag_root = out_root / "diagnostics"
metrics_root = out_root / "metrics"
heatmap_root = metrics_root / "silhouette_heatmaps"
for d in (plots_root, h5ad_root, diag_root, metrics_root, heatmap_root):
    d.mkdir(parents=True, exist_ok=True)

WRITE_H5AD = True
RANDOM_STATE = 0

# ============================================================
# BRANCH CONFIG
# ============================================================
# Each branch defines a preprocessing path before PCA.
# All three share the same downstream sweep grids.
BRANCHES = {
    "unscaled": {
        "description": "PCA on log-normalized data, no scaling",
        "scale": False,
        "filter_low_var": False,
    },
    "scaled": {
        "description": "sc.pp.scale(max_value=10) then PCA",
        "scale": True,
        "scale_max_value": 10,
        "filter_low_var": False,
    },
    "filtered": {
        "description": "Drop genes with std < threshold, no scaling, then PCA",
        "scale": False,
        "filter_low_var": True,
        "min_std_threshold": 0.05,
    },
}

# ============================================================
# SWEEP GRIDS
# ============================================================

# -- PCA --
N_PCS_COMPUTE = 50
PCA_USE_HIGHLY_VARIABLE = False
PCA_PRIMARY_SOLVER = "arpack"
PCA_FALLBACK_SOLVER = "auto"

# -- Neighbors (outer sweep) --
SWEEP_N_NEIGHBORS = [15, 30, 50]
SWEEP_N_PCS_NEIGHBORS = [15, 20, 30, 40]

# -- Leiden --
SWEEP_LEIDEN_RES = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
LEIDEN_FLAVOR = "igraph"
LEIDEN_N_ITER = 2
LEIDEN_DIRECTED = False

# -- t-SNE --
SWEEP_TSNE_PERPLEXITY = [30, 50, 100, 150]
SWEEP_TSNE_EARLY_EXAG = [12.0, 24.0]
TSNE_LEARNING_RATE = "auto"
TSNE_MAX_ITER = 1500
TSNE_SUBSAMPLE_N = None

# -- UMAP --
SWEEP_UMAP_MIN_DIST = [0.05, 0.1, 0.3, 0.5]
SWEEP_UMAP_SPREAD = [1.0, 1.5, 2.0]
UMAP_N_COMPONENTS = 2
UMAP_SUBSAMPLE_N = None

# -- Silhouette --
SILHOUETTE_SUBSAMPLE = 50_000
SILHOUETTE_METRIC = "euclidean"

# -- Plotting --
POINT_SIZE = 0.5
LEGEND_LOC = "on data"
LEGEND_FONTSIZE = 4
DPI = 300
FIGSIZE = (7, 6)

DOT_LABEL_SOURCES = [
    "dot_label", "cell_type", "celltype", "CellType",
    "annotation", "Annotation", "pred_label",
    "predicted_label", "label", "cluster_label",
]
DOT_LABEL_MISSING_VALUE = "Unlabeled"


# ============================================================
# HELPERS
# ============================================================
def timed(step_name: str):
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n>>> {step_name} ...", flush=True)
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            tag = "done" if exc is None else f"FAILED: {exc}"
            print(f"<<< {step_name} {tag} in {dt:.1f}s", flush=True)
    return _T()


def ensure_csr(adata):
    if sparse.issparse(adata.X) and not (
        sparse.isspmatrix_csr(adata.X) or sparse.isspmatrix_csc(adata.X)
    ):
        with timed("Convert sparse X -> CSR"):
            adata.X = adata.X.tocsr()


def save_gene_distribution_plot(adata, label, out_path):
    """Save gene mean/std distribution diagnostic."""
    if sparse.issparse(adata.X):
        rng = np.random.default_rng(0)
        idx = rng.choice(adata.n_obs, size=min(5000, adata.n_obs), replace=False)
        X = adata.X[idx].toarray()
    else:
        X = adata.X[:min(5000, adata.n_obs)]

    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(means, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", ls="--", lw=1.5)
    axes[0].set_title(f"{label} — Gene means (median={np.nanmedian(means):.3f})")
    axes[0].set_xlabel("Mean expression per gene")
    axes[1].hist(stds, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(1, color="red", ls="--", lw=1.5)
    axes[1].set_title(f"{label} — Gene stds (median={np.nanmedian(stds[stds>0]):.3f})")
    axes[1].set_xlabel("Std per gene")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_path}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_pca_with_fallback(adata, n_comps, random_state):
    kw = dict(
        n_comps=int(min(n_comps, adata.n_vars - 1)),
        svd_solver=PCA_PRIMARY_SOLVER,
        random_state=int(random_state),
        use_highly_variable=False,
    )
    try:
        sc.tl.pca(adata, **kw)
    except Exception as e:
        print(f"PCA primary solver failed ({e}); falling back.", flush=True)
        kw["svd_solver"] = PCA_FALLBACK_SOLVER
        sc.tl.pca(adata, **kw)


def save_elbow_plot(adata, branch_name, sweep_pcs):
    """Save elbow plot and return cumulative variance dict."""
    var_ratio = adata.uns["pca"]["variance_ratio"]
    cumvar = np.cumsum(var_ratio)
    n = len(var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, n + 1), var_ratio, "o-", ms=3, lw=1.2)
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Variance ratio")
    axes[0].set_title(f"Scree Plot — {branch_name}")
    for npc in sweep_pcs:
        if npc <= n:
            axes[0].axvline(npc, color="red", ls="--", alpha=0.5, lw=1)
            axes[0].text(npc + 0.5, var_ratio[0] * 0.9, f"n={npc}", fontsize=8, color="red")

    axes[1].plot(range(1, n + 1), cumvar, "o-", ms=3, lw=1.2)
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title(f"Cumulative Variance — {branch_name}")
    axes[1].axhline(0.90, color="gray", ls=":", lw=1, label="90%")
    axes[1].axhline(0.95, color="gray", ls="--", lw=1, label="95%")
    for npc in sweep_pcs:
        if npc <= n:
            axes[1].axvline(npc, color="red", ls="--", alpha=0.5, lw=1)
            axes[1].text(
                npc + 0.5, cumvar[-1] * 0.5,
                f"n={npc}\n({cumvar[min(npc,n)-1]:.1%})", fontsize=8, color="red",
            )
    axes[1].legend()
    fig.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(diag_root / f"elbow__{branch_name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    info = {}
    for npc in sorted(set(sweep_pcs)):
        if npc <= n:
            info[npc] = cumvar[npc - 1]
            print(f"  [{branch_name}] PC 1-{npc}: {cumvar[npc-1]:.1%} variance explained", flush=True)
    return info


def drop_problematic_pca_params(adata):
    pca = adata.uns.get("pca", None)
    if isinstance(pca, dict):
        params = pca.get("params", None)
        if isinstance(params, dict) and "mask_var" in params:
            del params["mask_var"]
            print("[safety] Removed pca mask_var", flush=True)


def ensure_categorical(adata, col):
    if col in adata.obs and not pd.api.types.is_numeric_dtype(adata.obs[col].dtype):
        adata.obs[col] = adata.obs[col].astype("category")


def ensure_dot_label(adata, sources, out_col="dot_label", missing_value="Unlabeled"):
    if out_col in adata.obs:
        return
    present = [c for c in sources if c in adata.obs.columns]
    if not present:
        adata.obs[out_col] = pd.Categorical([missing_value] * adata.n_obs)
        return
    label = adata.obs[present[0]].astype("string").copy()
    for c in present[1:]:
        s = adata.obs[c].astype("string")
        label = label.where(~label.isna() & (label != "nan"), s)
    label = label.fillna(missing_value)
    adata.obs[out_col] = pd.Categorical(label.astype(str))
    print(f"[dot_label] Created from: {present}", flush=True)


def tsne_kwargs_compat(**kwargs):
    sig = inspect.signature(sc.tl.tsne)
    allowed = set(sig.parameters.keys())
    out = {}
    for k, v in kwargs.items():
        if k in allowed:
            out[k] = v
    if "max_iter" in allowed and "max_iter" not in out and "n_iter" in kwargs:
        out["max_iter"] = kwargs["n_iter"]
    if "n_iter" in allowed and "n_iter" not in out and "max_iter" in kwargs:
        out["n_iter"] = kwargs["max_iter"]
    return out


def subsample_adata(adata, n, random_state):
    if n is None or adata.n_obs <= n:
        return adata, None
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(adata.n_obs, size=int(n), replace=False)
    return adata[idx].copy(), idx


def compute_silhouette(adata, label_col, *, n_pcs, subsample_n=None, random_state=0):
    if label_col not in adata.obs:
        return np.nan, {}
    labels = adata.obs[label_col]
    n_unique = labels.nunique()
    if n_unique < 2 or n_unique >= adata.n_obs - 1:
        return np.nan, {}

    X = adata.obsm["X_pca"][:, :n_pcs]
    lab = labels.values

    if subsample_n is not None and adata.n_obs > subsample_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=subsample_n, replace=False)
        X = X[idx]
        lab = lab[idx]

    try:
        mean_sil = silhouette_score(X, lab, metric=SILHOUETTE_METRIC)
        sil_samp = silhouette_samples(X, lab, metric=SILHOUETTE_METRIC)
        per_cluster = {}
        for c in np.unique(lab):
            mask = lab == c
            per_cluster[str(c)] = float(np.mean(sil_samp[mask]))
        return float(mean_sil), per_cluster
    except Exception as e:
        print(f"  [silhouette] Failed for {label_col}: {e}", flush=True)
        return np.nan, {}


def save_silhouette_heatmap(sil_records, tag):
    if not sil_records:
        return
    df = pd.DataFrame(sil_records)
    df = df.sort_values("resolution")

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.4 + 1)))
    bars = ax.barh(
        [f"res={r:.1f} (k={k})" for r, k in zip(df["resolution"], df["n_clusters"])],
        df["silhouette_mean"],
        color=plt.cm.RdYlGn(Normalize(vmin=-0.1, vmax=0.6)(df["silhouette_mean"])),
        edgecolor="black", lw=0.5,
    )
    ax.set_xlabel("Mean Silhouette Score")
    ax.set_title(f"Silhouette — {tag}")
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, df["silhouette_mean"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
        )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(heatmap_root / f"silhouette__{tag}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_embedding_plot(
    adata, basis, color_cols, *, out_stem, title_prefix="",
    point_size=POINT_SIZE, dpi=DPI, figsize=FIGSIZE,
):
    plot_fn = sc.pl.tsne if basis == "tsne" else sc.pl.umap
    for col in color_cols:
        if col not in adata.obs:
            continue
        n_cats = adata.obs[col].nunique() if hasattr(adata.obs[col], "cat") else 0
        if n_cats > 20:
            leg, fs = "right margin", 4
        else:
            leg, fs = LEGEND_LOC, LEGEND_FONTSIZE

        title = f"{title_prefix}{col}" if title_prefix else col
        for ext in ("png", "svg"):
            fname = f"{out_stem}__{col}.{ext}"
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plot_fn(
                adata, color=col, ax=ax, show=False,
                size=point_size, legend_loc=leg, legend_fontsize=fs,
                title=title, frameon=False,
            )
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    print(f"  Saved plots: {out_stem}__*.png/svg", flush=True)


# ============================================================
# MAIN
# ============================================================
print(f"Input : {in_path}")
print(f"Output: {out_root}")

with timed("Read h5ad"):
    adata_raw = sc.read_h5ad(in_path)

ensure_csr(adata_raw)
ensure_dot_label(adata_raw, DOT_LABEL_SOURCES)
ensure_categorical(adata_raw, "dot_label")
ensure_categorical(adata_raw, "sample")

n_cells = adata_raw.n_obs
n_genes = adata_raw.n_vars
print(f"Dataset: {n_cells:,} cells, {n_genes:,} genes")

# Pre-normalization diagnostic
with timed("Pre-normalization diagnostic"):
    save_gene_distribution_plot(
        adata_raw, "Pre-normalization (raw filtered counts)",
        str(diag_root / "gene_dist__raw_input"),
    )

# ============================================================
# LOG-NORMALIZATION (essential — data is raw counts)
# ============================================================
with timed("Normalize total counts per cell (target_sum=1e4)"):
    sc.pp.normalize_total(adata_raw, target_sum=1e4)

with timed("Log1p transform"):
    sc.pp.log1p(adata_raw)

# Post-normalization diagnostic
with timed("Post-normalization diagnostic"):
    save_gene_distribution_plot(
        adata_raw, "Post log1p normalization",
        str(diag_root / "gene_dist__post_log1p"),
    )

# ============================================================
# BUILD BRANCHES
# ============================================================
branch_adatas = {}
branch_elbow_info = {}

for branch_name, cfg in BRANCHES.items():
    print(f"\n{'='*60}")
    print(f"BRANCH: {branch_name} — {cfg['description']}")
    print(f"{'='*60}")

    ad = adata_raw.copy()

    # --- Optional gene filtering ---
    if cfg.get("filter_low_var", False):
        threshold = cfg["min_std_threshold"]
        with timed(f"Filter low-variance genes (std < {threshold})"):
            if sparse.issparse(ad.X):
                # Compute per-gene std on a sample for speed
                rng = np.random.default_rng(0)
                idx = rng.choice(ad.n_obs, size=min(10000, ad.n_obs), replace=False)
                gene_stds = np.asarray(ad.X[idx].toarray().std(axis=0)).ravel()
            else:
                gene_stds = np.std(ad.X[:min(10000, ad.n_obs)], axis=0)

            keep_mask = gene_stds >= threshold
            n_drop = (~keep_mask).sum()
            n_keep = keep_mask.sum()
            ad = ad[:, keep_mask].copy()
            print(f"  Kept {n_keep}/{n_genes} genes, dropped {n_drop}", flush=True)

    # --- Optional scaling ---
    if cfg.get("scale", False):
        max_val = cfg.get("scale_max_value", 10)
        with timed(f"sc.pp.scale(max_value={max_val})"):
            sc.pp.scale(ad, max_value=max_val)

    # --- Diagnostic ---
    save_gene_distribution_plot(
        ad, f"Branch: {branch_name}",
        str(diag_root / f"gene_dist__{branch_name}"),
    )

    # --- PCA ---
    n_comps = min(N_PCS_COMPUTE, ad.n_vars - 1)
    with timed(f"PCA ({branch_name}, n_comps={n_comps})"):
        run_pca_with_fallback(ad, n_comps, RANDOM_STATE)
    drop_problematic_pca_params(ad)

    # --- Elbow ---
    valid_pcs = [p for p in SWEEP_N_PCS_NEIGHBORS if p <= ad.obsm["X_pca"].shape[1]]
    with timed(f"Elbow plot ({branch_name})"):
        info = save_elbow_plot(ad, branch_name, valid_pcs)
        branch_elbow_info[branch_name] = info

    branch_adatas[branch_name] = ad
    print(f"PCA shape: {ad.obsm['X_pca'].shape}")

# ============================================================
# BRANCH COMPARISON ELBOW PLOT
# ============================================================
with timed("Branch comparison elbow plot"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"unscaled": "#2196F3", "scaled": "#F44336", "filtered": "#4CAF50"}

    for bname, ad in branch_adatas.items():
        vr = ad.uns["pca"]["variance_ratio"]
        cv = np.cumsum(vr)
        c = colors.get(bname, "gray")
        axes[0].plot(range(1, len(vr) + 1), vr, "o-", ms=2, lw=1.2, label=bname, color=c)
        axes[1].plot(range(1, len(cv) + 1), cv, "o-", ms=2, lw=1.2, label=bname, color=c)

    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Variance ratio")
    axes[0].set_title("Scree Plot — Branch Comparison")
    axes[0].legend()
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("Cumulative Variance — Branch Comparison")
    axes[1].axhline(0.90, color="gray", ls=":", lw=1)
    axes[1].axhline(0.95, color="gray", ls="--", lw=1)
    axes[1].legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(diag_root / f"branch_comparison.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# SWEEP LOOP: branch × neighbors × params
# ============================================================
all_silhouette_records = []

for branch_name, ad_branch in branch_adatas.items():
    max_pcs = ad_branch.obsm["X_pca"].shape[1]
    valid_pcs = [p for p in SWEEP_N_PCS_NEIGHBORS if p <= max_pcs]

    for nn, npc in itertools.product(SWEEP_N_NEIGHBORS, valid_pcs):
        combo_tag = f"nn{nn}_pcs{npc}"
        full_tag = f"{branch_name}__{combo_tag}"
        print(f"\n{'#'*60}")
        print(f"# {full_tag}")
        print(f"{'#'*60}")

        adata = ad_branch.copy()

        # --- Neighbors ---
        with timed(f"Neighbors ({full_tag})"):
            sc.pp.neighbors(
                adata,
                n_neighbors=int(nn),
                n_pcs=int(npc),
                use_rep="X_pca",
                random_state=RANDOM_STATE,
            )

        # --- Dirs ---
        combo_plot_dir = plots_root / branch_name / combo_tag
        tsne_dir = combo_plot_dir / "tsne"
        umap_dir = combo_plot_dir / "umap"
        for d in (tsne_dir, umap_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ======================================================
        # LEIDEN + SILHOUETTE
        # ======================================================
        combo_sil_records = []

        for res in SWEEP_LEIDEN_RES:
            col_name = f"leiden_res{res}"
            with timed(f"Leiden res={res} ({full_tag})"):
                sc.tl.leiden(
                    adata,
                    resolution=float(res),
                    flavor=LEIDEN_FLAVOR,
                    n_iterations=LEIDEN_N_ITER,
                    directed=LEIDEN_DIRECTED,
                    key_added=col_name,
                )
            ensure_categorical(adata, col_name)
            n_clust = adata.obs[col_name].nunique()

            sil_mean, _ = compute_silhouette(
                adata, col_name,
                n_pcs=npc,
                subsample_n=SILHOUETTE_SUBSAMPLE,
                random_state=RANDOM_STATE,
            )
            rec = dict(
                branch=branch_name,
                n_neighbors=nn, n_pcs=npc,
                combo_tag=combo_tag, full_tag=full_tag,
                resolution=res, n_clusters=n_clust,
                silhouette_mean=sil_mean,
            )
            combo_sil_records.append(rec)
            all_silhouette_records.append(rec)
            print(f"  res={res}: k={n_clust}, sil={sil_mean:.4f}", flush=True)

        # Best leiden by silhouette
        valid_sil = [r for r in combo_sil_records if not np.isnan(r["silhouette_mean"])]
        if valid_sil:
            best = max(valid_sil, key=lambda r: r["silhouette_mean"])
            best_res = best["resolution"]
            print(
                f"  >> Best: res={best_res} (k={best['n_clusters']}, "
                f"sil={best['silhouette_mean']:.4f})",
                flush=True,
            )
        else:
            best_res = min(SWEEP_LEIDEN_RES, key=lambda r: abs(r - 0.5))
        adata.obs["leiden"] = adata.obs[f"leiden_res{best_res}"].copy()
        ensure_categorical(adata, "leiden")

        save_silhouette_heatmap(combo_sil_records, full_tag)

        # ======================================================
        # t-SNE SWEEP
        # ======================================================
        ad_tsne_base, tsne_idx = subsample_adata(adata, TSNE_SUBSAMPLE_N, RANDOM_STATE)

        for pp, ee in itertools.product(SWEEP_TSNE_PERPLEXITY, SWEEP_TSNE_EARLY_EXAG):
            tsne_tag = f"pp{pp}_ee{ee}"
            obsm_key = f"X_tsne_{tsne_tag}"
            ad_t = ad_tsne_base.copy()
            tsne_npcs = min(npc, max_pcs)

            with timed(f"t-SNE {tsne_tag} ({full_tag}, n={ad_t.n_obs})"):
                kw = tsne_kwargs_compat(
                    use_rep="X_pca",
                    n_pcs=int(tsne_npcs),
                    perplexity=float(pp),
                    early_exaggeration=float(ee),
                    learning_rate=TSNE_LEARNING_RATE,
                    max_iter=TSNE_MAX_ITER,
                    n_iter=TSNE_MAX_ITER,
                    random_state=RANDOM_STATE,
                )
                sc.tl.tsne(ad_t, **kw)

            if tsne_idx is not None:
                full_emb = np.full((adata.n_obs, 2), np.nan, dtype=np.float32)
                full_emb[tsne_idx] = ad_t.obsm["X_tsne"].astype(np.float32)
                adata.obsm[obsm_key] = full_emb
            else:
                adata.obsm[obsm_key] = ad_t.obsm["X_tsne"].astype(np.float32)

            plot_cols = ["dot_label", "sample"] + [
                f"leiden_res{r}" for r in SWEEP_LEIDEN_RES
            ]
            for col in plot_cols:
                if col not in ad_t.obs and col in adata.obs:
                    if tsne_idx is not None:
                        ad_t.obs[col] = adata.obs[col].values[tsne_idx]
                    else:
                        ad_t.obs[col] = adata.obs[col].values
                    ensure_categorical(ad_t, col)

            save_embedding_plot(
                ad_t, "tsne", plot_cols,
                out_stem=str(tsne_dir / f"tsne_{tsne_tag}"),
                title_prefix=f"{full_tag} tsne {tsne_tag} | ",
            )

        # ======================================================
        # UMAP SWEEP
        # ======================================================
        ad_umap_base, umap_idx = subsample_adata(adata, UMAP_SUBSAMPLE_N, RANDOM_STATE)

        for md, sp in itertools.product(SWEEP_UMAP_MIN_DIST, SWEEP_UMAP_SPREAD):
            umap_tag = f"md{md}_sp{sp}"
            obsm_key = f"X_umap_{umap_tag}"
            ad_u = ad_umap_base.copy()

            with timed(f"UMAP {umap_tag} ({full_tag}, n={ad_u.n_obs})"):
                sc.tl.umap(
                    ad_u,
                    min_dist=float(md),
                    spread=float(sp),
                    n_components=UMAP_N_COMPONENTS,
                    random_state=RANDOM_STATE,
                )

            if umap_idx is not None:
                full_emb = np.full((adata.n_obs, 2), np.nan, dtype=np.float32)
                full_emb[umap_idx] = ad_u.obsm["X_umap"].astype(np.float32)
                adata.obsm[obsm_key] = full_emb
            else:
                adata.obsm[obsm_key] = ad_u.obsm["X_umap"].astype(np.float32)

            plot_cols = ["dot_label", "sample"] + [
                f"leiden_res{r}" for r in SWEEP_LEIDEN_RES
            ]
            for col in plot_cols:
                if col not in ad_u.obs and col in adata.obs:
                    if umap_idx is not None:
                        ad_u.obs[col] = adata.obs[col].values[umap_idx]
                    else:
                        ad_u.obs[col] = adata.obs[col].values
                    ensure_categorical(ad_u, col)

            save_embedding_plot(
                ad_u, "umap", plot_cols,
                out_stem=str(umap_dir / f"umap_{umap_tag}"),
                title_prefix=f"{full_tag} umap {umap_tag} | ",
            )

        # ======================================================
        # WRITE H5AD
        # ======================================================
        drop_problematic_pca_params(adata)

        if WRITE_H5AD:
            h5_path = h5ad_root / f"sweep__{full_tag}.h5ad"
            with timed(f"Write {h5_path.name}"):
                adata.write_h5ad(h5_path, compression="gzip")

        del adata

# Free branch copies
del branch_adatas

# ============================================================
# GLOBAL SUMMARY
# ============================================================
with timed("Write global silhouette summary"):
    sil_df = pd.DataFrame(all_silhouette_records)
    sil_df = sil_df.sort_values("silhouette_mean", ascending=False)
    sil_csv = metrics_root / "silhouette_summary.csv"
    sil_df.to_csv(sil_csv, index=False)

    print(f"\n{'='*60}")
    print("TOP 15 PARAMETER COMBOS BY SILHOUETTE SCORE")
    print("=" * 60)
    for _, row in sil_df.head(15).iterrows():
        print(
            f"  {row['branch']:>10s}  nn={int(row['n_neighbors']):>3d}  "
            f"pcs={int(row['n_pcs']):>2d}  res={row['resolution']:.1f}  "
            f"k={int(row['n_clusters']):>3d}  sil={row['silhouette_mean']:.4f}"
        )

    # Grand heatmap per branch
    for bname in BRANCHES:
        sub = sil_df[sil_df["branch"] == bname]
        if sub.empty:
            continue
        best_per = sub.groupby(["n_neighbors", "n_pcs"])["silhouette_mean"].max().reset_index()
        pivot = best_per.pivot(index="n_neighbors", columns="n_pcs", values="silhouette_mean")

        fig, ax = plt.subplots(figsize=(8, 5))
        vmax = max(0.5, pivot.values[~np.isnan(pivot.values)].max()) if not pivot.empty else 0.5
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(r) for r in pivot.index])
        ax.set_xlabel("n_pcs")
        ax.set_ylabel("n_neighbors")
        ax.set_title(f"Best Silhouette — {bname}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="Silhouette")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(
                metrics_root / f"silhouette_grand__{bname}.{ext}",
                dpi=DPI, bbox_inches="tight",
            )
        plt.close(fig)

    # Cross-branch comparison: best silhouette per branch
    branch_best = sil_df.groupby("branch")["silhouette_mean"].agg(["max", "mean", "median"])
    print(f"\nBRANCH SUMMARY:")
    print(branch_best.to_string())

print(f"\n{'='*60}")
print(f"SWEEP COMPLETE")
print(f"  Diagnostics : {diag_root}")
print(f"  Metrics     : {metrics_root}")
print(f"  Plots       : {plots_root}")
print(f"  H5ADs       : {h5ad_root}")
print(f"{'='*60}")