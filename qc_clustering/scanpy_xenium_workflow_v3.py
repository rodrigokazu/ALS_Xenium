#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | qc_clustering/scanpy_xenium_workflow_v3.py
#
# The standalone scanpy QC and clustering workflow, predating the current pipeline and still
# the reference for how these objects were first processed.
#
# Uses Pearson residuals instead of the standard log-normalise route, with theta 100 and
# clipping at 10. That suits count data from a targeted panel better than a library-size
# normalisation designed for whole transcriptomes. Scaling and covariate regression are both
# off by default. The switches exist, but the defaults are what was actually run.
#
# Leiden at resolution 1.0 with the igraph flavour and two iterations, PCA at 50 components,
# 30 neighbours. tSNE subsamples to 100,000 cells because it does not scale to the full
# object.
#
# It carries a lot of defensive machinery for large sparse data: non-finite values replaced in
# place, empty cells and all-zero genes dropped, a PCA solver fallback from randomized to
# auto, and signature-compatible tSNE keyword handling across scanpy versions. That was all
# written in response to real crashes.
#
# No module docstring on this one; the constants at the top are the documentation.
# ========================================================================================

import time
import math
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse

# ============================================================
# CONFIG (single configuration only)
# ============================================================
in_path = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/DOT/(RK)SC_filtered_with_deconvolution_v4.h5ad"
)
out_path = in_path.with_name(in_path.stem + "__best_emb_single.h5ad")

plots_dir = out_path.with_suffix("").parent / (out_path.with_suffix("").name + "__plots")
plots_dir.mkdir(parents=True, exist_ok=True)

dpi = 300
figsize_default = (6, 5)

# PCA/graph
N_PCS = 50

# Base neighbors (for Leiden + base graph)
N_NEIGHBORS = 30
N_PCS_NEIGHBORS = 50

# Leiden
LEIDEN_RES = 1.0
LEIDEN_FLAVOR = "igraph"
LEIDEN_N_ITER = 2
LEIDEN_DIRECTED = False

# Pearson residual normalization
USE_PEARSON = True
PEARSON_THETA = 100.0
PEARSON_CLIP = 10.0

RANDOM_STATE = 0

# Optional regress out (OFF by default)
DO_REGRESS_OUT = False
REGRESS_CANDIDATES = [
    "log1p_total_counts",
    "cell_area",
    "nucleus_area",
    "nucleus_count",
]

# Scale policy (single choice)
# If you want "noscale": set DO_SCALE=False
# If you want "scale_zc": set DO_SCALE=True and SCALE_ZERO_CENTER=True
DO_SCALE = False
SCALE_ZERO_CENTER = True
SCALE_MAX_VALUE = 10

# PCA solver preference
PCA_PRIMARY_SOLVER = "randomized"
PCA_FALLBACK_SOLVER = "auto"

# Optional densify before PCA (NOT recommended for this dataset)
DENSIFY_FOR_PCA = False

# ----------------------------
# SINGLE "BEST" UMAP CONFIG (edit this)
# ----------------------------
BEST_UMAP = {
    "metric": "cosine",
    "min_dist": 0.05,
    "spread": 1.0,
    "random_state": 0,
    "n_neighbors": 15,
    "n_pcs": 50,
}

# ----------------------------
# SINGLE "BEST" t-SNE CONFIG (edit this)
# ----------------------------
BEST_TSNE = {
    "perplexity": 50,
    "early_exaggeration": 24.0,
    "learning_rate": "auto",
    "max_iter": 1000,
    "random_state": 0,
    "n_pcs": 50,
    "init": "pca",
}

# For full-data tSNE: set to None (NOT recommended for ~775k)
TSNE_SUBSAMPLE_N = 100_000

# ----------------------------
# Marker gene panels (optional)
# ----------------------------
MARKER_GENES = ["STMN2", "MNX1"]
PLOT_MARKER_PANELS = True
MARKER_POINT_SIZE = 1.0
MARKER_NCOLS = 3

# ----------------------------
# Dotplots (TOP genes per group)
# ----------------------------
MAKE_DOTPLOTS = True
TOP_N_GENES_PER_GROUP = 3
DOTPLOT_GROUPS = ["dot_label", "leiden"]
DOTPLOT_USE_LAYER = "counts"
DOTPLOT_LOG1P = True
DOTPLOT_STANDARD_SCALE = "var"
DOTPLOT_MAX_DOT = 0.6
DOTPLOT_DPI = 400
DOTPLOT_FIGSIZE = (12, 6)
DOTPLOT_MIN_PCT = 0.05

# ============================================================
# SCANPY FIGURE SETTINGS
# ============================================================
sc.settings.figdir = str(plots_dir)
sc.set_figure_params(dpi=dpi, dpi_save=dpi, fontsize=10, frameon=False, transparent=False)

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
            if exc is None:
                print(f"<<< {step_name} done in {dt:.1f}s", flush=True)
            else:
                print(f"<<< {step_name} FAILED after {dt:.1f}s: {exc}", flush=True)
    return _T()


def save_plot_scanpy(plot_fn, *, out_png: Path, figsize=None, dpi=300, **kwargs):
    kwargs.setdefault("show", False)
    if figsize is not None:
        plt.figure(figsize=figsize)
    plot_fn(**kwargs)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def ensure_layer_counts_no_copy(adata):
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X  # reference; no deep copy


def replace_nonfinite_in_X_inplace(adata, fill_value=0.0) -> int:
    X = adata.X
    if sparse.issparse(X):
        bad = ~np.isfinite(X.data)
        nbad = int(bad.sum())
        if nbad:
            X.data[bad] = fill_value
        return nbad
    arr = np.asarray(X)
    bad = ~np.isfinite(arr)
    nbad = int(bad.sum())
    if nbad:
        arr[bad] = fill_value
        adata.X = arr
    return nbad


def drop_empty_cells_and_allzero_genes(adata):
    X = adata.X
    if sparse.issparse(X):
        row_nnz = np.asarray(X.getnnz(axis=1)).ravel()
        col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
    else:
        Xd = np.asarray(X)
        row_nnz = np.sum(Xd != 0, axis=1)
        col_nnz = np.sum(Xd != 0, axis=0)

    keep_cells = row_nnz > 0
    keep_genes = col_nnz > 0

    n_drop_cells = int((~keep_cells).sum())
    n_drop_genes = int((~keep_genes).sum())

    if n_drop_cells or n_drop_genes:
        print(f"Dropping empty cells: {n_drop_cells}, all-zero genes: {n_drop_genes}", flush=True)
        adata._inplace_subset_obs(keep_cells)
        adata._inplace_subset_var(keep_genes)
    else:
        print("No empty cells / all-zero genes detected in current X.", flush=True)


def safe_regress_covariates(adata, candidates):
    covars = []
    for c in candidates:
        if c not in adata.obs.columns:
            continue
        s = adata.obs[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
            adata.obs[c] = s
        v = s.to_numpy()
        v_f = v[np.isfinite(v)]
        if v_f.size == 0:
            continue
        if np.nanstd(v_f) < 1e-8:
            continue
        covars.append(c)
    return covars


def run_pca_with_fallback(adata, n_comps: int, random_state: int):
    try:
        sc.tl.pca(
            adata,
            n_comps=int(n_comps),
            svd_solver=str(PCA_PRIMARY_SOLVER),
            random_state=int(random_state),
        )
    except Exception as e:
        print(f"PCA failed with svd_solver='{PCA_PRIMARY_SOLVER}': {e}", flush=True)
        print(f"Retrying PCA with svd_solver='{PCA_FALLBACK_SOLVER}' ...", flush=True)
        sc.tl.pca(
            adata,
            n_comps=int(n_comps),
            svd_solver=str(PCA_FALLBACK_SOLVER),
            random_state=int(random_state),
        )


def compute_neighbors_inplace(adata, *, n_neighbors: int, n_pcs: int, metric: str, random_state: int):
    sc.pp.neighbors(
        adata,
        n_neighbors=int(n_neighbors),
        n_pcs=int(n_pcs),
        use_rep="X_pca",
        metric=str(metric),
        random_state=int(random_state),
    )


def tsne_kwargs_compatible(**kwargs):
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


def _get_X_for_rank(adata, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Requested layer='{layer}' not found in adata.layers: {list(adata.layers.keys())}")
    return adata.layers[layer]


def top_genes_per_group_mean(
    adata,
    groupby: str,
    *,
    layer: str | None,
    n_top: int = 3,
    min_pct: float = 0.05,
):
    if groupby not in adata.obs:
        raise KeyError(f"groupby='{groupby}' not in adata.obs")

    X = _get_X_for_rank(adata, layer)
    if not sparse.issparse(X):
        X = np.asarray(X)

    groups = adata.obs[groupby].astype("category")
    cats = list(groups.cat.categories)

    gene_names = adata.var_names.to_numpy()
    out = {}

    for g in cats:
        idx = np.where(groups.to_numpy() == g)[0]
        if idx.size == 0:
            continue

        Xg = X[idx, :]
        if sparse.issparse(Xg):
            mean = np.asarray(Xg.mean(axis=0)).ravel()
            det = np.asarray((Xg > 0).mean(axis=0)).ravel()
        else:
            mean = np.mean(Xg, axis=0)
            det = np.mean(Xg > 0, axis=0)

        ok = det >= float(min_pct)
        if not np.any(ok):
            ok = np.ones_like(det, dtype=bool)

        order = np.lexsort((det[ok], mean[ok]))  # ascending
        top_idx = np.where(ok)[0][order][-int(n_top):][::-1]
        out[str(g)] = list(gene_names[top_idx])

    return out


def flatten_unique_gene_list(per_group: dict[str, list[str]]):
    seen = set()
    genes = []
    for _, gl in per_group.items():
        for g in gl:
            if g not in seen:
                seen.add(g)
                genes.append(g)
    return genes


def save_dotplot(adata, *, groupby: str, genes: list[str], out_png: Path):
    dp = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=groupby,
        layer=DOTPLOT_USE_LAYER if (DOTPLOT_USE_LAYER in adata.layers) else None,
        log=DOTPLOT_LOG1P,
        standard_scale=DOTPLOT_STANDARD_SCALE,
        dot_max=DOTPLOT_MAX_DOT,
        show=False,
        return_fig=True,
    )

    # DotPlot API differs across scanpy versions
    try:
        if hasattr(dp, "style"):
            dp = dp.style(figsize=DOTPLOT_FIGSIZE)
    except Exception as e:
        print(f"[WARN] DotPlot.style() failed: {e}", flush=True)

    if hasattr(dp, "savefig"):
        dp.savefig(out_png, dpi=DOTPLOT_DPI, bbox_inches="tight")
        plt.close("all")
        return

    fig = None
    if hasattr(dp, "figure"):
        fig = dp.figure
    elif hasattr(dp, "fig"):
        fig = dp.fig

    if fig is not None:
        try:
            fig.set_size_inches(*DOTPLOT_FIGSIZE)
        except Exception:
            pass
        fig.savefig(out_png, dpi=DOTPLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    plt.savefig(out_png, dpi=DOTPLOT_DPI, bbox_inches="tight")
    plt.close()


def save_embedding_gene_panels(adata, *, basis: str, out_prefix: str, genes: list[str], out_root: Path):
    for g in genes:
        if g not in adata.var_names:
            print(f"[WARN] Gene '{g}' not in var_names; skipping {basis} plot.", flush=True)
            continue
        out_png = out_root / f"{out_prefix}__{basis}__gene__{g}.png"
        save_plot_scanpy(
            sc.pl.embedding,
            out_png=out_png,
            adata=adata,
            basis=basis,
            color=g,
            layer="counts" if ("counts" in adata.layers) else None,
            size=MARKER_POINT_SIZE,
            legend_loc=None,
            color_map="viridis",
            figsize=(6, 5),
            dpi=dpi,
        )

    ok_genes = [g for g in genes if g in adata.var_names]
    if len(ok_genes) == 0:
        return
    out_png = out_root / f"{out_prefix}__{basis}__genes__{'_'.join(ok_genes)}.png"
    save_plot_scanpy(
        sc.pl.embedding,
        out_png=out_png,
        adata=adata,
        basis=basis,
        color=ok_genes,
        layer="counts" if ("counts" in adata.layers) else None,
        size=MARKER_POINT_SIZE,
        ncols=min(MARKER_NCOLS, len(ok_genes)),
        wspace=0.25,
        legend_loc="right margin",
        figsize=(7, 4 + 3 * math.ceil(len(ok_genes) / min(MARKER_NCOLS, len(ok_genes)))),
        dpi=dpi,
    )


def plot_selected_embeddings(adata, *, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["dot_label", "leiden", "sample"] if c in adata.obs]

    if "X_umap" in adata.obsm:
        save_plot_scanpy(
            sc.pl.umap,
            out_png=out_dir / f"{prefix}__umap__labels.png",
            adata=adata,
            color=cols if cols else None,
            ncols=3 if len(cols) >= 3 else 2,
            legend_loc="right margin",
            figsize=(10, 6),
            dpi=dpi,
        )
        if PLOT_MARKER_PANELS:
            save_embedding_gene_panels(
                adata,
                basis="umap",
                out_prefix=f"{prefix}__umap",
                genes=MARKER_GENES,
                out_root=out_dir,
            )

    if "X_tsne" in adata.obsm:
        save_plot_scanpy(
            sc.pl.tsne,
            out_png=out_dir / f"{prefix}__tsne__labels.png",
            adata=adata,
            color=cols if cols else None,
            ncols=3 if len(cols) >= 3 else 2,
            legend_loc="right margin",
            figsize=(10, 6),
            dpi=dpi,
        )
        if PLOT_MARKER_PANELS:
            save_embedding_gene_panels(
                adata,
                basis="tsne",
                out_prefix=f"{prefix}__tsne",
                genes=MARKER_GENES,
                out_root=out_dir,
            )


# ============================================================
# RUN
# ============================================================
print(f"Input : {in_path}", flush=True)
print(f"Output: {out_path}", flush=True)
print(f"Plots : {plots_dir} (PNG, dpi={dpi})", flush=True)

with timed("Read h5ad (CPU, scanpy)"):
    adata = sc.read_h5ad(in_path)

with timed("Ensure raw counts in layers['counts'] (no copy)"):
    ensure_layer_counts_no_copy(adata)

with timed("calculate_qc_metrics"):
    sc.pp.calculate_qc_metrics(adata, inplace=True)

if "log1p_total_counts" not in adata.obs.columns and "total_counts" in adata.obs.columns:
    adata.obs["log1p_total_counts"] = np.log1p(adata.obs["total_counts"].to_numpy())

# ----------------------------
# Normalization
# ----------------------------
if USE_PEARSON:
    with timed(f"Pearson residuals (theta={PEARSON_THETA}, clip={PEARSON_CLIP})"):
        adata.X = adata.layers["counts"]
        sc.experimental.pp.normalize_pearson_residuals(
            adata,
            theta=float(PEARSON_THETA),
            clip=float(PEARSON_CLIP),
        )
else:
    with timed("normalize_total(target_sum=1e4) + log1p"):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

with timed("Sanity: replace non-finite values after normalization"):
    nbad = replace_nonfinite_in_X_inplace(adata, fill_value=0.0)
    print(f"Non-finite in X fixed: {nbad}", flush=True)

with timed("Drop empty cells / all-zero genes in current X"):
    drop_empty_cells_and_allzero_genes(adata)

# ----------------------------
# Optional regress_out
# ----------------------------
if DO_REGRESS_OUT:
    covars = safe_regress_covariates(adata, REGRESS_CANDIDATES)
    print(f"regress_out covariates (safe): {covars}", flush=True)
    if covars:
        with timed("regress_out"):
            sc.pp.regress_out(adata, keys=covars)
        with timed("Sanity: replace non-finite after regress_out"):
            nbad = replace_nonfinite_in_X_inplace(adata, fill_value=0.0)
            print(f"Non-finite in X fixed: {nbad}", flush=True)
        with timed("Drop empty cells / all-zero genes after regress_out"):
            drop_empty_cells_and_allzero_genes(adata)
    else:
        print("No safe regress_out covariates; skipping regress_out.", flush=True)

# ----------------------------
# Scale (optional)
# ----------------------------
if DO_SCALE:
    with timed(f"scale(max_value={SCALE_MAX_VALUE}, zero_center={SCALE_ZERO_CENTER})"):
        sc.pp.scale(adata, max_value=float(SCALE_MAX_VALUE), zero_center=bool(SCALE_ZERO_CENTER))
    with timed("Sanity: replace non-finite after scale"):
        nbad = replace_nonfinite_in_X_inplace(adata, fill_value=0.0)
        print(f"Non-finite in X fixed: {nbad}", flush=True)
    with timed("Drop empty cells / all-zero genes after scale"):
        drop_empty_cells_and_allzero_genes(adata)
else:
    print("Scaling disabled.", flush=True)

# ----------------------------
# PCA
# ----------------------------
if DENSIFY_FOR_PCA:
    with timed("Densify X for PCA (WARNING: memory heavy)"):
        if sparse.issparse(adata.X):
            adata.X = adata.X.astype(np.float32).toarray()
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32, order="C")

with timed(f"PCA (n_comps={N_PCS}, svd_solver={PCA_PRIMARY_SOLVER} -> {PCA_FALLBACK_SOLVER})"):
    run_pca_with_fallback(adata, n_comps=N_PCS, random_state=RANDOM_STATE)

# ----------------------------
# Base neighbors + Leiden
# ----------------------------
with timed(f"neighbors (base) (n_neighbors={N_NEIGHBORS}, n_pcs={N_PCS_NEIGHBORS})"):
    compute_neighbors_inplace(
        adata,
        n_neighbors=int(N_NEIGHBORS),
        n_pcs=int(N_PCS_NEIGHBORS),
        metric="euclidean",
        random_state=int(RANDOM_STATE),
    )

with timed(f"Leiden (resolution={LEIDEN_RES}, flavor={LEIDEN_FLAVOR}, n_iterations={LEIDEN_N_ITER}, directed={LEIDEN_DIRECTED})"):
    sc.tl.leiden(
        adata,
        resolution=float(LEIDEN_RES),
        flavor=str(LEIDEN_FLAVOR),
        n_iterations=int(LEIDEN_N_ITER),
        directed=bool(LEIDEN_DIRECTED),
    )

# ============================================================
# SELECTED "BEST" UMAP -> X_umap
# ============================================================
selected_dir = plots_dir / "selected_embeddings"
selected_dir.mkdir(parents=True, exist_ok=True)

with timed(
    f"Compute BEST UMAP -> X_umap "
    f"(metric={BEST_UMAP['metric']}, nn={BEST_UMAP['n_neighbors']}, n_pcs={BEST_UMAP['n_pcs']}, "
    f"md={BEST_UMAP['min_dist']}, sp={BEST_UMAP['spread']}, seed={BEST_UMAP['random_state']})"
):
    compute_neighbors_inplace(
        adata,
        n_neighbors=int(BEST_UMAP["n_neighbors"]),
        n_pcs=int(BEST_UMAP["n_pcs"]),
        metric=str(BEST_UMAP["metric"]),
        random_state=int(BEST_UMAP["random_state"]),
    )
    sc.tl.umap(
        adata,
        min_dist=float(BEST_UMAP["min_dist"]),
        spread=float(BEST_UMAP["spread"]),
        random_state=int(BEST_UMAP["random_state"]),
    )

# ============================================================
# SELECTED "BEST" t-SNE (full or subsample)
# ============================================================
with timed(f"Prepare t-SNE base (subsample={TSNE_SUBSAMPLE_N})"):
    if TSNE_SUBSAMPLE_N is not None and adata.n_obs > int(TSNE_SUBSAMPLE_N):
        rng = np.random.default_rng(int(BEST_TSNE["random_state"]))
        tsne_idx = rng.choice(adata.n_obs, size=int(TSNE_SUBSAMPLE_N), replace=False)
        ad_tsne = adata[tsne_idx].copy()
    else:
        tsne_idx = None
        ad_tsne = adata.copy()

with timed(
    f"Compute BEST t-SNE "
    f"(pp={BEST_TSNE['perplexity']}, ee={BEST_TSNE['early_exaggeration']}, mi={BEST_TSNE['max_iter']}, seed={BEST_TSNE['random_state']})"
):
    kwargs = tsne_kwargs_compatible(
        use_rep="X_pca",
        n_pcs=int(BEST_TSNE["n_pcs"]),
        perplexity=float(BEST_TSNE["perplexity"]),
        early_exaggeration=float(BEST_TSNE["early_exaggeration"]),
        learning_rate=BEST_TSNE["learning_rate"],
        max_iter=int(BEST_TSNE["max_iter"]),
        n_iter=int(BEST_TSNE["max_iter"]),  # for older scanpy
        random_state=int(BEST_TSNE["random_state"]),
        init=BEST_TSNE["init"],
    )
    sc.tl.tsne(ad_tsne, **kwargs)

# Store into main object
if tsne_idx is None:
    with timed("Store BEST t-SNE into adata.obsm['X_tsne'] (full)"):
        adata.obsm["X_tsne"] = ad_tsne.obsm["X_tsne"].astype(np.float32, copy=True)
else:
    with timed("Store BEST t-SNE into adata.obsm['X_tsne_subsample'] (NaNs elsewhere)"):
        full = np.full((adata.n_obs, 2), np.nan, dtype=np.float32)
        full[tsne_idx, :] = ad_tsne.obsm["X_tsne"].astype(np.float32, copy=False)
        adata.obsm["X_tsne_subsample"] = full
    with timed("Record t-SNE subsample indices in adata.uns"):
        adata.uns["tsne_subsample_idx"] = tsne_idx.astype(np.int64, copy=False)

# ============================================================
# PLOTS (labels + optional marker panels)
# ============================================================
tag = (
    f"best__umap__metric{BEST_UMAP['metric']}__nn{BEST_UMAP['n_neighbors']}__npc{BEST_UMAP['n_pcs']}__"
    f"md{BEST_UMAP['min_dist']}__sp{BEST_UMAP['spread']}__seed{BEST_UMAP['random_state']}__"
    f"tsne__pp{BEST_TSNE['perplexity']}__ee{BEST_TSNE['early_exaggeration']}__mi{BEST_TSNE['max_iter']}__"
    f"seed{BEST_TSNE['random_state']}__n{ad_tsne.n_obs}"
)

with timed("Plot selected embeddings"):
    plot_selected_embeddings(adata, out_dir=selected_dir, prefix=tag)
    # For tSNE plot, plot from ad_tsne (since it definitely has X_tsne)
    plot_selected_embeddings(ad_tsne, out_dir=selected_dir, prefix=tag + "__tsne_base")

# ============================================================
# SAVE ADATA *BEFORE* DOTPLOTS
# ============================================================
with timed(f"Write AnnData to {out_path.name} [BEFORE dotplots]"):
    adata.write(out_path)

# ============================================================
# DOTPLOTS (optional)
# ============================================================
if MAKE_DOTPLOTS:
    dp_dir = plots_dir / "dotplots"
    dp_dir.mkdir(parents=True, exist_ok=True)

    for groupby in DOTPLOT_GROUPS:
        if groupby not in adata.obs:
            print(f"[WARN] groupby='{groupby}' not in adata.obs; skipping dotplot.", flush=True)
            continue

        try:
            with timed(f"Top {TOP_N_GENES_PER_GROUP} genes per group: groupby={groupby}"):
                per_group = top_genes_per_group_mean(
                    adata,
                    groupby=groupby,
                    layer=DOTPLOT_USE_LAYER if DOTPLOT_USE_LAYER in adata.layers else None,
                    n_top=int(TOP_N_GENES_PER_GROUP),
                    min_pct=float(DOTPLOT_MIN_PCT),
                )
                genes = flatten_unique_gene_list(per_group)

                table_path = dp_dir / f"top_genes__{groupby}__n{TOP_N_GENES_PER_GROUP}.tsv"
                rows = []
                for g, gl in per_group.items():
                    for rank, gene in enumerate(gl, start=1):
                        rows.append((g, rank, gene))
                pd.DataFrame(rows, columns=[groupby, "rank", "gene"]).to_csv(table_path, sep="\t", index=False)

            with timed(f"Save dotplot figure for groupby={groupby} (genes={len(genes)})"):
                out_png = dp_dir / f"dotplot__top{TOP_N_GENES_PER_GROUP}_per_{groupby}.png"
                save_dotplot(adata, groupby=groupby, genes=genes, out_png=out_png)

        except Exception as e:
            print(f"[WARN] Dotplot failed for groupby='{groupby}': {e}", flush=True)

print("\n✅ Finished.", flush=True)
print(f"Wrote: {out_path}", flush=True)
print(f"Saved plots to: {plots_dir}", flush=True)
print(f"Selected embedding plots saved to: {selected_dir}", flush=True)
print(f"Dotplots saved to: {plots_dir / 'dotplots'}", flush=True)