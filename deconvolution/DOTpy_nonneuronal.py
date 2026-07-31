#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | deconvolution/DOTpy_nonneuronal.py
#
# Cell type deconvolution against the Japanese spinal cord reference, using DOT (Rahimi et
# al.). This is the active version and the one that produced usable labels.
#
# The reference has only 101 motor neurons against 4,363 generic neurons. Nowhere near enough
# to deconvolve a rare type reliably, so both are collapsed into a single Neuron label before
# fitting. Motor neurons are identified afterwards from ventral horn anatomy and markers
# instead. Accepting that limit is what made the rest of the output trustworthy: DOT then
# cleanly separates astrocytes, endothelium, ependymal cells, lymphocytes, microglia, neurons,
# oligodendrocytes and OPCs.
#
# Four parameters were wrong in earlier versions and all four are fixed here, with the
# reasoning, because they are easy to set back. th_spatial is 0.84 not 0.94, since at 0.94
# almost no neighbour pairs qualify in sparse Xenium data and spatial smoothing switches
# itself off entirely. ratios_weight is 0.0 not 0.20, because the abundance prior penalised
# rare types against common glia and suppressed exactly the calls we cared about; use a prior
# only when you have a validated one. Gene boosting is off, because multiplying STMN2 by 20
# distorted the cosine geometry for every neuron rather than just motor neurons, and DOT
# selects genes internally anyway. Iterations are 500 not 100 so Frank-Wolfe can converge, and
# the gap threshold ends it early when it does.
#
# Two API notes. subcluster_size is the maximum number of subclusters per type, not cells per
# cluster. And force CSR after any sparse operation, because COO crashes on column slicing.
#
# Use spatial_orig for the spatial graph. The offset spatial coordinates would put every
# section in a different place.
# ========================================================================================

"""
DOTpy non-neuronal deconvolution sweep.

Reference has 101 MNs and 4363 generic 'Neuron' cells — both are collapsed into
a single 'Neuron' label before fitting. DOT then cleanly deconvolves:
  Astro | Endo | Ependymal | Lymphocyte | Micro | Neuron | Oligo | Opc

Changes from v3_fixed:
  - COLLAPSE_TO_NEURON: merges 'MNs' -> 'Neuron' in reference before setup_reference
  - HVG_SWEEP: narrowed to [2000, 5000] — enough for 8 cell types, avoids noise
  - All MN-specific outputs removed (no MN_maps, no MN marker enrichment)
  - Non-neuronal focused metrics: confidence + entropy per cell type, spatial boxplot
  - Saved h5ad: (RK)SC_filtered_deconvolution_nonneuronal__{sweep_tag}.h5ad
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import issparse, csr_matrix, csc_matrix

from dotpy import DOT, setup_reference, setup_spatial, plot_spatial_weights


# ============================================================
# CONFIG
# ============================================================

REF_H5AD = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Spinal_cord_reference/Japanese/adata_export/SC_atlas.h5ad"
)
SPATIAL_H5AD = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd/ALS_SCXenium_concatenated_offset_postQC.h5ad"
)
OUT_ROOT = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/DOT/DOTpy_nonneuronal"
)

# Collapse these reference labels into 'Neuron' before fitting.
# The reference has only 101 MNs — too few for reliable subclustering.
# Merging them with the 4363 generic 'Neuron' cells avoids a noisy 9th class.
COLLAPSE_TO_NEURON: list[str] = ["MNs"]

CELL_TYPE_KEY   = "PrimaryAnnotation"  # working column (modified in-place on copy)
SUBCLUSTER_SIZE = 10
HVG_SWEEP       = [2000, 5000]          # 8 cell types; 2000-5000 is sufficient

SPATIAL_KEY       = "spatial"
TH_SPATIAL        = 0.84
DOT_MODE          = "highres"
DOT_ITERATIONS    = 500
DOT_BATCH_SIZE    = 500
CHECKPOINT_FREQ   = 25
DOT_RATIOS_WEIGHT = 0.0    # no abundance priors

SAVE_FULL_CELLTYPE_MAPS = False
SAVE_CELLTYPE_BOX_PLOT  = True

COORDS_KEYS_PRIORITY = ("spatial_orig", "spatial")
DPI                  = 300
RNG_SEED             = 0

# Non-neuronal cell types we specifically care about for metrics
NON_NEURONAL_TYPES = ["Astro", "Oligo", "Opc", "Micro", "Endo", "Ependymal", "Lymphocyte"]


# ============================================================
# HELPERS
# ============================================================

_rng = np.random.default_rng(RNG_SEED)


def safe_name(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def get_xy(ad) -> np.ndarray:
    for k in COORDS_KEYS_PRIORITY:
        if k in ad.obsm:
            xy = np.asarray(ad.obsm[k])
            if xy.ndim == 2 and xy.shape[1] >= 2:
                return xy[:, :2]
    if {"x_centroid", "y_centroid"}.issubset(ad.obs.columns):
        return np.c_[ad.obs["x_centroid"].to_numpy(), ad.obs["y_centroid"].to_numpy()]
    raise KeyError("Could not find coordinates.")


def _get_processed_X(processed: dict):
    for k in ("X_sparse", "X"):
        if k in processed:
            return k, processed[k]
    raise KeyError(f"No expression matrix in processed. keys={list(processed.keys())}")


def _force_csr(processed: dict, tag: str = "") -> dict:
    xk, X = _get_processed_X(processed)
    if issparse(X) and not isinstance(X, (csr_matrix, csc_matrix)):
        processed[xk] = X.tocsr()
        print(f"[csr_fix:{tag}] converted {type(X).__name__} -> csr_matrix")
    elif issparse(X) and isinstance(X, csc_matrix):
        processed[xk] = X.tocsr()
    return processed


def cleanup_dot(adata) -> None:
    for k in ["dot_v3_weights", "dot_weights"]:
        if k in adata.obsm:
            del adata.obsm[k]
    for k in ["dot_by_sample", "dot_cell_types", "dot_config", "dot_sweep_info"]:
        adata.uns.pop(k, None)
    to_drop = [c for c in adata.obs.columns if c.startswith("dot_")]
    if to_drop:
        adata.obs.drop(columns=to_drop, inplace=True)


def compute_argmax_labels(W, cell_types):
    W = np.asarray(W)
    n = W.shape[0]
    valid = ~np.all(np.isnan(W), axis=1)
    W2 = np.where(np.isnan(W), -np.inf, W)
    idx = np.argmax(W2, axis=1).astype(int)
    idx[~valid] = -1
    labels = np.full(n, "NA", dtype=object)
    ct_arr = np.asarray(cell_types, dtype=object)
    labels[valid] = ct_arr[idx[valid]]
    scores = np.full(n, np.nan, dtype=np.float32)
    if np.any(valid):
        scores[valid] = np.nanmax(W[valid, :], axis=1).astype(np.float32)
    return labels, scores, idx


def compute_entropy(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    rs = np.nansum(W, axis=1, keepdims=True)
    rs[rs == 0] = np.nan
    P = W / rs
    P = np.where(np.isnan(P) | (P <= 0), 1e-12, P)
    return (-np.sum(P * np.log2(P), axis=1)).astype(np.float32)


def _prefer_counts(adata, tag=""):
    if "counts" not in adata.layers:
        return adata
    Xc = adata.layers["counts"]
    try:
        data = Xc.data if issparse(Xc) else np.asarray(Xc).ravel()
        if data.size == 0:
            return adata
        samp = data if data.size <= 200000 else _rng.choice(data, size=200000, replace=False)
        if np.mean(np.abs(samp - np.round(samp)) > 1e-6) <= 0.001:
            ad2 = adata.copy()
            ad2.X = ad2.layers["counts"]
            print(f"[counts:{tag}] using layers['counts'] for HVG")
            return ad2
    except Exception:
        pass
    return adata


# ============================================================
# METRICS
# ============================================================

def print_and_save_assignment_metrics(W, cell_types, labels, scores, entropy,
                                      sweep_tag, out_csv):
    n_total = len(labels)
    valid = ~np.isnan(scores)
    n_valid = int(valid.sum())

    rows = []
    for ct in cell_types:
        m = labels == ct
        n = int(m.sum())
        rows.append({
            "cell_type": ct,
            "n_cells": n,
            "fraction": round(n / n_total, 4),
            "mean_score": round(float(np.nanmean(scores[m])), 4) if n else float("nan"),
            "median_score": round(float(np.nanmedian(scores[m])), 4) if n else float("nan"),
            "mean_entropy": round(float(np.nanmean(entropy[m])), 4) if n else float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n[metrics:{sweep_tag}]  n_valid={n_valid:,} / {n_total:,}")
    print(df.to_string(index=False))

    max_H = np.log2(len(cell_types))
    for thr in (0.5, 0.7, 0.9):
        pct = 100 * float(np.sum(scores[valid] >= thr)) / n_valid if n_valid else 0
        print(f"  score >= {thr}: {pct:.1f}%")
    print(f"  mean entropy: {float(np.nanmean(entropy)):.3f} bits  (max = {max_H:.2f})")
    return df


def plot_confidence_histogram(scores, entropy, sweep_tag, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    valid_s = scores[~np.isnan(scores)]
    axes[0].hist(valid_s, bins=50, color="steelblue", edgecolor="none")
    for thr, col in [(0.5, "orange"), (0.7, "red"), (0.9, "darkred")]:
        axes[0].axvline(thr, color=col, linestyle="--", linewidth=1, label=f">={thr}")
    axes[0].set_xlabel("Max weight (assignment confidence)")
    axes[0].set_ylabel("Cells")
    axes[0].set_title(f"{sweep_tag} - confidence")
    axes[0].legend(fontsize=8)
    valid_e = entropy[~np.isnan(entropy)]
    axes[1].hist(valid_e, bins=50, color="coral", edgecolor="none")
    axes[1].set_xlabel("Weight entropy (bits)")
    axes[1].set_ylabel("Cells")
    axes[1].set_title(f"{sweep_tag} - entropy")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"[metrics:{sweep_tag}] histogram: {out_png}")


def make_boxplot(W_full, cell_types, run_ids_all, adata, out_png):
    rows = []
    for rid in np.unique(run_ids_all):
        m = run_ids_all == str(rid)
        w = W_full[m, :]
        keep = ~np.all(np.isnan(w), axis=1)
        w = w[keep, :]
        if w.shape[0] == 0:
            continue
        rs = np.nansum(w, axis=1, keepdims=True)
        rs[rs == 0] = np.nan
        mu = np.nanmean(w / rs, axis=0)
        sample = (str(adata.obs.loc[m, "sample"].iloc[0])
                  if "sample" in adata.obs.columns else "")
        rows.append([str(rid), sample, *mu.tolist()])
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows, columns=["run_id", "sample", *cell_types])
    long = df.melt(id_vars=["run_id", "sample"], value_vars=cell_types,
                   var_name="cell_type", value_name="proportion").dropna()
    order = (long.groupby("cell_type")["proportion"].median()
             .sort_values(ascending=False).index.tolist())
    data = [long.loc[long["cell_type"] == ct, "proportion"].to_numpy() for ct in order]
    means = [np.mean(x) if len(x) else np.nan for x in data]
    sems  = [np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan for x in data]
    plt.figure(figsize=(max(10, 0.35 * len(order)), 6))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.errorbar(np.arange(1, len(order)+1), means, yerr=sems, fmt="o", capsize=3)
    plt.ylabel("Per-run mean DOT proportion")
    plt.title(f"Cell type proportions across runs (n={df.shape[0]})")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"[boxplot] {out_png}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device='{device}'")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── load reference + collapse neuronal labels ──────────────────────────────
    print(f"[load] reference: {REF_H5AD}")
    ref_adata_raw = sc.read_h5ad(str(REF_H5AD))

    print(f"[collapse] merging {COLLAPSE_TO_NEURON} -> 'Neuron' in {CELL_TYPE_KEY}")
    ref_adata_raw.obs[CELL_TYPE_KEY] = (
        ref_adata_raw.obs[CELL_TYPE_KEY]
        .astype(str)
        .replace({k: "Neuron" for k in COLLAPSE_TO_NEURON})
    )
    print("[collapse] final cell type counts:")
    print(ref_adata_raw.obs[CELL_TYPE_KEY].value_counts().to_string())

    # ── load spatial ───────────────────────────────────────────────────────────
    print(f"[load] spatial: {SPATIAL_H5AD}")
    spatial_adata = sc.read_h5ad(str(SPATIAL_H5AD))

    if "run_id" not in spatial_adata.obs.columns:
        raise KeyError("spatial_adata.obs['run_id'] not found.")

    run_ids_all    = spatial_adata.obs["run_id"].astype(str).to_numpy()
    run_ids_unique = np.sort(np.unique(run_ids_all))
    print(f"[info] n_cells={spatial_adata.n_obs:,} | n_runs={len(run_ids_unique)}")

    # ── HVG sweep ──────────────────────────────────────────────────────────────
    for hvg_n in HVG_SWEEP:
        sweep_tag = f"HVG{int(hvg_n)}"
        outdir    = OUT_ROOT / sweep_tag
        outdir.mkdir(parents=True, exist_ok=True)
        metrics_dir = outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        cleanup_dot(spatial_adata)

        print("\n" + "="*80)
        print(f"[SWEEP] {sweep_tag} | th_spatial={TH_SPATIAL} | ratios_weight={DOT_RATIOS_WEIGHT}")

        ref_for_hvg = _prefer_counts(ref_adata_raw, tag="ref")
        ref_processed = setup_reference(
            ref_for_hvg.copy(),
            cell_type_key=CELL_TYPE_KEY,
            subcluster_size=SUBCLUSTER_SIZE,
            max_genes=int(hvg_n),
            verbose=True,
        )
        ref_processed = _force_csr(ref_processed, tag="ref")

        n_obs          = spatial_adata.n_obs
        all_cell_types = None
        W_full         = None

        spatial_adata.uns["dot_by_sample"] = {}
        spatial_adata.uns["dot_config"] = {
            "hvg_max_genes": int(hvg_n),
            "cell_type_key": CELL_TYPE_KEY,
            "collapsed_to_neuron": COLLAPSE_TO_NEURON,
            "th_spatial": float(TH_SPATIAL),
            "ratios_weight": float(DOT_RATIOS_WEIGHT),
            "dot_iterations": int(DOT_ITERATIONS),
            "device": device,
            "script_version": "nonneuronal_v1",
        }

        # ── per-run fitting ────────────────────────────────────────────────────
        for s in run_ids_unique:
            m = run_ids_all == str(s)
            n = int(m.sum())
            if n == 0:
                continue

            sample_label = (
                str(spatial_adata.obs.loc[m, "sample"].astype(str).iloc[0])
                if "sample" in spatial_adata.obs.columns else str(s)
            )
            print(f"\n[run] {s} | sample={sample_label} | n={n:,}")

            ad_s = spatial_adata[m].copy()
            if "spatial_orig" in ad_s.obsm:
                ad_s.obsm["spatial"] = ad_s.obsm["spatial_orig"].copy()

            spatial_proc = setup_spatial(ad_s, spatial_key=SPATIAL_KEY,
                                         th_spatial=TH_SPATIAL, verbose=True)
            spatial_proc = _force_csr(spatial_proc, tag=f"run{s}")

            dot_s = DOT(spatial_proc, ref_processed,
                        batch_size=DOT_BATCH_SIZE, device=device)
            dot_s.fit(mode=DOT_MODE, ratios_weight=DOT_RATIOS_WEIGHT,
                      iterations=DOT_ITERATIONS, checkpoint_freq=CHECKPOINT_FREQ,
                      verbose=True)

            w_s  = dot_s.get_weights(normalize=True)
            ct_s = list(dot_s.get_cell_types())

            if w_s is None:
                print(f"[warn] get_weights()=None for {s}. Skipping.")
                del dot_s, spatial_proc, ad_s
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            w_s = np.asarray(w_s)

            if all_cell_types is None:
                all_cell_types = ct_s
                W_full = np.full((n_obs, len(all_cell_types)), np.nan, dtype=np.float32)

            if ct_s != all_cell_types:
                raise ValueError(f"Cell type mismatch at run {s}.")
            if w_s.shape != (n, len(all_cell_types)):
                raise ValueError(f"Weight shape mismatch at run {s}.")

            W_full[m, :] = w_s.astype(np.float32)

            spatial_adata.uns["dot_by_sample"][str(s)] = {
                "n_cells": int(n), "status": "ok",
                "run_id": str(s), "sample": sample_label,
            }

            del dot_s, spatial_proc, ad_s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if W_full is None:
            raise RuntimeError("No successful DOT runs.")

        # ── attach results ─────────────────────────────────────────────────────
        spatial_adata.obsm["dot_v3_weights"] = W_full
        spatial_adata.uns["dot_cell_types"]  = all_cell_types
        spatial_adata.uns["dot_sweep_info"]  = {"sweep_tag": sweep_tag}

        labels, scores, idx = compute_argmax_labels(W_full, all_cell_types)
        entropy = compute_entropy(W_full)

        spatial_adata.obs["dot_label"]         = pd.Categorical(labels)
        spatial_adata.obs["dot_label_score"]   = scores
        spatial_adata.obs["dot_label_idx"]     = idx.astype(np.int32)
        spatial_adata.obs["dot_weight_entropy"] = entropy

        # ── metrics ────────────────────────────────────────────────────────────
        print_and_save_assignment_metrics(
            W_full, all_cell_types, labels, scores, entropy,
            sweep_tag, metrics_dir / f"assignment_metrics__{sweep_tag}.csv",
        )
        plot_confidence_histogram(
            scores, entropy, sweep_tag,
            metrics_dir / f"confidence_histogram__{sweep_tag}.png",
        )

        if SAVE_CELLTYPE_BOX_PLOT:
            make_boxplot(W_full, all_cell_types, run_ids_all, spatial_adata,
                         outdir / "boxplot__proportions_across_runs.png")

        # ── save ───────────────────────────────────────────────────────────────
        h5ad_path = outdir / f"(RK)SC_filtered_deconvolution_nonneuronal__{sweep_tag}.h5ad"
        spatial_adata.write(str(h5ad_path))
        print(f"[save] {h5ad_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll sweeps complete.")
    print(f"Outputs: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
