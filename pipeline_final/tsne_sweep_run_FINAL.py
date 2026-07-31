#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/tsne_sweep_run_FINAL.py
#
# One tSNE parameter combination per array task. The grid is six perplexities (15 to 200)
# crossed with four late-exaggeration values. That is the 24 tasks.
#
# Prefers cuML on the GPU and falls back to openTSNE on CPU when no GPU is available, so the
# same script works whether or not a card was allocated. The fallback is silent by design;
# check the log if you care which path ran.
#
# Each task emits one comparison PNG panelled by Leiden, astrocyte call, astrocyte score,
# WDR49 and sample, labelled with its parameters, plus the embedding coordinates as .npy so
# the winning combination can be re-plotted later without recomputing.
#
# Judge these by eye against the marker panels rather than by a single score. A tSNE that
# optimises one number and separates nothing useful is easy to produce.
# ========================================================================================

"""tsne_sweep_run_FINAL.py -- one tSNE parameter combo per SLURM array task. Prefers
cuML GPU t-SNE; falls back to openTSNE (CPU, multicore) if no GPU. Emits one
comparison PNG (Leiden | astrocyte call | astro-score | WDR49 | sample) labelled
with params, plus the embedding coords as .npy for re-plotting the winner later.

ADAPTED FROM /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/tsne_sweep_run.py.
Logic (parameter grid, cuML->openTSNE fallback, comparison panel) PRESERVED
VERBATIM; _FINAL deltas: [F1] import final_config for --in_h5ad/--outdir defaults
(astro_isolation_FINAL/tsne_sweep); [F3] belt-and-braces ~/.local sys.path strip.
Run under opentsne_gpu (runner sets LD_LIBRARY_PATH=$ENV/lib for the GPU/cuML libs).
"""
import sys
# [FINAL F3] drop user-site paths so the env's own anndata/scanpy win.
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
# [FINAL F1] make the shared foundation importable.
sys.path.insert(0, "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code")

import os, argparse, numpy as np
from pathlib import Path
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import final_config as cfg  # [FINAL F1]

# ---- parameter grid (index by SLURM_ARRAY_TASK_ID) ----
PERPLEXITY = [15, 30, 50, 80, 120, 200]
LATE_EXAG  = [1.0, 1.5, 2.5, 4.0]
GRID = [(p, l) for p in PERPLEXITY for l in LATE_EXAG]   # 24 combos

ap = argparse.ArgumentParser()
# [FINAL F1] defaults from the foundation; runner still passes them explicitly.
ap.add_argument("--in_h5ad", default=str(cfg.ASTRO_DIR / "tsne_sweep" / "sweep_input.h5ad"))
ap.add_argument("--outdir", default=str(cfg.ASTRO_DIR / "tsne_sweep"))
ap.add_argument("--task_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
ap.add_argument("--goi", default="WDR49")
ap.add_argument("--early_exag", type=float, default=12.0)
ap.add_argument("--n_iter", type=int, default=1000)
args = ap.parse_args()

perp, late = GRID[args.task_id]
tag = f"p{int(perp)}_late{late:g}"
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(f"[sweep {tag}] {m}", flush=True)
log(f"task {args.task_id}/{len(GRID)-1}  perplexity={perp} late_exag={late}")

adata = sc.read_h5ad(args.in_h5ad)
X = np.asarray(adata.obsm["X_pca"], dtype="float32")
n = X.shape[0]
lr = max(200.0, n / 12.0)

engine = None
try:
    from cuml.manifold import TSNE as cuTSNE
    log("using cuML GPU TSNE")
    tsne = cuTSNE(n_components=2, perplexity=perp, early_exaggeration=args.early_exag,
                  late_exaggeration=late, learning_rate=lr, n_iter=args.n_iter,
                  method="fft", metric="euclidean", init="random", random_state=0, verbose=False)
    emb = tsne.fit_transform(X)
    emb = emb.values if hasattr(emb, "values") else np.asarray(emb)
    engine = "cuml"
except Exception as e:
    log(f"cuML unavailable ({e}); falling back to openTSNE CPU")
    from openTSNE import TSNE as oTSNE
    tsne = oTSNE(n_components=2, perplexity=perp, exaggeration=late,
                 early_exaggeration=args.early_exag, learning_rate=lr,
                 n_iter=args.n_iter, initialization="pca", metric="euclidean",
                 n_jobs=8, random_state=0)
    emb = np.asarray(tsne.fit(X))
    engine = "opentsne"

adata.obsm["X_tsne"] = emb
np.save(OUT / f"coords_{tag}.npy", emb)
log(f"embedded ({engine}); coords saved")

# ---- comparison panel ----
adata.obs["astrocyte"] = adata.obs["is_astro"].map({True: "astrocyte", False: "other"}).astype("category")
goi = f"{args.goi}_expr"
xy = emb
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# leiden
cats = list(adata.obs["leiden"].astype("category").cat.categories)
cmap = plt.get_cmap("tab20"); cd = {c: cmap(i % 20) for i, c in enumerate(cats)}
axes[0].scatter(xy[:, 0], xy[:, 1], s=0.6, c=[cd[v] for v in adata.obs["leiden"].astype(str)], linewidths=0, rasterized=True)
axes[0].set_title("Leiden")
# astrocyte call
acol = adata.obs["astrocyte"].map({"astrocyte": "#d1495b", "other": "#d9d9d9"}).values
axes[1].scatter(xy[:, 0], xy[:, 1], s=0.6, c=acol, linewidths=0, rasterized=True)
axes[1].set_title("Astrocyte call")
# astro score
sca = axes[2].scatter(xy[:, 0], xy[:, 1], s=0.6, c=adata.obs["score_astro"], cmap="viridis", linewidths=0, rasterized=True)
axes[2].set_title("score_astro"); fig.colorbar(sca, ax=axes[2], fraction=0.046)
# WDR49
if goi in adata.obs:
    scw = axes[3].scatter(xy[:, 0], xy[:, 1], s=0.6, c=adata.obs[goi], cmap="magma", linewidths=0, rasterized=True)
    fig.colorbar(scw, ax=axes[3], fraction=0.046)
axes[3].set_title(args.goi)
# sample (batch)
skey = "sample" if "sample" in adata.obs else ("sd_code" if "sd_code" in adata.obs else None)
if skey:
    scats = list(adata.obs[skey].astype("category").cat.categories)
    scmap = plt.get_cmap("tab20"); scd = {c: scmap(i % 20) for i, c in enumerate(scats)}
    axes[4].scatter(xy[:, 0], xy[:, 1], s=0.6, c=[scd[v] for v in adata.obs[skey].astype(str)], linewidths=0, rasterized=True)
axes[4].set_title("Sample (batch)")

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.suptitle(f"t-SNE  perplexity={perp}  late_exaggeration={late}  (early={args.early_exag}, n={n:,}, {engine})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / f"tsne_{tag}.png", dpi=150)
log(f"wrote tsne_{tag}.png")
