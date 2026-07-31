#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/tsne_sweep_prep_FINAL.py
#
# Cuts one fixed subsample so that every task in the tSNE sweep embeds exactly the same cells.
# Without this the sweep compares parameters and sampling noise at the same time and you
# cannot tell which produced the difference.
#
# Reads the astrocyte clustering checkpoint, subsamples N cells with a fixed seed, keeps X_pca
# plus the obs columns needed for colouring (leiden, is_astro, sample, the score_* columns and
# a WDR49 expression column), and drops X entirely so the file stays small enough for 24 array
# tasks to load without contention.
# ========================================================================================

"""tsne_sweep_prep_FINAL.py -- prep a single fixed subsample for the tSNE parameter
sweep, so every array task embeds the IDENTICAL cells (only the tSNE params differ
-> comparable).

Reads the _FINAL clustered checkpoint (astro_isolation_FINAL/clustered_full.h5ad),
subsamples N cells (fixed seed), keeps X_pca + the obs columns needed for colouring
(leiden, is_astro, sample, score_* and a WDR49 expression column). Drops X to keep
the file tiny.

ADAPTED FROM /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/tsne_sweep_prep.py.
Logic PRESERVED VERBATIM; _FINAL deltas: [F1] import final_config for --ckpt/--out
defaults (astro_isolation_FINAL); [F3] belt-and-braces ~/.local sys.path strip.
"""
import sys
# [FINAL F3] drop user-site paths so the env's own anndata/scanpy win.
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
# [FINAL F1] make the shared foundation importable.
sys.path.insert(0, "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code")

import argparse, numpy as np, scanpy as sc
from pathlib import Path

import final_config as cfg  # [FINAL F1]

ap = argparse.ArgumentParser()
# [FINAL F1] defaults from the foundation; runner still passes them explicitly.
ap.add_argument("--ckpt", default=str(cfg.ASTRO_DIR / "clustered_full.h5ad"))
ap.add_argument("--out", default=str(cfg.ASTRO_DIR / "tsne_sweep" / "sweep_input.h5ad"))
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--goi", default="WDR49")
args = ap.parse_args()

print(f"[prep] reading {args.ckpt}", flush=True)
adata = sc.read_h5ad(args.ckpt)
print(f"[prep] {adata.n_obs:,} cells", flush=True)

# pull WDR49 expression into obs before we drop X
if args.goi in adata.var_names:
    w = adata[:, args.goi].X
    adata.obs[f"{args.goi}_expr"] = (w.toarray().ravel() if hasattr(w, "toarray") else np.asarray(w).ravel())

if adata.n_obs > args.n:
    sc.pp.subsample(adata, n_obs=args.n, random_state=0)
print(f"[prep] subsampled to {adata.n_obs:,}", flush=True)

# keep only what plotting needs: X_pca + obs
keep_obs = [c for c in adata.obs.columns
            if c in ("leiden", "is_astro", "sample", "sd_code", f"{args.goi}_expr")
            or c.startswith("score_")]
slim = sc.AnnData(
    X=np.zeros((adata.n_obs, 1), dtype="float32"),
    obs=adata.obs[keep_obs].copy(),
    obsm={"X_pca": np.asarray(adata.obsm["X_pca"], dtype="float32")},
)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
slim.write_h5ad(args.out, compression="gzip")
print(f"[prep] wrote {args.out}  (X_pca {slim.obsm['X_pca'].shape}, obs {list(slim.obs.columns)})", flush=True)
