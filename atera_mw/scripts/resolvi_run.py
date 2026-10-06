#!/usr/bin/env python3
"""Run resolVI (scvi-tools, standard settings, unsupervised) on one segmentation's window AnnData and score raw vs corrected.
Corrected counts = median of 3 posterior samples of px_rate under the corrected model (as in the resolVI tutorial).
Scores on the same genes for raw and corrected: MECR over the MARKERS panel (presence = count >= 1, i.e. corrected >= 0.5),
and Purkinje-gene leakage = mean share of Purkinje-specific genes' counts in cells whose own Purkinje share is < 1%.
Run in the separate `resolvi` env:  python resolvi_run.py <in.h5ad> <out prefix>"""
import sys, itertools, json
import numpy as np, pandas as pd, anndata as ad, scvi, torch
from scvi.external import RESOLVI
MARKERS = {"neuron": ["STMN2", "RBFOX3", "SLC17A6", "GAD1", "GAD2"], "oligodendrocyte": ["MBP", "MOBP", "MOG", "MAG", "OPALIN", "ST18"],
           "astrocyte": ["AQP4", "SLC1A2", "GJA1"], "microglia": ["C1QC", "P2RY12", "PTPRC"]}            # = splitmerge.metrics.MARKERS
GS = json.load(open("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare/purkinje/gene_sets.json"))
scvi.settings.seed = 0; torch.set_num_threads(8)
a = ad.read_h5ad(sys.argv[1]); a = a[np.asarray(a.layers["counts"].sum(1)).ravel() >= 5].copy(); a.X = a.layers["counts"].copy()   # resolVI needs >= 5 counts per cell on the gene set
RESOLVI.setup_anndata(a, layer="counts", batch_key="window")
m = RESOLVI(a); m.train(max_epochs=50, accelerator="cpu")
s = m.sample_posterior(model=m.module.model_corrected, return_sites=["px_rate"], summary_fun={"q50": np.median}, num_samples=3, summary_frequency=100, accelerator="cpu")
s = pd.DataFrame(s).T; corr = np.asarray(s.loc["q50", "px_rate"])
raw = a.layers["counts"].toarray()
def mecr(Xm, genes, min_tx=5):
    keep = Xm.sum(1) >= min_tx; P = (Xm[keep] >= 0.5); gi = {g: i for i, g in enumerate(genes)}; rates = []
    for (ta, ga), (tb, gb) in itertools.combinations([(t, g) for t, gg in MARKERS.items() for g in gg], 2):
        if ta == tb or ga not in gi or gb not in gi: continue
        x, y = P[:, gi[ga]], P[:, gi[gb]]; e = (x | y).sum()
        if e: rates.append((x & y).sum() / e)
    return float(np.mean(rates))
genes = list(a.var_names); pk = np.isin(genes, GS["purkinje_A"] + GS["purkinje_B"])
def leak(Xm):
    tot = Xm.sum(1); f = Xm[:, pk].sum(1) / np.maximum(tot, 1e-9); own = f < 0.01
    return float(f[(tot > 0)].mean()), float((Xm[:, pk] >= 0.5).any(1).mean())
res = {"cells": a.n_obs, "genes": a.n_vars, "mecr_raw": mecr(raw, genes), "mecr_resolvi": mecr(corr, genes),
       "purkinje_share_raw": leak(raw)[0], "purkinje_share_resolvi": leak(corr)[0],
       "cells_with_purkinje_gene_raw": leak(raw)[1], "cells_with_purkinje_gene_resolvi": leak(corr)[1],
       "counts_per_cell_raw": float(np.median(raw.sum(1))), "counts_per_cell_resolvi": float(np.median(corr.sum(1)))}
per_win = {w: {"mecr_raw": mecr(raw[(a.obs.window == w).to_numpy()], genes), "mecr_resolvi": mecr(corr[(a.obs.window == w).to_numpy()], genes)} for w in a.obs.window.unique()}
json.dump({"overall": res, "per_window": per_win}, open(sys.argv[2] + "_scores.json", "w"), indent=1)
np.save(sys.argv[2] + "_corrected.npy", corr.astype(np.float32)); print(json.dumps(res, indent=1))
