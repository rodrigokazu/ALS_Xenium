#!/usr/bin/env python3
"""Robustness of the Purkinje dendritic index (oq_dendrites.py), for the 41 Purkinje-specific genes.

1 Replication: the index is computed separately in the left and right half of the section (split at the median x of the
  Purkinje cells; somata and neuropil from the same half); Spearman between halves.
2 Distance profile: molecular-layer neuropil counted in bands of distance from the granule layer (25-75, 75-150, 150-300 um),
  each band divided by the gene's soma count and by the band's tissue area, so a gene reaching distal dendrites keeps signal
  in the outer bands. Reported as the share of a gene's molecular-layer neuropil that lies beyond 75 um.
3 Abundance: Spearman of the index with the soma count.
python oq_dendrites_robust.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse, stats, ndimage as ndi
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; Q = O / "open_questions"; B = 25.0
layer = np.load(Q / "layer_map.npy"); Hh, Ww = layer.shape
dgl = ndi.distance_transform_edt(layer != "granule layer") * B
pcd = pd.read_csv(Q / "purkinje_dendritic_index.csv"); G = list(pcd.gene)
nb = pd.read_parquet(Q / "neuropil_bin_gene.parquet"); nb = nb[(nb.by < Hh) & (nb.bx < Ww) & nb.gene.isin(G)].copy()
nb["layer"] = layer[nb.by, nb.bx]; nb["d"] = dgl[nb.by, nb.bx]; ml = nb[nb.layer == "molecular layer"]
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.Index(pd.read_csv(O / "genes.csv").gene.astype(str))
pz = pd.read_parquet(Q / "purkinje_zebrin_v2.parquet"); idx = pd.Index(cells.cell).get_indexer(pz.cell); gi = genes.get_indexer(G)
xmid = pz.x_centroid_um.median()
out = pcd[["gene", "di_rel", "S_PC", "N_ML"]].copy()
for half, mpc, mnb in (("left", pz.x_centroid_um < xmid, ml.bx * B < xmid), ("right", pz.x_centroid_um >= xmid, ml.bx * B >= xmid)):
    soma = np.asarray(X[idx[mpc.to_numpy()]][:, gi].sum(0)).ravel(); neu = ml[mnb.to_numpy()].groupby("gene").n.sum().reindex(G).fillna(0).to_numpy()
    di = np.log2((neu + 1) / (soma + 1)); out[f"di_{half}"] = di - np.median(di)
rho_halves = stats.spearmanr(out.di_left, out.di_right).statistic
bands = [(25, 75), (75, 150), (150, 300)]
area = {b: int(((layer == "molecular layer") & (dgl > b[0]) & (dgl <= b[1])).sum()) for b in bands}
for b in bands:
    s = ml[(ml.d > b[0]) & (ml.d <= b[1])].groupby("gene").n.sum().reindex(G).fillna(0)
    out[f"band_{b[0]}_{b[1]}"] = (s / out.set_index("gene").S_PC.reindex(G) / area[b]).to_numpy() * 1e3
tot = ml.groupby("gene").n.sum().reindex(G); far = ml[ml.d > 75].groupby("gene").n.sum().reindex(G).fillna(0)
out["share_beyond_75um"] = (far / tot).to_numpy()
rho_abund = stats.spearmanr(out.di_rel, np.log(out.S_PC)).statistic; rho_far = stats.spearmanr(out.di_rel, out.share_beyond_75um).statistic
out.to_csv(Q / "dendritic_index_robustness.csv", index=False)
print(f"left/right halves: Spearman {rho_halves:.3f} over {len(out)} genes | index vs log soma count: {rho_abund:.3f} | index vs share beyond 75 um: {rho_far:.3f}")
print("band areas (bins):", area)
print(out.sort_values("di_rel", ascending=False)[["gene", "di_rel", "di_left", "di_right", "share_beyond_75um"]].round(2).to_string(index=False))
