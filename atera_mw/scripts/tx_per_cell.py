#!/usr/bin/env python3
"""Transcripts per cell, ours vs 10x, on the same QC transcripts (qv >= 20, genes only).

A: distribution over all cells of the section (10x counts recounted with the same filter: tenx_qc_counts_per_cell.parquet).
B: per cell type (our transcript annotation), median transcripts per cell for the same nuclei: our cell vs the 10x cell that
   holds that nucleus (joined by the 10x nucleus id our base cell was seeded from); the ratio is the median of per-cell ratios.
python tx_per_cell.py
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
O = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full"); TENX, OURS, MUTED, GRID = "#eb6834", "#2a78d6", "#52514e", "#e1e0d9"
ours = pd.read_parquet(O / "cells.parquet"); ann = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ann.index = ann.index.astype(str)
tq = pd.read_parquet(O / "tenx_qc_counts_per_cell.parquet").qc_transcripts
base = pd.read_parquet(O / "base_cells.parquet")[["label", "nucleus_id", "has_nucleus"]]
ours["key"] = ours.cell.astype(str)
d = ours.merge(ann[["cell_type"]], left_on="key", right_index=True, how="left").merge(base, left_on="cell", right_on="label", how="left")
d["tenx_n"] = d.nucleus_id.map(tq)
fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150, gridspec_kw=dict(width_ratios=[0.9, 1.4], wspace=0.28))
ax = axs[0]; bins = np.linspace(1, 5, 81)
for v, colr, name in ((tq.to_numpy(), TENX, "10x"), (ours.n_transcripts.to_numpy(), OURS, "This work")):
    v = v[v > 0]; ax.hist(np.log10(v), bins=bins, histtype="step", lw=2, color=colr, label=f"{name}: {len(v):,} cells, median {np.median(v):,.0f}")
    ax.axvline(np.log10(np.median(v)), color=colr, lw=1, ls="--")
ax.set_xticks([1, 2, 3, 4, 5], ["10", "100", "1,000", "10,000", "100,000"]); ax.set_xlabel("QC transcripts per cell (log scale)"); ax.set_ylabel("cells")
ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("A  Transcripts per cell, whole section", loc="left", fontsize=11.5, fontweight="bold")
p = d[d.tenx_n.notna() & d.cell_type.notna() & ~d.cell_type.astype(str).str.startswith("mixed") & (d.n_transcripts > 0)].copy()
p["ratio"] = p.n_transcripts / p.tenx_n
g = p.groupby("cell_type").agg(n=("ratio", "size"), ours=("n_transcripts", "median"), tenx=("tenx_n", "median"), ratio=("ratio", "median"))
g = g[g.n >= 100].sort_values("ours")
ax = axs[1]; yy = np.arange(len(g))
for i, (t, r) in enumerate(g.iterrows()):
    ax.plot([r.tenx, r.ours], [i, i], color="#c3c2b7", lw=2, zorder=1)
    ax.text(max(r.ours, r.tenx) * 1.12, i, f"×{r.ratio:.2f}  (n = {int(r.n):,})", va="center", fontsize=8.5, color=MUTED)
ax.scatter(g.tenx, yy, s=46, color=TENX, zorder=3, label="10x, same nuclei"); ax.scatter(g.ours, yy, s=46, color=OURS, zorder=3, label="This work")
ax.set_xscale("log"); ax.set_xticks([1000, 2000, 5000, 10000], ["1,000", "2,000", "5,000", "10,000"]); ax.minorticks_off(); ax.set_yticks(yy, g.index); ax.set_xlabel("median QC transcripts per cell (log scale)")
ax.set_xlim(g[["ours", "tenx"]].min().min() * 0.7, g[["ours", "tenx"]].max().max() * 3.2)
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("B  Per cell type, the same nuclei in both segmentations (× = median per-cell ratio)", loc="left", fontsize=11.5, fontweight="bold")
fig.savefig(O / "transcripts_per_cell.png", facecolor="white", bbox_inches="tight")
g.round(2).to_csv(O / "transcripts_per_cell_by_type.csv"); print(g.round(2).to_string())
print("matched cells", len(p), "of", len(d), "| 10x median", tq.median(), "ours median", ours.n_transcripts.median())
