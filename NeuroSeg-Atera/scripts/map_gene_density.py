#!/usr/bin/env python3
"""Whole-section map of one gene's transcript density, plus the same relative to total transcript density.

python map_gene_density.py GJD2 [bin_um]
Reads connexin_transcripts.parquet (QC transcripts of the gene) and the whole-sample density zarr (block-averaged to
the same bins, so the denominator is the total QC transcripts per bin), DAPI low-res for context.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, zarr, tifffile, matplotlib
from scipy import ndimage as ndi
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
GENE = sys.argv[1]; BIN = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); UM = 0.2125
t = pd.read_parquet(SSD / "analysis/connexin_transcripts.parquet")
t = t[(t.feature_name == GENE) & t.is_gene & (t.qv >= 20)]
dens = zarr.open_array(str(SSD / "work/density/tx_density.zarr"), mode="r")
H, W = dens.shape
blk = int(round(BIN / UM)); nby, nbx = H // blk, W // blk
tot = np.zeros((nby, nbx))
step = blk * 32
for y0 in range(0, nby * blk, step):
    y1 = min(nby * blk, y0 + step)
    a = np.asarray(dens[y0:y1, : nbx * blk], dtype=np.float32)
    tot[y0 // blk:y1 // blk] = a.reshape((y1 - y0) // blk, blk, nbx, blk).mean(axis=(1, 3)) * (BIN * BIN)   # tx per bin
gx = (t.x_location.to_numpy() / BIN).astype(int); gy = (t.y_location.to_numpy() / BIN).astype(int)
ok = (gx < nbx) & (gy < nby)
cnt = np.zeros((nby, nbx)); np.add.at(cnt, (gy[ok], gx[ok]), 1)
sm = lambda a, s: ndi.gaussian_filter(a, s)
s = 1.5                                                            # smoothing in bins
area_mm2 = (BIN / 1000) ** 2
dens_g = sm(cnt, s) / area_mm2                                      # gene transcripts per mm^2
rel = sm(cnt, s) / np.maximum(sm(tot, s), 1) * 1e5                  # per 100k total transcripts
tissue = sm(tot, 1) > 0.15 * np.percentile(tot[tot > 0], 90)
rel = np.where(tissue, rel, np.nan)
lv = tifffile.TiffFile(str(SSD / "Cerebellum_sample/morphology_focus/ch0000_dapi.ome.tif")).series[0].levels[4].asarray()[0].astype(float)
lv = np.clip(lv / np.percentile(lv[lv > 0], 99.5), 0, 1)
ext = (0, nbx * BIN / 1000, nby * BIN / 1000, 0)
fig, axs = plt.subplots(1, 2, figsize=(19, 5.9), dpi=150)
for ax, (img, title, lab, vmax) in zip(axs, ((dens_g, f"{GENE}: QC transcripts per mm²", "transcripts / mm²", np.percentile(dens_g, 99.7)),
                                              (rel, f"{GENE} relative to total transcript density", "per 100,000 transcripts", np.nanpercentile(rel, 99.7)))):
    ax.imshow(lv, extent=(0, W * UM / 1000, H * UM / 1000, 0), cmap="gray_r", alpha=0.18, interpolation="antialiased")
    m = ax.imshow(np.ma.masked_less(img, 0.02 * vmax) if img is dens_g else np.ma.masked_invalid(img), extent=ext, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest", alpha=0.95)
    cb = fig.colorbar(m, ax=ax, fraction=0.025, pad=0.01); cb.set_label(lab); cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xlim(0, W * UM / 1000); ax.set_ylim(H * UM / 1000, 0)
    ax.tick_params(colors="#898781", labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.suptitle(f"Atera cerebellum, {GENE}: {len(t):,} QC transcripts, {BIN:g} µm bins, Gaussian σ {s * BIN:g} µm  |  grey = DAPI for tissue context", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = SSD / f"analysis/{GENE}_density_map.png"; fig.savefig(out, facecolor="white"); print("saved", out)
# hotspots: smoothed relative density, bins with enough transcripts, spaced apart
cand = np.where(tissue & (sm(cnt, s) * (2 * s + 1) ** 2 > 5), rel, np.nan)
spots = []
for i in np.argsort(np.nan_to_num(cand, nan=-1).ravel())[::-1]:
    y, x = np.unravel_index(i, cand.shape)
    if np.isnan(cand[y, x]): break
    if all(np.hypot(y - a, x - b) * BIN > 400 for a, b in spots):
        spots.append((y, x))
    if len(spots) == 6: break
print("top relative-density spots (x, y in mm):", [(round((x + .5) * BIN / 1000, 2), round((y + .5) * BIN / 1000, 2)) for y, x in spots])
print(f"median GJD2 per mm^2 over tissue bins: {np.median(dens_g[tissue]):.1f}; 99th pct {np.percentile(dens_g[tissue], 99):.1f}")
