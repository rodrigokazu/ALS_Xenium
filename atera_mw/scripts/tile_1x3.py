#!/usr/bin/env python3
"""1x3 of one tile: morphology merge (DAPI, 18S) | every QC transcript as a dot, one random colour per gene | transcript density
(the whole-sample density image, tx / um2, fixed scale). python tile_1x3.py <window> <soma> <out.png>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from scipy import ndimage as ndi
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); win, soma, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = pd.read_csv(SSD / "compare/purkinje/pc_somata.csv").query("window == @win and soma == @soma").iloc[0]
HALF = int(35 / UM); cx, cy = int(s.cx_px), int(s.cy_px); X0, Y0 = cx - HALF, cy - HALF; n = 2 * HALF; L = n * UM
bundle = SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
z = zarr.open(tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
im = np.asarray(z[:, Y0:Y0 + n, X0:X0 + n]).astype(float)
mg = np.clip(np.clip((im[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((im[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]), 0, 1)
tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < (X0 + n) * UM)
     & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < (Y0 + n) * UM) & (pc.field("qv") >= 20) & pc.field("is_gene")).to_pandas()
ntot, ngen = len(tx), tx.feature_name.nunique()
codes, uniq = pd.factorize(tx.feature_name); rng = np.random.default_rng(0)
sub = rng.random(len(tx)) < 0.12
lut = matplotlib.colors.hsv_to_rgb(np.c_[rng.uniform(0, 1, len(uniq)), rng.uniform(0.5, 1, len(uniq)), rng.uniform(0.7, 1, len(uniq))])
dens = np.asarray(zarr.open_array(str(SSD / "work/density/tx_density.zarr"), mode="r")[Y0:Y0 + n, X0:X0 + n]).astype(float)
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.9), dpi=170, gridspec_kw=dict(wspace=0.04))
ext = (0, L, L, 0)
axs[0].imshow(mg, extent=ext); axs[0].set_title("Morphology (DAPI blue, 18S yellow)", loc="left", fontsize=11, fontweight="bold")
axs[1].set_facecolor("black"); axs[1].scatter(tx.x_location[sub] - X0 * UM, tx.y_location[sub] - Y0 * UM, s=0.9, c=lut[codes[sub]], lw=0, rasterized=True)
axs[1].set_xlim(0, L); axs[1].set_ylim(L, 0); axs[1].set_aspect("equal")
axs[1].set_title(f"Transcripts, colour = gene (12% of {ntot:,})", loc="left", fontsize=11, fontweight="bold")
imd = axs[2].imshow(ndi.gaussian_filter(dens, 0.25 / UM), extent=ext, cmap="magma", vmin=0, vmax=250)
axs[2].set_title("Transcript density (tx / µm², σ 0.5 µm)", loc="left", fontsize=11, fontweight="bold")
cax = axs[2].inset_axes([0.04, 0.06, 0.03, 0.32]); cb = fig.colorbar(imd, cax=cax); cb.ax.tick_params(colors="white", labelsize=8); cb.set_label("tx / µm²", color="white", fontsize=8.5)
for a in axs:
    a.set_xticks([]); a.set_yticks([]); a.plot([L - 14, L - 4], [L - 3.5] * 2, color="white", lw=3, solid_capstyle="butt"); a.text(L - 9, L - 5, "10 µm", color="white", ha="center", fontsize=9)
fig.suptitle(f"Purkinje cell (held-out window {win}, soma {s.area_um2:.0f} µm²)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=0.97)
fig.savefig(out, facecolor="white", bbox_inches="tight"); print("saved", out)
