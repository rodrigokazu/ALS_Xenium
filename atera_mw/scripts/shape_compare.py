#!/usr/bin/env python3
"""Mask shape before / after: current (I2 + repair) | membrane landscape | membrane + 1 um opening, on two 60 um tiles."""
from pathlib import Path
import numpy as np, tifffile, zarr, matplotlib, json
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
TILES = [("held5", 1060, 1000, "granule / Purkinje layer"), ("held1", 600, 900, "granule layer")]   # tile origin in window px
V = [("current", "crops_PCR_I2_500"), ("membrane landscape", "crops_PCR_I2m_500"), ("membrane + 1 µm opening", "crops_PCR_I2mo_500")]
N = int(60 / UM); rng = np.random.default_rng(3)
fig, axs = plt.subplots(len(TILES), 4, figsize=(17, 4.4 * len(TILES)), dpi=150, gridspec_kw=dict(wspace=0.03, hspace=0.12))
for i, (w, ox, oy, name) in enumerate(TILES):
    x0, y0, _, _ = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]; X0, Y0 = x0 + ox, y0 + oy
    z = zarr.open(tifffile.TiffFile(SSD / "work/local_bundle_heldout/morphology_focus/ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
    im = np.asarray(z[:, Y0:Y0 + N, X0:X0 + N]).astype(float)
    mg = np.clip(np.clip((im[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((im[1] - 2) / 110, 0, 1)[..., None] * np.array([.92, .25, .78])
                 + np.clip((im[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]) * 0.6, 0, 1)
    axs[i, 0].imshow(mg); axs[i, 0].set_ylabel(f"{w}, {name}", fontsize=9.5)
    for j, (lab_, d) in enumerate(V, start=1):
        L = np.asarray(np.load(RUNS / d / w / "final_labels.npy", mmap_mode="r")[Y0:Y0 + N, X0:X0 + N]).astype(np.int64)
        u = np.unique(L); lut = {k: rng.uniform(0.35, 1, 3) for k in u}; out = mg * 0.4
        for k in u[u > 0]: m = L == k; out[m] = 0.45 * mg[m] + 0.55 * lut[k]
        out[find_boundaries(L, mode="inner") & (L > 0)] = 1.0
        axs[i, j].imshow(out, interpolation="none")
        if i == 0: axs[i, j].set_title(lab_, loc="left", fontsize=11, fontweight="bold")
    if i == 0: axs[i, 0].set_title("Morphology (DAPI, membrane magenta, 18S)", loc="left", fontsize=11, fontweight="bold")
    for a in axs[i]: a.set_xticks([]); a.set_yticks([]); a.plot([N - 57, N - 10], [N - 10] * 2, color="white", lw=2.5)
fig.savefig(SSD / "compare/purkinje/shape_compare.png", facecolor="white", bbox_inches="tight"); print("saved")
