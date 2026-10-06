#!/usr/bin/env python3
"""Own nuclei vs 10x on 100 um zooms: DAPI | transcript nuclear score (no DAPI) | outlines (10x blue, ours orange, new filled)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib, pickle
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); import nuclei_detect as N
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; OUT = N.OUT
Z = [("spot1_purkinje_left", 200, 200), ("spot3_purkinje_right", 100, 170)]
w = pickle.load(open(OUT / "gene_weights.pkl", "rb"))
fig, axs = plt.subplots(2, 3, figsize=(16, 11), dpi=150, gridspec_kw=dict(wspace=0.03, hspace=0.12))
for i, (win, ox, oy) in enumerate(Z):
    d = N.load(win); sc, _ = N.score_map(d, w); r = np.load(OUT / f"{win}_nuclei.npz"); own = r["own"]; new = set(r["new"].tolist())
    s = (slice(d["pad"] + int(oy / UM), d["pad"] + int((oy + 100) / UM)), slice(d["pad"] + int(ox / UM), d["pad"] + int((ox + 100) / UM)))
    dp = d["dapi"][s]; scs = sc[s]; o = own[s]; t = d["nuc10x"][s]
    ext = (0, 100, 100, 0)
    axs[i, 0].imshow(dp, cmap="Greys_r", vmin=3, vmax=321, extent=ext)
    axs[i, 1].imshow(scs, cmap="magma", vmin=np.percentile(sc, 2), vmax=np.percentile(sc, 99.5), extent=ext)
    base = np.clip((dp - 3) / 318, 0, 1)[..., None] * np.ones(3) * 0.75
    nm = np.isin(o, list(new)); base[nm] = 0.4 * base[nm] + 0.6 * np.array(matplotlib.colors.to_rgb("#d95926"))
    base[find_boundaries(t, mode="inner") & (t > 0)] = matplotlib.colors.to_rgb("#3987e5")
    base[find_boundaries(o, mode="inner") & (o > 0)] = matplotlib.colors.to_rgb("#eda100")
    axs[i, 2].imshow(base, extent=ext, interpolation="none")
    for a in axs[i]: a.set_xticks([]); a.set_yticks([]); [x.set_visible(False) for x in a.spines.values()]; a.plot([85, 95], [96, 96], color="white", lw=2.5)
    rep = json.load(open(OUT / "nuclei_report.json"))[win]
    axs[i, 0].set_ylabel(win.replace("_", " "), fontsize=10)
    axs[i, 2].set_title(f"window: 10x {rep['tenx']:,} · ours {rep['own']:,} · matched {rep['matched_iou50']:,} · new {rep['own_new_no_10x_overlap']:,}", loc="left", fontsize=8.5, color="#52514e")
for j, tt in enumerate(["DAPI", "Transcript nuclear score (no DAPI)", "10x nuclei blue · ours yellow · new (no 10x overlap) orange fill"]):
    axs[0, j].text(0, 1.06, tt, transform=axs[0, j].transAxes, fontsize=11, fontweight="bold")
fig.subplots_adjust(top=0.93, bottom=0.01, left=0.03, right=0.995)
fig.savefig(OUT / "nuclei_compare.png", facecolor="white"); print("saved")
