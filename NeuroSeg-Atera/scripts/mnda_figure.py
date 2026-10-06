#!/usr/bin/env python3
"""MNDA figure (Rodrigo's layout), with panel b replaced: a = section coloured by lineage from our segmentation, boxes at the two
panel-b somata; b = 2x2, per soma merge (DAPI, 18S) | Purkinje-gene transcripts with Ranger (magenta) and our (gold) outlines;
c, d = Rodrigo's panels, cropped unchanged from Atera_future_work_MNDA.pdf. python mnda_figure.py <rendered page png> <out.pdf>"""
import sys, json, textwrap
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from PIL import Image
from scipy import sparse
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; D = SSD / "compare/purkinje"
page = Image.open(sys.argv[1]); s = page.size[0] / 1568
cimg = page.crop((int(745 * s), int(30 * s), int(1195 * s), int(535 * s))); dimg = page.crop((int(1212 * s), int(30 * s), int(1535 * s), int(535 * s)))
EX = [(7640.2, 2168.1), (3623.0, 3999.0)]; SIZE_UM = 70
# ---- panel a data: lineage from the annotation, Purkinje = gene-defined (>= 4.3% Purkinje-gene transcripts)
ann = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ann.index = ann.index.astype(str)
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str)
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); n = np.asarray(X.sum(1)).ravel()
fpc = np.asarray(X[:, np.isin(genes, gs["purkinje_A"] + gs["purkinje_B"])].sum(1)).ravel() / np.maximum(n, 1)
cells["pk"] = (n >= 200) & (fpc >= 0.043); cells["lin"] = cells.cell.astype(str).map(ann.lineage)
GROUP = {"Granule / UBC": "Granule layer", "Inhibitory interneuron": "Interneurons", "Astroglia": "Astroglia", "Oligodendroglia": "Oligodendroglia",
         "Immune": "Vascular, immune", "Vascular": "Vascular, immune", "Fibroblast / meningeal": "Vascular, immune"}
COL = {"Purkinje": "#2c6a9a", "Vascular, immune": "#5a5a5a", "Interneurons": "#a37cc4", "Astroglia": "#e0927a", "Oligodendroglia": "#77b56e", "Granule layer": "#cbbfa6"}
cells["grp"] = cells.lin.map(GROUP)
cells.loc[cells.pk, "grp"] = "Purkinje"
XM, Y0 = 14375, 1091                                            # Rodrigo's orientation: rows = section x (flipped), columns = section y
fig = plt.figure(figsize=(17.79, 8.83), dpi=300)
axA = fig.add_axes([0.012, 0.40, 0.155, 0.57])
sub = cells[cells.grp.notna() & (cells.grp != "Purkinje")].sample(220000, random_state=0)
for g in ["Granule layer", "Vascular, immune", "Interneurons", "Astroglia", "Oligodendroglia"]:
    q = sub[sub.grp == g]; axA.scatter(q.y_centroid_um - Y0, XM - q.x_centroid_um, s=0.06 if g == "Granule layer" else 0.5, color=COL[g], lw=0, rasterized=True)
q = cells[cells.grp == "Purkinje"]; axA.scatter(q.y_centroid_um - Y0, XM - q.x_centroid_um, s=2.2, color=COL["Purkinje"], lw=0, rasterized=True)
B = 300
for k, (ex, ey) in enumerate(EX, start=1):
    cx, cy = ey - Y0, XM - ex
    axA.add_patch(plt.Rectangle((cx - B / 2, cy - B / 2), B, B, fill=False, ec="black", lw=1.2)); axA.text(cx + B * 0.7, cy, str(k), fontsize=15, va="center")
axA.plot([1300, 3300], [-700, -700], color="black", lw=3); axA.text(2300, -380, "2 mm", ha="center", va="top", fontsize=15)
axA.set_aspect("equal"); axA.set_ylim(XM + 200, -1200); axA.axis("off")
axA.text(-0.02, 1.0, "a", transform=axA.transAxes, fontsize=24, fontweight="bold", va="top")
h = [plt.Line2D([], [], marker="o", ls="", color=COL[g], ms=9, label=g) for g in COL]
axA.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.07, 0.02), frameon=False, fontsize=10, handletextpad=0.1, labelspacing=0.12, ncol=2, columnspacing=0.3)
# ---- panel b: 2x2 image (rows = somata, columns = merge | masks)
FW, FH = 17.79, 8.83; bh = 0.64; bw = bh * FH / FW
axB = fig.add_axes([0.172, 0.33, bw, bh]); bimg = np.asarray(Image.open(D / "mnda_panel_b.png").convert("RGB")); axB.imshow(bimg); axB.axis("off")
Hb, Wb = bimg.shape[:2]
for k, yy in enumerate((0.03, 0.53), start=1): axB.text(0.02 * Wb, yy * Hb, str(k), color="white", fontsize=16, fontweight="bold", va="top")
pxum = (Wb / 2) / SIZE_UM; x1 = Wb * 0.985; axB.plot([x1 - 10 * pxum, x1], [Hb * 0.975] * 2, color="white", lw=3, solid_capstyle="butt")
axB.text(x1 - 11 * pxum, Hb * 0.975, "10 µm", color="white", ha="right", va="center", fontsize=11.5)
axB.text(-0.07, 1.0, "b", transform=axB.transAxes, fontsize=24, fontweight="bold", va="top")
hb = [plt.Line2D([], [], color="#cc3399", lw=3, label="Ranger"), plt.Line2D([], [], color="#e69f00", lw=3, label="This pipeline"),
      plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", ms=6, label="Purkinje-gene transcripts")]
axB.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False, fontsize=11, handlelength=1.4, columnspacing=1.0, labelspacing=0.15)
# ---- panels c, d unchanged
x0 = 0.172 + bw + 0.008
for im_ in (cimg, dimg):
    h_ = 0.64; w_ = h_ * FH / FW * im_.size[0] / im_.size[1]
    ax_ = fig.add_axes([x0, 0.33, w_, h_]); ax_.imshow(np.asarray(im_)); ax_.axis("off"); x0 += w_ + 0.006
cap = ("**Our segmentation pipeline keeps large neurons intact in whole-transcriptome Atera data.** We applied the segmentation pipeline we developed "
       "for motor neurons in ALS spinal cord to a human cerebellum section that 10x Genomics profiled on its Atera platform (18,050 genes, compared with 480 on our "
       "Xenium panel). Purkinje cells provide the test case because, like spinal motor neurons, they are large neurons that default segmentation tends to fragment. "
       f"**a**, Cerebellum section segmented with our pipeline. Colour marks the class of each of its {len(cells):,} cells, and the {int(cells.pk.sum()):,} Purkinje cells appear in blue. "
       "Scale bar, 2 mm. **b**, Two Purkinje somata (boxes 1 and 2 in a). Left, DAPI (blue) and 18S rRNA (yellow). Right, transcripts of Purkinje-specific genes (white) "
       "with Xenium Ranger, the default 10x segmentation (magenta), and our pipeline (gold); thick outlines mark the cells holding the soma, thin outlines all other cells. "
       "Ranger assigns only a small part of soma 1 and the lower part of soma 2 to a cell; our pipeline keeps each soma whole. Scale bar, 10 µm. "
       "**c**, Transcripts and genes per cell. In our ALS spinal cord Xenium data, cells hold a median of 32 transcripts and 22 genes, and motor neurons 146.5 and 51. "
       "In the Atera section, cells hold 4,787 and 2,996, and Purkinje cells 13,650 and 5,118. **d**, Purkinje soma capture across 139 somata. Our pipeline places a "
       "fraction of 0.77 of each soma in a single cell, against 0.51 for Xenium Ranger, and splits 7% of somata, against 28%. These data come from one section of one donor. "
       "We plan to apply the same pipeline to Atera sections of ALS spinal cord, where it should give each motor neuron whole-transcriptome depth with an intact soma.")
from matplotlib import mathtext
lines = textwrap.wrap(cap, 212); y = 0.245
for ln in lines:
    parts = ln.split("**"); x = 0.012; t = fig.text(x, y, "", fontsize=11)
    txt = "".join((f"$\\bf{{{p.replace(' ', '~').replace(',', '{,}').replace('-', '\u2010')}}}$" if i % 2 else p) for i, p in enumerate(parts))
    t.set_text(txt); y -= 0.025
fig.savefig(sys.argv[2], facecolor="white"); fig.savefig(sys.argv[2].replace(".pdf", ".png"), facecolor="white", dpi=200); print("saved", sys.argv[2])
