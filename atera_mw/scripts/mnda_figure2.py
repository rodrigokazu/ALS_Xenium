#!/usr/bin/env python3
"""Poster figure, everything on our segmentations:
a section by lineage (Atera, our pipeline) | b 2x2 Purkinje somata, Ranger (pre-production) vs ours | c transcripts and genes per cell,
Xenium spinal cord (our v3i segmentation, 480 genes) vs Atera (our pipeline) | d Purkinje soma capture (Rodrigo's panel, cropped) |
e concept: the features our pipeline combines | f transcripts in cells, whole section.
python mnda_figure2.py <rendered MNDA page png> <out.pdf>"""
import sys, json, textwrap
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from PIL import Image
from scipy import sparse, ndimage as ndi, stats
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_hybridC"; D = SSD / "compare/purkinje"; B = SSD / "Cerebellum_sample"
SCR = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad")
RANGER, OURS, INK, MUTED = "#cc3399", "#e69f00", "#0b0b0b", "#52514e"
page = Image.open(sys.argv[1]); s = page.size[0] / 1568
dimg = page.crop((int(1212 * s), int(30 * s), int(1535 * s), int(535 * s)))
EX = [(7640.2, 2168.1), (3623.0, 3999.0)]; SIZE_UM = 70
# ---------------- data
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str)
ann = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ann.index = ann.index.astype(str)
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); n = np.asarray(X.sum(1)).ravel(); ng = np.diff(X.indptr)
fpc = np.asarray(X[:, np.isin(genes, gs["purkinje_A"] + gs["purkinje_B"])].sum(1)).ravel() / np.maximum(n, 1)
cells["n"], cells["ng"], cells["pk"] = n, ng, (n >= 200) & (fpc >= 0.043); cells["lin"] = cells.cell.astype(str).map(ann.lineage)
xen = pd.read_parquet(SCR / "xenium_percell.parquet"); xen = xen[xen.n_tx > 0]
# Purkinje in panel c = our cells holding the nuclei of 10x's Purkinje cluster (33), so both segmentations are compared on the same cells
cl33 = pd.read_csv(B / "analysis/clustering/gene_expression_graphclust/clusters.csv"); c33 = set(cl33.Barcode[cl33.Cluster == 33])
base = pd.read_parquet(SSD / "segmentation_full/base_cells.parquet"); pk_lab = set(base.label[base.nucleus_id.isin(c33)])
cells["pk10x"] = cells.cell.isin(pk_lab)
tq = pd.read_parquet(SSD / "segmentation_full/tenx_qc_counts_per_cell.parquet").qc_transcripts
pair = base[base.nucleus_id.isin(c33)].merge(cells[["cell", "n"]], left_on="label", right_on="cell"); pair["tenx"] = pair.nucleus_id.map(tq)
pk_ratio = float((pair.n / pair.tenx).median()); pk_n10x = len(c33)
summ, tsum = json.load(open(O / "summary.json")), json.load(open(SSD / "segmentation_full/tenx_summary.json"))
FW, FH = 17.8, 12.6
fig = plt.figure(figsize=(FW, FH), dpi=300)
ROW1, H1 = 0.535, 0.44
# ---------------- a
GROUP = {"Granule / UBC": "Granule layer", "Inhibitory interneuron": "Interneurons", "Astroglia": "Astroglia", "Oligodendroglia": "Oligodendroglia",
         "Immune": "Vascular, immune", "Vascular": "Vascular, immune", "Fibroblast / meningeal": "Vascular, immune"}
COL = {"Purkinje": "#2c6a9a", "Vascular, immune": "#5a5a5a", "Interneurons": "#a37cc4", "Astroglia": "#e0927a", "Oligodendroglia": "#77b56e", "Granule layer": "#cbbfa6"}
cells["grp"] = cells.lin.map(GROUP); cells.loc[cells.pk, "grp"] = "Purkinje"
XM, Y0 = 14375, 1091
axA = fig.add_axes([0.012, ROW1 + 0.075, 0.155, H1 - 0.07])
sub = cells[cells.grp.notna() & (cells.grp != "Purkinje")].sample(220000, random_state=0)
for g in ["Granule layer", "Vascular, immune", "Interneurons", "Astroglia", "Oligodendroglia"]:
    q = sub[sub.grp == g]; axA.scatter(q.y_centroid_um - Y0, XM - q.x_centroid_um, s=0.06 if g == "Granule layer" else 0.5, color=COL[g], lw=0, rasterized=True)
q = cells[cells.grp == "Purkinje"]; axA.scatter(q.y_centroid_um - Y0, XM - q.x_centroid_um, s=2.2, color=COL["Purkinje"], lw=0, rasterized=True)
for k, (ex, ey) in enumerate(EX, start=1):
    cx, cy = ey - Y0, XM - ex; axA.add_patch(plt.Rectangle((cx - 150, cy - 150), 300, 300, fill=False, ec="black", lw=1.2)); axA.text(cx + 210, cy, str(k), fontsize=15, va="center")
axA.plot([1300, 3300], [-700, -700], color="black", lw=3); axA.text(2300, -380, "2 mm", ha="center", va="top", fontsize=15)
axA.set_aspect("equal"); axA.set_ylim(XM + 200, -1200); axA.axis("off")
axA.text(-0.02, 1.02, "a", transform=axA.transAxes, fontsize=24, fontweight="bold", va="top")
axA.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[g], ms=8, label=g) for g in COL], loc="upper left", bbox_to_anchor=(-0.07, 0.02),
           frameon=False, fontsize=10, handletextpad=0.1, labelspacing=0.12, ncol=2, columnspacing=0.3)
# ---------------- b
bh = H1 - 0.01; bw = bh * FH / FW
axB = fig.add_axes([0.172, ROW1 + 0.01, bw, bh]); bimg = np.asarray(Image.open(D / "mnda_panel_b.png").convert("RGB")); axB.imshow(bimg); axB.axis("off")
Hb, Wb = bimg.shape[:2]
for k, yy in enumerate((0.03, 0.53), start=1): axB.text(0.02 * Wb, yy * Hb, str(k), color="white", fontsize=16, fontweight="bold", va="top")
pxum = (Wb / 2) / SIZE_UM; x1 = Wb * 0.985; axB.plot([x1 - 10 * pxum, x1], [Hb * 0.975] * 2, color="white", lw=3, solid_capstyle="butt")
axB.text(x1 - 11 * pxum, Hb * 0.975, "10 µm", color="white", ha="right", va="center", fontsize=11.5)
axB.text(-0.06, 1.0, "b", transform=axB.transAxes, fontsize=24, fontweight="bold", va="top")
axB.legend(handles=[plt.Line2D([], [], color=RANGER, lw=3, label="Xenium Ranger (pre-production)"), plt.Line2D([], [], color=OURS, lw=3, label="This pipeline"),
                    plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", ms=6, label="Purkinje-gene transcripts")],
           loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False, fontsize=10.5, handlelength=1.3, columnspacing=0.8)
# ---------------- c: ridges
xc0 = 0.172 + bw + 0.072
rows = [("All cells", xen.n_tx, xen.ng, "#c4c4c4", MUTED), ("MNs", xen.n_tx[xen.is_MN == "True"], xen.ng[xen.is_MN == "True"], "#7f7f7f", MUTED),
        ("All cells", cells.n[cells.n > 0], cells.ng[cells.n > 0], "#a3bdd4", "#2c6a9a"), ("Purkinje", cells.n[cells.pk10x], cells.ng[cells.pk10x], "#2c6a9a", "#2c6a9a")]
for j, (col_i, xlabel) in enumerate(((1, "Transcripts per cell"), (2, "Genes per cell"))):
    ax = fig.add_axes([xc0, ROW1 + 0.06 + (1 - j) * 0.205, 0.19, 0.16])
    grid = np.linspace(0, 5, 400)
    for i, r in enumerate(rows):
        v = np.log10(np.asarray(r[col_i], float)); v = v[np.isfinite(v)]
        kde = stats.gaussian_kde(v if len(v) < 60000 else np.random.default_rng(0).choice(v, 60000, replace=False), bw_method=0.18)(grid); kde /= kde.max()
        base = 3 - i; m = kde > 0.02
        ax.fill_between(grid[m], base, base + 0.8 * kde[m], color=r[3], lw=0)
        med = np.median(np.asarray(r[col_i])); ax.plot([np.log10(med)] * 2, [base, base + 0.8 * kde[np.argmin(abs(grid - np.log10(med)))]], color=INK, lw=2.5)
        lbl = f"{med:,.0f}" if med >= 100 or float(med).is_integer() else f"{med:,.1f}"
        hi = grid[m].max() if i < 2 else grid[m].min()
        ax.text(hi + (0.12 if i < 2 else -0.12), base + 0.3, lbl, ha="left" if i < 2 else "right", va="center", fontsize=13, color=INK)
        ax.text(-0.04, base + 0.3, r[0], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14, color=r[4])
    ax.set_xlim(0, 5); ax.set_ylim(-0.1, 4); ax.set_yticks([]); ax.spines["left"].set_visible(True)
    ax.set_xticks(range(6), ["1", "10", "100", "1k", "10k", "100k"], fontsize=13); ax.set_xlabel(xlabel, fontsize=15)
    if j == 0:
        ax.text(-0.33, 1.25, "c", transform=ax.transAxes, fontsize=24, fontweight="bold", va="top")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#7f7f7f", label="Xenium, 480 genes"), plt.Rectangle((0, 0), 1, 1, color="#2c6a9a", label="Atera, 18,050 genes")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=13, handlelength=1.2)
# ---------------- d (Rodrigo's, unchanged)
h_ = H1 - 0.02; w_ = h_ * FH / FW * dimg.size[0] / dimg.size[1]
axD = fig.add_axes([0.995 - w_, ROW1 + 0.02, w_, h_]); axD.imshow(np.asarray(dimg)); axD.axis("off")
# ---------------- e: concept strip on one 50 um tile around soma 2
cx, cy = EX[1]; half = int(25 / UM); X0, Yp0 = int(cx / UM) - half, int(cy / UM) - half; N = 2 * half
z = zarr.open(tifffile.TiffFile(B / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r"); im = np.asarray(z[:, Yp0:Yp0 + N, X0:X0 + N]).astype(float)
dens = np.asarray(zarr.open_array(str(SSD / "work/density/tx_density.zarr"), mode="r")[Yp0:Yp0 + N, X0:X0 + N]).astype(float)
lab = np.asarray(zarr.open_array(str(O / "final_labels.zarr"), mode="r")[Yp0:Yp0 + N, X0:X0 + N]).astype(np.int64)
de = pd.read_csv(B / "analysis/diffexp/gene_expression_graphclust/differential_expression.csv"); cl = pd.read_csv(B / "analysis/clustering/gene_expression_graphclust/clusters.csv")
GRP = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19], "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
ncl = cl.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
Mm = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)]); inf = ((Lf >= 2) & (Mm >= 0.3) & (Pv < 0.01)).any(1)
prof = np.array([(Mm[inf][:, np.array(k) - 1] * ncl[np.array(k) - 1]).sum(1) / ncl[np.array(k) - 1].sum() for k in GRP.values()]) + 1e-4; prof /= prof.sum(1, keepdims=True); logp = np.log(prof)
gidx = pd.Index(de["Feature Name"][inf])
tx = ds.dataset(B / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < (X0 + N) * UM)
     & (pc.field("y_location") >= Yp0 * UM) & (pc.field("y_location") < (Yp0 + N) * UM) & (pc.field("qv") >= 20)).to_pandas()
gi = gidx.get_indexer(tx.feature_name.astype(str)); ok = gi >= 0
rr = np.clip((tx.y_location[ok] / UM).astype(int) - Yp0, 0, N - 1); cc = np.clip((tx.x_location[ok] / UM).astype(int) - X0, 0, N - 1)
S = np.stack([ndi.gaussian_filter(np.bincount(rr * N + cc, weights=logp[t][gi[ok]], minlength=N * N).reshape(N, N), 2 / UM) for t in range(6)])
zz = 3.5 * (S - S.max(0)) * (0.5 / UM) ** 2                                  # stage-1 tau, evidence rescaled from 0.5 um bins to 0.2125 um pixels
Pt = np.exp(zz); Pt /= Pt.sum(0)
TCOL = np.array([matplotlib.colors.to_rgb(c) for c in ["#cbbfa6", "#2c6a9a", "#e0927a", "#77b56e", "#5a5a5a", "#5a5a5a"]])
ident = np.einsum("tyx,tc->yxc", Pt, TCOL) * np.clip(dens / 60, 0, 1)[..., None]
dapi = np.clip((im[0] - 3) / 160, 0, 1); r18 = np.clip((im[2] - 2) / 606, 0, 1)
segimg = np.clip(dapi[..., None] * np.array([.22, .42, 1]) + r18[..., None] * np.array([1, .82, .12]) * 0.8, 0, 1) * 0.55
bd = find_boundaries(lab, mode="inner") & (lab > 0)
segimg[bd & (lab < 2_000_000_000)] = matplotlib.colors.to_rgb(OURS)                  # cells kept from our segmentation
segimg[bd & (lab >= 2_000_000_000) & (lab < 3_000_000_000)] = matplotlib.colors.to_rgb(RANGER)   # glia / vascular / immune from Ranger
tiles = [(dapi[..., None] * np.array([.22, .42, 1]), "1  Nuclei", "DAPI: seeds"), (r18[..., None] * np.array([1, .82, .12]), "2  Cell bodies", "18S: split / merge rules"),
         (plt.cm.magma(np.clip(ndi.gaussian_filter(dens, 0.5 / UM) / 220, 0, 1))[..., :3], "3  Transcript density", "tissue, cytoplasm growth"),
         (np.clip(ident, 0, 1), "4  Transcript identity", "type-gated growth,\nPurkinje repair"), (segimg, "5  Final cells", "neurons ours (gold),\nglia Ranger (magenta)")]
ew = 0.085; eh = ew * FW / FH; ey = 0.33; ex0 = 0.03
for k, (img, t1, t2) in enumerate(tiles):
    xk = ex0 + k * (ew + 0.03); ax = fig.add_axes([xk, ey, ew, eh]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_color("#c3c2b7")
    fig.text(xk + ew / 2, ey - 0.012, t1, ha="center", va="top", fontsize=13, fontweight="bold"); fig.text(xk + ew / 2, ey - 0.032, t2, ha="center", va="top", fontsize=11.5, color=MUTED, linespacing=1.15)
    if k < 4: fig.text(xk + ew + 0.015, ey + eh / 2, "+" if k < 3 else "→", ha="center", va="center", fontsize=26, color=MUTED)
fig.text(ex0 - 0.018, ey + eh + 0.012, "e", fontsize=24, fontweight="bold", va="bottom")
# ---------------- f: transcripts in cells, whole section
axF = fig.add_axes([0.845, ey + 0.005, 0.13, eh - 0.01])
vals = [100 * tsum["tenx_assigned"] / tsum["n_qc_tx"], summ["pct_assigned"]]
axF.bar([0, 1], vals, color=[RANGER, OURS], width=0.62)
for i, v in enumerate(vals): axF.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=14)
axF.set_ylim(80, 91); axF.set_xticks([0, 1], ["Ranger", "This\npipeline"], fontsize=13); axF.set_ylabel("Transcripts in cells (%)", fontsize=14); axF.tick_params(labelsize=13)
axF.text(0.5, 1.02, f"+{(summ['n_assigned'] - tsum['tenx_assigned']) / 1e6:.0f} M transcripts", transform=axF.transAxes, ha="center", va="top", fontsize=12.5, color=MUTED)
fig.text(0.80, ey + eh + 0.05, "g", fontsize=24, fontweight="bold", va="top")
# ---------------- g: UMAP of the annotated cells, colours as in a
ua = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ua.index = ua.index.astype(str)
ua["grp"] = ua.lineage.map(GROUP); ua.loc[ua.index.isin(cells.cell[cells.pk].astype(str)), "grp"] = "Purkinje"
ua = ua[ua.grp.notna()]; n_ua = len(ua); n_gran = int((ua.grp == "Granule layer").sum())
GRAN_KEEP = 0.1                                                    # granule cells subsampled to 10% for display; all other cells shown
ua = pd.concat([ua[ua.grp == "Granule layer"].sample(frac=GRAN_KEEP, random_state=0), ua[ua.grp != "Granule layer"]]).sample(frac=1, random_state=0)
axG = fig.add_axes([0.63, ey - 0.045, 0.17, 0.17 * FW / FH])
for g in ["Granule layer", "Vascular, immune", "Interneurons", "Astroglia", "Oligodendroglia"]:
    q = ua[ua.grp == g]; axG.scatter(q.umap1, q.umap2, s=0.15 if g == "Granule layer" else 0.25, color=COL[g], lw=0, rasterized=True)
q = ua[ua.grp == "Purkinje"]; axG.scatter(q.umap1, q.umap2, s=2.5, color=COL["Purkinje"], lw=0, rasterized=True)
axG.set_xticks([]); axG.set_yticks([]); axG.set_xlabel("UMAP 1", fontsize=12); axG.set_ylabel("UMAP 2", fontsize=12)
axG.text(-0.12, 1.08, "f", transform=axG.transAxes, fontsize=24, fontweight="bold", va="top")
axG.text(0.02, 0.98, f"{n_ua:,} cells\ngranule cells: 10% shown", transform=axG.transAxes, va="top", fontsize=10.5, color=MUTED)
# ---------------- caption
xm = lambda v: np.median(np.asarray(v))
cap = ("**Our segmentation pipeline keeps large neurons intact in whole-transcriptome Atera data.** We applied the segmentation pipeline we developed for motor neurons "
       "in ALS spinal cord to a pre-production human cerebellum section that 10x Genomics profiled on its Atera platform (18,050 genes, compared with 480 on our Xenium "
       "panel). Purkinje cells provide the test case because, like spinal motor neurons, they are large neurons that segmentation tends to fragment. "
       f"**a**, Cerebellum section segmented with our pipeline. Colour marks the class of each of its {len(cells):,} cells; the {int(cells.pk.sum()):,} Purkinje cells appear in blue. "
       "Scale bar, 2 mm. **b**, Two Purkinje somata (boxes 1 and 2 in a). Left, DAPI (blue) and 18S rRNA (yellow). Right, transcripts of Purkinje-specific genes (white) with the "
       "pre-production Xenium Ranger segmentation (magenta) and our pipeline (gold); thick outlines mark the cells holding the soma. Our pipeline keeps each soma as one cell. "
       f"Scale bar, 10 µm. **c**, Transcripts and genes per cell, both datasets segmented with our pipeline. ALS spinal cord Xenium: all cells {xm(xen.n_tx):,.0f} transcripts and "
       f"{xm(xen.ng):,.0f} genes, motor neurons {xm(xen.n_tx[xen.is_MN == 'True']):,.0f} and {xm(xen.ng[xen.is_MN == 'True']):,.0f}. Atera: all cells {xm(cells.n[cells.n > 0]):,.0f} and "
       f"{xm(cells.ng[cells.n > 0]):,.0f}, Purkinje cells {xm(cells.n[cells.pk10x]):,.0f} and {xm(cells.ng[cells.pk10x]):,.0f} (our cells holding {int(cells.pk10x.sum())} of the {pk_n10x} nuclei of 10x\'s Purkinje cluster; for the same nuclei they hold {pk_ratio:.2f} times the transcripts of the Ranger cells). **d**, Purkinje soma capture across 139 somata: our pipeline places "
       "0.77 of each soma in a single cell (Ranger 0.51) and splits 7% of somata (Ranger 28%). **e**, The pipeline, shown on soma 2: (1) DAPI nuclei seed the cells; (2) 18S rRNA "
       "outlines cell bodies, which rule-based splitting and merging resolve; (3) transcript density marks tissue and lets cells grow into their cytoplasm; (4) the local cell "
       "type read from the transcripts limits growth to pixels of the cell's own type and rebuilds each Purkinje soma as one cell; (5) neurons and granule cells come from "
       "this pipeline, glial, vascular and immune cells from Ranger, which captures them slightly better. **g**, Transcripts assigned to a cell over the whole "
       f"section: {summ['pct_assigned']:.1f}% with our pipeline against {vals[0]:.1f}% with Ranger. One section of one donor. We plan to apply the same pipeline to Atera sections of "
       "ALS spinal cord, where it should give each motor neuron whole-transcriptome depth with an intact soma.")
cap = cap.replace(" **g**, Transcripts", f" **f**, UMAP of the {n_ua:,} annotated cells, coloured as in a; granule cells ({n_gran:,}) are subsampled to 10% for display. **g**, Transcripts")
y = 0.215
for ln in textwrap.wrap(cap, 205):
    parts = ln.split("**"); txt = "".join((f"$\\bf{{{p.replace(' ', '~').replace(',', '{,}').replace('-', chr(0x2010))}}}$" if i % 2 else p) for i, p in enumerate(parts))
    fig.text(0.012, y, txt, fontsize=11.5); y -= 0.0175
fig.savefig(sys.argv[2], facecolor="white", bbox_inches="tight", pad_inches=0.15); fig.savefig(sys.argv[2].replace(".pdf", ".png"), facecolor="white", dpi=200, bbox_inches="tight", pad_inches=0.15); print("saved", sys.argv[2])
print({"xen_all": (xm(xen.n_tx), xm(xen.ng)), "xen_MN": (xm(xen.n_tx[xen.is_MN == "True"]), xm(xen.ng[xen.is_MN == "True"])), "atera_all": (xm(cells.n[cells.n > 0]), xm(cells.ng[cells.n > 0])),
       "atera_pk": (xm(cells.n[cells.pk10x]), xm(cells.ng[cells.pk10x]), int(cells.pk10x.sum()), pk_ratio)})
