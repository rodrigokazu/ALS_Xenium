#!/usr/bin/env python3
"""Summary of the biological readouts from the Atera cerebellum section, each panel tagged with its status.
A ranking of dendritic vs soma-restricted Purkinje mRNAs | B ITPR1 vs CA8 transcripts across Purkinje -> molecular layer |
C replication between the two halves of the section | D reach into the molecular layer | E neuropil shift by source cell type | F GJD2 (connexin 36) by cell type (validation).
python findings_fig.py"""
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); Q = SSD / "segmentation_full/open_questions"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
TAG = {"new": ("POTENTIALLY NEW", "#1c5cab", "#cde2fb"), "known": ("KNOWN BIOLOGY RECOVERED", "#52514e", "#ecebe5"), "no": ("TESTED, NOT SUPPORTED", "#9f3b17", "#fbe0d4")}
def tag(ax, kind, x=0.0, y=1.13):
    t, fc, bc = TAG[kind]; ax.text(x, y, t, transform=ax.transAxes, fontsize=8, fontweight="bold", color=fc, bbox=dict(boxstyle="round,pad=0.3", fc=bc, ec="none"))
pcd = pd.read_csv(Q / "purkinje_dendritic_index.csv"); rb = pd.read_csv(Q / "dendritic_index_robustness.csv"); allc = pd.read_csv(Q / "dendritic_index_all_classes.csv"); con = pd.read_csv(Q / "connexins_by_type.csv")
from scipy import stats
fig = plt.figure(figsize=(17, 11.8), dpi=150); g = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.95], height_ratios=[1, 0.9], hspace=0.45, wspace=0.42)
# A ranking
ax = fig.add_subplot(g[0, 0]); rk = rb.sort_values("di_adj", ascending=False).reset_index(drop=True)
LIT = {"PCP2": "known dendritic", "RGS8": "known dendritic", "ITPR1": "known dendritic", "CALB1": "known soma-only"}
sel = pd.concat([rk.head(10), rk.tail(8)]).iloc[::-1]
ax.barh(range(len(sel)), sel.di_adj, color=[BLUE if v > 0 else ORANGE for v in sel.di_adj], height=0.7)
ax.text(0.02, 0.98, "literature check:\nPCP2 +1.4, RGS8 +1.1, ITPR1 +0.1\n(known dendritic)\nCALB1 −0.3 (soma-only by FISH)", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
ax.set_yticks(range(len(sel)), sel.gene); ax.axvline(0, color=INK, lw=0.8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("dendritic index, corrected for expression level\nlog2 (molecular-layer neuropil / Purkinje somata)")
ax.set_title(f"A  Purkinje mRNAs in the dendrites vs the soma\n10 most dendritic and 8 most soma-restricted of {len(pcd)} Purkinje-specific genes", loc="left", fontsize=10.5, fontweight="bold")
tag(ax, "new", y=1.16)
# B transcripts
ax = fig.add_subplot(g[0, 1]); UM = 0.2125; X0, Y0, S = 46824 + 250, 17365 + 950, int(220 / UM)
z = zarr.open(tifffile.TiffFile(SSD / "Cerebellum_sample/morphology_focus/ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
im = np.asarray(z[0, Y0:Y0 + S, X0:X0 + S]).astype(float)
t = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < (X0 + S) * UM)
    & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < (Y0 + S) * UM) & (pc.field("qv") >= 20) & pc.field("feature_name").isin(["ITPR1", "CA8"])).to_pandas()
ax.imshow(np.clip((im - 3) / 318, 0, 1), cmap="Greys", extent=(0, S * UM, S * UM, 0), alpha=0.35)
for gname, colr in (("CA8", ORANGE), ("ITPR1", BLUE)):
    q = t[t.feature_name == gname]; ax.scatter(q.x_location - X0 * UM, q.y_location - Y0 * UM, s=2.6, color=colr, lw=0, label=f"{gname} ({len(q):,})")
ax.set_xlim(0, S * UM); ax.set_ylim(S * UM, 0); ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=True, loc="lower right", markerscale=5, fontsize=9, facecolor="white", edgecolor="none")
ax.text(4, 10, "Purkinje layer", fontsize=8.5, color=MUTED); ax.text(4, S * UM - 8, "molecular layer", fontsize=8.5, color=MUTED)
ax.plot([S * UM - 58, S * UM - 8], [S * UM - 28] * 2, color=INK, lw=2.5); ax.text(S * UM - 33, S * UM - 32, "50 µm", ha="center", fontsize=8.5)
ax.set_title("B  ITPR1 spreads into the molecular layer, CA8 stays\nnear the somata (220 µm, nuclei grey)", loc="left", fontsize=10.5, fontweight="bold")
tag(ax, "new", y=1.16)
# C replication between halves
ax = fig.add_subplot(g[0, 2]); r_h = stats.spearmanr(rb.di_left, rb.di_right).statistic
ax.scatter(rb.di_left, rb.di_right, s=22, color=BLUE, lw=0); lim = [min(rb.di_left.min(), rb.di_right.min()) - 0.3, max(rb.di_left.max(), rb.di_right.max()) + 0.3]
ax.plot(lim, lim, color="#898781", lw=0.8, ls="--"); ax.set_xlim(lim[0], lim[1] + 1.2); ax.set_ylim(lim)
for _, r in rb[rb.gene.isin(["ARHGEF33", "GRID2IP", "ITPR1", "CA8", "CNTN3"])].iterrows(): ax.annotate(r.gene, (r.di_left, r.di_right), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.set_xlabel("dendritic index, left half of the section"); ax.set_ylabel("dendritic index, right half"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title(f"C  The ranking replicates between the two halves\nSpearman ρ = {r_h:.2f}, {len(rb)} genes", loc="left", fontsize=10.5, fontweight="bold"); tag(ax, "new", y=1.16)
# D reach into the molecular layer
ax = fig.add_subplot(g[1, 0]); r_f = stats.spearmanr(rb.di_rel, rb.share_beyond_75um).statistic; r_a = stats.spearmanr(rb.di_rel, np.log(rb.S_PC)).statistic
ax.scatter(rb.di_rel, 100 * rb.share_beyond_75um, s=22, color=BLUE, lw=0)
for _, r in rb[rb.gene.isin(["ARHGEF33", "GRID2IP", "DAGLA", "ITPR1", "CA8", "CNTN3", "RSPO3"])].iterrows(): ax.annotate(r.gene, (r.di_rel, 100 * r.share_beyond_75um), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.set_xlabel("dendritic index (uncorrected)"); ax.set_ylabel("molecular-layer neuropil > 75 µm from the granule layer (%)"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title(f"D  Dendritic mRNAs reach the outer molecular layer\nρ = {r_f:.2f}; index vs expression level ρ = {r_a:.2f}", loc="left", fontsize=10.5, fontweight="bold"); tag(ax, "new", y=1.16)
# E: is the neuropil Purkinje-derived? enrichment near interneuron and granule cell bodies
ax = fig.add_subplot(g[1, 1]); px = pd.read_csv(Q / "neuropil_MLI_granule_proximity.csv").set_index("feature_name")
grp = [("interneuron genes", ["KIT", "LYPD6"]), ("granule genes", ["GABRA6", "CBLN3"]), ("known Purkinje dendritic", ["PCP2", "RGS8"]), ("new: KCNC3, CACNA1G", ["KCNC3", "CACNA1G"])]
for i, (lab_, gg) in enumerate(grp):
    for j, (col, colr, off) in enumerate((("enrich_near_MLI", "#1baf7a", -0.18), ("enrich_near_granule", "#898781", 0.18))):
        v = px.loc[gg, col].to_numpy(); ax.scatter(np.full(len(v), i + off), np.log2(v + 0.05), s=34, color=colr, zorder=3, label=(["near interneuron", "near granule cell"][j] if i == 0 else None))
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(len(grp)), [x[0] for x in grp], rotation=12, ha="right")
ax.text(-0.45, np.log2(0.05) + 0.25, "none near", fontsize=7.5, color=MUTED)
ax.set_ylabel("log2 enrichment of molecular-layer neuropil\nwithin 4 µm of a cell body (0 = no preference)"); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("E  KCNC3 / CACNA1G neuropil is not from\ninterneurons or granule cells (9 windows)", loc="left", fontsize=10.5, fontweight="bold"); tag(ax, "new", y=1.16)
# F GJD2
ax = fig.add_subplot(g[1, 2]); gj = con[con.gene == "GJD2"].sort_values("pct_cells")
ax.barh(range(len(gj)), gj.pct_cells, color=[BLUE if t_ == "MLI (basket / stellate)" else "#b7d3f6" for t_ in gj.cell_type], height=0.7)
ax.set_yticks(range(len(gj)), gj.cell_type); ax.set_xlabel("cells with ≥ 1 GJD2 transcript (%)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for i, v in enumerate(gj.pct_cells): ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_title("F  Connexin 36 (GJD2) is confined to\nbasket / stellate interneurons", loc="left", fontsize=10.5, fontweight="bold"); tag(ax, "known", y=1.16)
fig.text(0.01, 0.995, "Atera human cerebellum (18,050 genes, one section, one donor): dendritic mRNA localisation in human Purkinje cells", fontsize=13.5, fontweight="bold", va="top")
fig.text(0.01, 0.968, "Showcase level: one section, replicated only between its two halves, no orthogonal validation. Purkinje-specific genes only (10x Purkinje cluster ≥ 3× every other cluster), so molecular-layer neuropil can be read as Purkinje dendrite.",
         fontsize=9.5, color=MUTED, va="top")
fig.subplots_adjust(top=0.86, bottom=0.05, left=0.07, right=0.99)
fig.savefig(Q / "biological_findings_summary.png", facecolor="white"); print("saved")
