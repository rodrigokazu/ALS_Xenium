#!/usr/bin/env python3
"""Benchmark figure for the Atera cerebellum: Xenium Ranger (pre-production) vs our segmentation vs the final hybrid
(ours for neurons and granule cells, Ranger for glia / vascular / immune; rule C in hybrid_seg.py).
A-C whole windows never used for tuning (spot3 + held1-6): coverage, MECR, MECR at 1,000 tx; Wilcoxon signed-rank vs Ranger.
D-E Purkinje somata (139, 9 windows, scored with held-out set-B genes): capture in one cell, split / missed; Wilcoxon, McNemar.
F-G every 10x nucleus with >= 20 own-type specific transcripts: capture and split by cell type.
H ResolVI (Nature Methods 2026) on Ranger and on our segmentation, 27 windows (20 random new + 7 held-out): MECR before / after.
python benchmark_atera.py <out.pdf>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from scipy import stats
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare")
RANGER, OURS, FINAL, INK, MUTED, GRID = "#cc3399", "#f2c879", "#e69f00", "#0b0b0b", "#52514e", "#e1e0d9"
M3 = [("Ranger", "Xenium Ranger\n(pre-production)", RANGER), ("final", "This pipeline", FINAL)]          # ours = final pipeline, all steps
TUNE = {"spot1_purkinje_left", "spot2_gl_wm_centre"}
w1 = pd.read_csv(C / "purkinje/win_scores_hybridC.csv"); w2 = pd.read_csv(C / "purkinje/win_scores_pcr_v2.csv")
w = pd.concat([w1[w1.method.isin(["10x", "HYBC"])], w2[w2.method == "PCR_I2"]]).replace({"method": {"10x": "Ranger", "HYBC": "final", "PCR_I2": "ours"}})
w = w[~w.window.isin(TUNE)]
pcs = pd.read_csv(C / "purkinje_hybrid/pc_scores.csv").replace({"method": {"10x": "Ranger", "I2 + Purkinje repair": "ours", "Hybrid C": "final"}})
pcs = pcs[pcs.method.isin(["Ranger", "ours", "final"])]
ct = pd.read_csv(C / "celltype_capture/celltype_capture.csv")
RVF = C / "resolvi/resolvi_mecr_per_window_final.csv"; rv = pd.read_csv(RVF, header=[0, 1], index_col=0) if RVF.exists() else None
def p_fmt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
fig = plt.figure(figsize=(17.5, 11.2), dpi=220); gsp = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.42)
# ---- A-C windows
for k, (col, lab, fmt) in enumerate((("pct_assigned", "Transcripts in cells (%)", "{:.1f}"), ("mecr", "MECR (lower = cleaner)", "{:.4f}"), ("mecr_n1000", "MECR at 1,000 tx / cell", "{:.4f}"))):
    ax = fig.add_subplot(gsp[0, k]); pv = w.pivot(index="window", columns="method", values=col)[["Ranger", "final"]]
    for _, r in pv.iterrows(): ax.plot([0, 1], r.to_numpy(), color="#c3c2b7", lw=0.8, zorder=1)
    for i, (m, _, colr) in enumerate(M3): ax.scatter(np.full(len(pv), i), pv[m], s=40, color=colr, zorder=3, edgecolor="white", lw=0.6)
    for i, m in enumerate(["final"], start=1):
        p = stats.wilcoxon(pv[m], pv.Ranger).pvalue; better = (pv[m] > pv.Ranger).sum() if k == 0 else (pv[m] < pv.Ranger).sum()
        ax.text(i, ax.get_ylim()[1] if False else pv.max().max() + (pv.max().max() - pv.min().min()) * 0.12, f"{better}/{len(pv)} better\n{p_fmt(p)}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks([0, 1], ["Ranger", "This\npipeline"]); ax.set_xlim(-0.5, 1.5); ax.set_ylabel(lab); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(pv.min().min() - (pv.max().max() - pv.min().min()) * 0.1, pv.max().max() + (pv.max().max() - pv.min().min()) * 0.45)
    ax.set_title(["A  Coverage", "B  Marker mixing", "C  Mixing at equal depth"][k] + "\n7 windows not used for tuning", loc="left", fontsize=12, fontweight="bold")
# ---- D Purkinje capture
ax = fig.add_subplot(gsp[0, 3]); pv = pcs.pivot_table(index=["window", "soma"], columns="method", values="capture")[["Ranger", "final"]]
for i, (m, _, colr) in enumerate(M3):
    v = pv[m].to_numpy(); ax.scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=9, color=colr, alpha=0.7, lw=0)
    ax.plot([i - .22, i + .22], [np.median(v)] * 2, color=INK, lw=2.5); ax.text(i + .26, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=9.5)
pf = stats.wilcoxon(pv.final, pv.Ranger).pvalue
ax.set_xticks([0, 1], ["Ranger", "This\npipeline"]); ax.set_xlim(-0.5, 1.6); ax.set_ylabel("Purkinje-soma transcripts in one cell (bar = median)"); ax.set_ylim(-0.03, 1.12); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.text(0.5, 1.05, f"{p_fmt(pf)}", ha="center", fontsize=9, color=MUTED)
ax.set_title(f"D  Purkinje somata ({len(pv)})\nheld-out genes", loc="left", fontsize=12, fontweight="bold")
# ---- E split / missed
ax = fig.add_subplot(gsp[1, 0]); nc = pcs.pivot_table(index=["window", "soma"], columns="method", values="n_cells")[["Ranger", "final"]]
for j, (lab, fn) in enumerate((("split ≥ 2 cells", lambda v: v >= 2), ("no cell ≥ 10%", lambda v: v == 0))):
    for i, (m, _, colr) in enumerate(M3):
        v = 100 * fn(nc[m]).mean(); ax.bar(j + (i - 0.5) * 0.36, v, width=0.34, color=colr); ax.text(j + (i - 0.5) * 0.36, v + 0.6, f"{v:.0f}", ha="center", fontsize=9)
    a, b = fn(nc.Ranger), fn(nc.final); bb, cc = int((a & ~b).sum()), int((~a & b).sum())
    ax.text(j, 33, f"McNemar\n{p_fmt(stats.binomtest(bb, bb + cc).pvalue)}", ha="center", fontsize=9, color=MUTED)
ax.set_xticks([0, 1], ["Split across\n≥ 2 cells", "No cell holds\n≥ 10%"]); ax.set_ylabel("Purkinje somata (%)"); ax.set_ylim(0, 40); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("E  Fragmented and missed somata", loc="left", fontsize=12, fontweight="bold")
# ---- F-G cell types
order = ["Purkinje", "MLI (basket/stellate)", "Astroglia", "Oligodendroglia", "Vascular", "Immune"]
ax = fig.add_subplot(gsp[1, 1]); ax2 = fig.add_subplot(gsp[1, 2])
for t_i, t in enumerate(order):
    g = ct[ct.type == t]
    for i, (key, colr) in enumerate((("tenx", RANGER), ("HYBC", FINAL))):
        ax.scatter(g[f"capture_{key}"].median(), t_i + (i - 0.5) * 0.3, s=60, color=colr, zorder=3, edgecolor="white", lw=0.6)
        ax2.barh(t_i + (i - 0.5) * 0.36, 100 * (g[f"ncells_{key}"] >= 2).mean(), height=0.34, color=colr)
    pc_ = stats.wilcoxon(g.capture_HYBC, g.capture_tenx).pvalue if (g.capture_HYBC != g.capture_tenx).any() else 1
    ax.text(1.005, t_i, p_fmt(pc_), transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED, va="center")
    for a_ in (ax, ax2): a_.text(-0.02, t_i, f"n = {len(g):,}", transform=a_.get_yaxis_transform(), ha="right", va="center", fontsize=0)
lbls = [f"{t.replace(' (basket/stellate)', '')}\n(n = {int((ct.type == t).sum()):,})" for t in order]
ax.set_yticks(range(len(order)), lbls); ax.invert_yaxis(); ax.set_xlabel("own-type transcripts in one cell (median)"); ax.set_xlim(0.65, 1.03); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("F  Capture by cell type\nevery 10x nucleus", loc="left", fontsize=12, fontweight="bold")
ax2.set_yticks(range(len(order)), [""] * len(order)); ax2.invert_yaxis(); ax2.set_xlabel("cells split across ≥ 2 cells (%)"); ax2.grid(axis="x", color=GRID, lw=0.6); ax2.set_axisbelow(True)
ax2.set_title("G  Fragmentation by cell type", loc="left", fontsize=12, fontweight="bold")
# ---- H ResolVI
ax = fig.add_subplot(gsp[1, 3])
if rv is None: ax.axis("off"); ax.text(0.5, 0.5, "ResolVI on the final\npipeline: running", ha="center", transform=ax.transAxes, color=MUTED)
if rv is not None:
    for i, (k, colr, lab) in enumerate((("tenx", RANGER, "Ranger"), ("ours", FINAL, "This pipeline"))):
        a, b = rv[(k, "mecr_raw")], rv[(k, "mecr_resolvi")]
        for x0_, y0_, y1_ in zip([i * 2.4] * len(a), a, b): ax.plot([x0_, x0_ + 1], [y0_, y1_], color=colr, lw=0.7, alpha=0.6)
        ax.scatter([i * 2.4] * len(a), a, s=12, color=colr, lw=0); ax.scatter([i * 2.4 + 1] * len(b), b, s=12, color=colr, lw=0)
        ax.text(i * 2.4 + 0.5, rv.max().max() + 0.0012, f"−{100 * (1 - b.mean() / a.mean()):.0f}%\n{p_fmt(stats.wilcoxon(b, a).pvalue)}", ha="center", fontsize=9, color=MUTED)
    d = rv[("ours", "mecr_resolvi")] - rv[("tenx", "mecr_resolvi")]
    ax.set_ylim(rv.min().min() - 0.002, rv.max().max() + 0.0075); ax.set_xticks([0, 1, 2.4, 3.4], ["Ranger", "+ResolVI", "This\npipeline", "+ResolVI"], fontsize=10); ax.set_ylabel("MECR"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"H  ResolVI on top ({len(rv)} windows)\nafter ResolVI: ours vs Ranger {p_fmt(stats.wilcoxon(d).pvalue)}", loc="left", fontsize=12, fontweight="bold")
h = [plt.Rectangle((0, 0), 1, 1, color=c_, label=l.replace("\n", " ")) for _, l, c_ in M3]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
fig.savefig(sys.argv[1], facecolor="white", bbox_inches="tight"); fig.savefig(sys.argv[1].replace(".pdf", ".png"), facecolor="white", bbox_inches="tight", dpi=170)
print("saved")
