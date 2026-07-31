#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/replot_nature.py
#
# Regenerates every figure from the completed MN-corrected niche run in publication style,
# using the shared nature_style helpers. Originals in plots/ are left alone, output goes to
# plots_publication/.
#
# It exists because the analysis run and the figure standard evolved at different speeds. The
# compute was fine; the plots predated the style module. Re-plotting from the saved summaries
# and per-sample h5ads is far cheaper than re-running a GPU job to get nicer axes.
#
# Reads the summary CSVs for the bar, proportion and enrichment plots and the per-sample h5ads
# for the spatial maps with motor neurons overlaid.
# ========================================================================================

"""
replot_nature.py
============================================================
Regenerate ALL figures from the completed Novae MN-corrected niches run
(job 51996872) in Nature-publication style, using the crew's own
nature_style.py (Wong colourblind-safe palette, despined axes, 7pt Arial,
vector PDF + 300dpi PNG). Originals in plots/ are left untouched.

Reads:
  summary/*.csv                 -> bar / stacked-proportion / enrichment / count plots
  per_sample/{sample}__niches.h5ad -> per-sample spatial niche maps (+ MN overlay)
Writes:
  plots_publication/*.{pdf,png}
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nature_style as ns

BASE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches_MNcorrected"
SUM = os.path.join(BASE, "summary")
PSAMP = os.path.join(BASE, "per_sample")
OUT = os.path.join(BASE, "plots_publication")
os.makedirs(OUT, exist_ok=True)

PRIMARY_KEY = "novae_domains_n6"
PRIMARY_LEVEL = 6
MN_KEY = "is_MN"
UNASSIGNED = "unassigned"
POINT_SIZE = 2.0

ns.apply_style()

# ----- ONE canonical domain->colour map used across every figure -----
DOMAINS_SORTED = ["D1011", "D1014", "D1015", "D1016", "D1017", "D991", "unassigned"]
DOM_COLOR = ns.domain_colors(DOMAINS_SORTED, unassigned=UNASSIGNED)


def _ok(name):
    print(f"[ok] {name}")


# ============================================================
# 1. Stacked-proportion plots (by sample / status / spinal level)
# ============================================================
def stacked_proportion(csv, title, base, note=None, rot=90):
    ct = pd.read_csv(csv, index_col=0)
    ct.columns = [str(c) for c in ct.columns]
    prop = ct.div(ct.sum(axis=0), axis=1)
    groups = list(prop.columns)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    fig, ax = plt.subplots(figsize=(max(3.2, 0.28 * len(groups) + 1.6), 3.0))
    for d in prop.index.astype(str):
        vals = prop.loc[d].values
        ax.bar(x, vals, bottom=bottom, color=DOM_COLOR.get(d, "0.5"),
               label=d, width=0.82, linewidth=0)
        bottom += vals
    ax.set_ylabel("Proportion of cells")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rot, ha="center" if rot == 90 else "right")
    ax.margins(x=0.01)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="Domain", handlelength=1.0, handleheight=1.0)
    if note:
        ax.text(0.0, -0.42, note, transform=ax.transAxes, fontsize=5,
                color="firebrick", wrap=True, va="top")
    ns.save_both(fig, os.path.join(OUT, base))
    _ok(base)


stacked_proportion(os.path.join(SUM, "domain_by_sample_counts.csv"),
                   f"Novae domain composition by sample (level {PRIMARY_LEVEL})",
                   "domain_proportions_by_sample")
stacked_proportion(
    os.path.join(SUM, "domain_by_status_counts.csv"),
    f"Novae domain composition by disease status (level {PRIMARY_LEVEL})",
    "domain_proportions_by_status", rot=0,
    note=("CONFOUND-FLAGGED: spinal level confounded with disease (ALS = 8 lumbar + 2 "
          "cervical; controls = 10 cervical). Read against domain_proportions_by_spinal_"
          "level and the cervical-only ALS cases (SD03522, SD01320) before calling it biology."))
stacked_proportion(os.path.join(SUM, "domain_by_spinal_level_counts.csv"),
                   f"Novae domain composition by spinal level (level {PRIMARY_LEVEL})",
                   "domain_proportions_by_spinal_level", rot=0)


# ============================================================
# 2. MN enrichment (log2) + MN absolute count bars
# ============================================================
enr = pd.read_csv(os.path.join(SUM, "MN_enrichment_by_domain.csv"), index_col=0)
enr.index = enr.index.astype(str)          # already sorted by log2 enrichment (desc)
order = list(enr.index)
colors = [DOM_COLOR.get(d, "0.5") for d in order]

# --- 2a. log2 enrichment ---
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.bar(np.arange(len(order)), enr["log2_MN_enrichment"].values, color=colors,
       width=0.78, linewidth=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("log$_2$(MN fraction / all-cell fraction)")
ax.set_title(f"Motor-neuron enrichment per Novae niche (level {PRIMARY_LEVEL})")
ns.save_both(fig, os.path.join(OUT, "MN_enrichment_by_domain"))
_ok("MN_enrichment_by_domain")

# --- 2b. absolute MN count ---
nmn = enr["n_MN"].astype(int)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(np.arange(len(order)), nmn.values, color=colors, width=0.78, linewidth=0)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("Number of motor neurons (n)")
ax.set_title(f"Motor-neuron count per Novae niche (level {PRIMARY_LEVEL})")
ax.set_ylim(0, nmn.max() * 1.14)
for b, v in zip(bars, nmn.values):
    ax.text(b.get_x() + b.get_width() / 2, v + nmn.max() * 0.015, str(int(v)),
            ha="center", va="bottom", fontsize=6)
ns.save_both(fig, os.path.join(OUT, "MN_count_by_domain"))
_ok("MN_count_by_domain")


# ============================================================
# 3. Overall domain composition (replaces the native novae_domains_proportions)
# ============================================================
fa = enr["frac_of_all_cells"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(np.arange(len(fa)), fa.values,
              color=[DOM_COLOR.get(d, "0.5") for d in fa.index], width=0.78, linewidth=0)
ax.set_xticks(np.arange(len(fa)))
ax.set_xticklabels(list(fa.index), rotation=45, ha="right")
ax.set_ylabel("Proportion of all cells")
ax.set_title(f"Overall Novae domain composition (level {PRIMARY_LEVEL})")
for b, v in zip(bars, fa.values):
    ax.text(b.get_x() + b.get_width() / 2, v + fa.max() * 0.015, f"{v:.2f}",
            ha="center", va="bottom", fontsize=6)
ax.set_ylim(0, fa.max() * 1.14)
ns.save_both(fig, os.path.join(OUT, "novae_domains_proportions"))
_ok("novae_domains_proportions")


# ============================================================
# 4. Per-sample spatial niche maps (+ MN overlay), despined + scale bar
# ============================================================
try:
    import anndata as ad
    reader = ad.read_h5ad
except Exception:
    import scanpy as sc
    reader = sc.read_h5ad

# shared legend handles (domains present anywhere + MN ring)
files = sorted(glob.glob(os.path.join(PSAMP, "*__niches.h5ad")))
print(f"[per-sample] {len(files)} files")
for fp in files:
    sample = os.path.basename(fp).replace("__niches.h5ad", "")
    a = reader(fp)
    xy = np.asarray(a.obsm["spatial_orig"], dtype=float)
    dvals = a.obs[PRIMARY_KEY].astype(str).values
    mmn = a.obs[MN_KEY].values.astype(bool)

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for d in DOMAINS_SORTED:
        m = dvals == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[DOM_COLOR[d]],
                       label=d, linewidths=0, rasterized=True)
    if mmn.any():
        ax.scatter(xy[mmn, 0], xy[mmn, 1], s=POINT_SIZE * 6, facecolors="none",
                   edgecolors="k", linewidths=0.35, zorder=5, rasterized=True,
                   label=f"MN (n={int(mmn.sum())})")
    ns.despine_spatial(ax)          # hides spines/ticks, equal aspect, inverts y
    ax.set_title(f"{sample} — Novae niches (level {PRIMARY_LEVEL})")

    # 500 um scale bar near lower-left of the point cloud
    xr = xy[:, 0].max() - xy[:, 0].min()
    x0 = xy[:, 0].min() + 0.03 * xr
    y0 = xy[:, 1].max() - 0.03 * (xy[:, 1].max() - xy[:, 1].min())
    ns.add_scalebar(ax, (x0, y0), length_um=500)

    ax.legend(markerscale=4, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="Domain", handletextpad=0.4, borderaxespad=0.0)
    ns.save_both(fig, os.path.join(OUT, f"{sample}__niches_spatial"))
    _ok(f"{sample}__niches_spatial")
    del a

print("DONE. Wrote figures to", OUT)
