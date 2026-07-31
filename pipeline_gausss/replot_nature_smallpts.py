#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/replot_nature_smallpts.py
#
# The same re-plot with smaller marks, and the version to use. Point size drops from 2.0 to
# 0.55 and the motor-neuron ring from 12 to 3.5 with a thinner stroke.
#
# The first pass was too heavy. At s=2.0 the points merged into a solid mass and the tissue
# architecture stopped reading, while the motor-neuron rings dominated sections where there
# are only a few hundred of them among a hundred thousand cells. Smaller marks let the cord
# show through.
#
# Legends use fixed-size proxy handles so the swatches stay readable even though the plotted
# points are tiny, instead of being coupled to markerscale, which would have made the legend
# illegible along with the data.
#
# Writes to plots_publication_v2 so both versions survive for comparison.
# ========================================================================================

"""
replot_nature_smallpts.py
============================================================
Re-render every figure of the Novae MN-corrected niches run in Nature-tier
style, but with SMALLER spatial points and SMALLER MN overlay rings than
replot_nature.py (which used point s=2.0 and MN ring s=12, too heavy).

Key changes vs replot_nature.py:
  * POINT_SIZE 2.0 -> 0.55            (thinner, tissue architecture reads through)
  * MN ring: fixed s=3.5, lw=0.25     (was 12 / 0.35; markers no longer dominate)
  * Legend uses FIXED-size proxy handles so the swatches stay readable even
    though the plotted points are tiny (no markerscale coupling).
Everything else identical: despined axes, 500 um scale bar, Wong palette,
vector PDF + 300 dpi PNG, originals in plots/ and plots_publication/ untouched.
Writes to plots_publication_v2/.
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
OUT = os.path.join(BASE, "plots_publication_v2")
os.makedirs(OUT, exist_ok=True)

PRIMARY_KEY = "novae_domains_n6"
PRIMARY_LEVEL = 6
MN_KEY = "is_MN"
UNASSIGNED = "unassigned"

# --- the tuning knobs the user asked for ---
POINT_SIZE = 0.55        # domain cell points (was 2.0)
MN_RING_SIZE = 3.5       # MN overlay ring area (was POINT_SIZE*6 = 12)
MN_RING_LW = 0.25        # MN ring stroke width (was 0.35)

ns.apply_style()

DOMAINS_SORTED = ["D1011", "D1014", "D1015", "D1016", "D1017", "D991", "unassigned"]
DOM_COLOR = ns.domain_colors(DOMAINS_SORTED, unassigned=UNASSIGNED)


def _ok(name):
    print(f"[ok] {name}")


# ============================================================
# 1. Stacked-proportion plots (by sample / status / spinal level)
# ============================================================
def stacked_proportion(csv, title, base, note=None, rot=90):
    if not os.path.exists(csv):
        print(f"[skip] {base}: missing {csv}")
        return
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
enr_path = os.path.join(SUM, "MN_enrichment_by_domain.csv")
if os.path.exists(enr_path):
    enr = pd.read_csv(enr_path, index_col=0)
    enr.index = enr.index.astype(str)
    order = list(enr.index)
    colors = [DOM_COLOR.get(d, "0.5") for d in order]

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
else:
    print(f"[skip] MN enrichment bars: missing {enr_path}")


# ============================================================
# 3. Per-sample spatial niche maps (+ MN overlay), despined + scale bar
# ============================================================
try:
    import anndata as ad
    reader = ad.read_h5ad
except Exception:
    import scanpy as sc
    reader = sc.read_h5ad

files = sorted(glob.glob(os.path.join(PSAMP, "*__niches.h5ad")))
print(f"[per-sample] {len(files)} files")
for fp in files:
    sample = os.path.basename(fp).replace("__niches.h5ad", "")
    a = reader(fp)
    xy = np.asarray(a.obsm["spatial_orig"], dtype=float)
    dvals = a.obs[PRIMARY_KEY].astype(str).values
    mmn = a.obs[MN_KEY].values.astype(bool)

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    present = []
    for d in DOMAINS_SORTED:
        m = dvals == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[DOM_COLOR[d]],
                       linewidths=0, rasterized=True)
            present.append(d)
    n_mn = int(mmn.sum())
    if mmn.any():
        ax.scatter(xy[mmn, 0], xy[mmn, 1], s=MN_RING_SIZE, facecolors="none",
                   edgecolors="k", linewidths=MN_RING_LW, zorder=5, rasterized=True)
    ns.despine_spatial(ax)
    ax.set_title(f"{sample} — Novae niches (level {PRIMARY_LEVEL})")

    xr = xy[:, 0].max() - xy[:, 0].min()
    x0 = xy[:, 0].min() + 0.03 * xr
    y0 = xy[:, 1].max() - 0.03 * (xy[:, 1].max() - xy[:, 1].min())
    ns.add_scalebar(ax, (x0, y0), length_um=500)

    # fixed-size legend proxies (decoupled from the tiny plotted point size)
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                      markerfacecolor=DOM_COLOR[d], markeredgecolor='none', label=d)
               for d in present]
    if n_mn:
        handles.append(Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                              markerfacecolor='none', markeredgecolor='k',
                              markeredgewidth=0.5, label=f"MN (n={n_mn})"))
    ax.legend(handles=handles, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="Domain", handletextpad=0.4, borderaxespad=0.0, labelspacing=0.5)
    ns.save_both(fig, os.path.join(OUT, f"{sample}__niches_spatial"))
    _ok(f"{sample}__niches_spatial")
    del a

print("DONE. Wrote figures to", OUT)
