#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_figures_FINAL.py
#
# Turns one sample's scores into four figures. No decisions are made here; it consumes what
# dpqc_compute_FINAL produced and the shared colour registries from dpqc_style_FINAL.
#
# The four: a master hero page (domain atlas, composition, count and area violins, smear score
# ranking, decision ledger) that should let you call keep-or-remove at a glance; a highlight
# page of per-domain spatial small multiples, the actual smear detector, because an artefact
# stops looking like anatomy the moment it is isolated; a marker dotplot; and a QC panel page
# covering morphology, negative controls and spatial coherence.
#
# Maps use obsm['spatial_orig'] so distances are real microns, are despined, and carry a
# scalebar. Scatter is rasterised inside an otherwise vector PDF. Without that a 120,000 cell
# section produces a file nobody can open.
#
# Domain colours are derived per sample by descending cell count. They are not a fixed global
# palette, because with independent per-sample models domain 3 in one section has nothing to
# do with domain 3 in another and a shared palette would imply it did.
# ========================================================================================

"""dpqc_figures_FINAL.py -- Nature-tier despined per-sample figures for the Dual-pass QC (_FINAL
re-run). Consumes (A, df, meta, rg) from dpqc_compute_FINAL and the shared colour registries in
dpqc_style_FINAL. Figure layout/logic identical to the proven _gausss dpqc_figures.

Produces per sample:
  <sample>__master.{png,pdf}        the hero page: domain atlas + composition + count/area violins +
                                    smear-score ranking + decision ledger, one glance -> keep/remove
  <sample>__highlight.{png,pdf}     per-domain spatial small-multiples (THE smear detector), verdict-framed
  <sample>__markers.{png,pdf}       per-domain marker dotplot (fraction expressing x mean z-expression)
  <sample>__qcpanels.{png,pdf}      morphology (cell/nucleus area), negative-control, spatial-coherence bars

All maps use obsm['spatial_orig'] (true microns), despined, with a micron scalebar; scatter is
rasterized so the vector PDF stays small. Colours: domains per-sample (descending size), cell types
global-fixed, verdict via FLAG_COLORS. apply_style() sets Nature rcParams (Arial 7pt, fonttype 42).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dpqc_style_FINAL as st
import nature_style as ns

RES = "novae_domains_n6"


# ------------------------------------------------------------------ helpers
def domain_order(df):
    """Domains by DESCENDING cell count; 'unassigned' forced last."""
    d = df.copy()
    d["_u"] = (d["domain"].astype(str) == "unassigned").astype(int)
    d = d.sort_values(["_u", "n_cells"], ascending=[True, False])
    return list(d["domain"].astype(str))


def dom_colors(order):
    return st.domain_palette(order)


def _size(n):
    return float(np.clip(60000.0 / max(n, 1), 0.4, 3.0))


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _title(meta, extra=""):
    v = meta.get("verdicts", {})
    return (f"{meta['sample']}  |  {st.status_label(meta['status'])}  |  {meta['level']}  |  "
            f"{meta['seg_version']}  |  n={meta['n_cells']:,} cells" + (f"  |  {extra}" if extra else ""))


# ------------------------------------------------------------------ panels
def panel_atlas(ax, A, order, cmap, meta):
    xy = np.asarray(A.obsm["spatial_orig"])
    dom = A.obs[RES].astype(str).values
    s = _size(A.n_obs)
    draw = (["unassigned"] if "unassigned" in order else []) + [d for d in order if d != "unassigned"]
    for d in draw:
        m = dom == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=cmap[d], linewidths=0, rasterized=True, alpha=0.75)
    ns.despine_spatial(ax)
    x0 = xy[:, 0].min() + 0.03 * np.ptp(xy[:, 0]); y0 = xy[:, 1].max() - 0.03 * np.ptp(xy[:, 1])
    ns.add_scalebar(ax, (x0, y0), 500, label="500 um", color="black", fontsize=6)
    ax.set_title("Domain atlas (novae_domains_n6)", fontsize=9)
    h = [Line2D([0], [0], marker='o', ls='none', ms=5, mfc=cmap[d], mec='none',
                label=f"{d} ({int(df_lookup(meta, d, 'pct_cells'))}%)" if False else d) for d in order]
    ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
              fontsize=6, title="domain (per-sample)", title_fontsize=6, handletextpad=0.3)


def df_lookup(meta, d, col):
    for r in meta.get("domains_detail", []):
        if str(r["domain"]) == str(d):
            return r.get(col, 0)
    return 0


def panel_composition(ax, df, order):
    rows = [r for r in order]
    y = np.arange(len(rows))[::-1]
    for yi, d in zip(y, rows):
        rec = df[df.domain.astype(str) == d]
        if not len(rec):
            continue
        fr = rec.iloc[0]["celltype_fracs"]
        if isinstance(fr, str):
            import json; fr = json.loads(fr.replace("'", '"'))
        left = 0.0
        for ct in st.CELLTYPE_ORDER:
            v = float(fr.get(ct, 0.0))
            if v <= 0:
                continue
            ax.barh(yi, v, left=left, height=0.72, color=st.CELLTYPE_COLORS[ct], edgecolor="none")
            left += v
        n = int(rec.iloc[0]["n_cells"]); unt = rec.iloc[0]["pct_untyped"]
        ax.text(1.01, yi, f"n={n:,}  unt {unt:.0f}%", va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel("cell-type fraction (Untyped shown)")
    ax.set_title("Cell-type composition per domain", fontsize=9)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def _violin(ax, A, df, order, col, meta, logscale=False, ref=None, xlabel=""):
    dom = A.obs[RES].astype(str).values
    vals = np.asarray(A.obs[col], float)
    if logscale:
        vals = np.log1p(vals)
        if ref is not None:
            ref = np.log1p(ref)
    y = np.arange(len(order))[::-1]
    data, pos, colors = [], [], []
    for yi, d in zip(y, order):
        m = dom == d
        if m.sum() >= 5:
            data.append(vals[m]); pos.append(yi)
            rec = df[df.domain.astype(str) == d]
            colors.append(st.flag_color(rec.iloc[0]["verdict"]) if len(rec) else "0.6")
    if data:
        vp = ax.violinplot(data, positions=pos, vert=False, widths=0.8, showmedians=True,
                           showextrema=False)
        for b, c in zip(vp["bodies"], colors):
            b.set_facecolor(c); b.set_edgecolor("0.3"); b.set_alpha(0.85); b.set_linewidth(0.4)
        if "cmedians" in vp:
            vp["cmedians"].set_color("0.15"); vp["cmedians"].set_linewidth(0.8)
    if ref is not None:
        ax.axvline(ref, ls="--", lw=0.8, color="0.4")
        ax.text(ref, len(order) - 0.4, " sample median", fontsize=5.5, color="0.4", va="top")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel(xlabel); ax.set_title(xlabel, fontsize=9)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def panel_smearscore(ax, df):
    d = df.copy()
    d["_u"] = (d.domain.astype(str) == "unassigned").astype(int)
    d = d.sort_values(["smear_score"], ascending=True)          # worst (highest) on top after barh
    y = np.arange(len(d))
    cols = [st.flag_color(v) for v in d["verdict"]]
    ax.barh(y, d["smear_score"], color=cols, edgecolor="0.3", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(d["domain"].astype(str), fontsize=7)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["smear_score"] + 0.01, yi, r["verdict"], va="center", fontsize=5.5,
                color=st.flag_color(r["verdict"]))
    ax.axvline(0.62, ls="--", lw=0.7, color="0.3"); ax.axvline(0.42, ls=":", lw=0.7, color="0.5")
    ax.set_xlim(0, 1.0); ax.set_xlabel("composite smear score")
    ax.set_title("Smear score + verdict (worst at top)", fontsize=9)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def panel_ledger(ax, df, order, cmap):
    ax.axis("off")
    ax.text(0.0, 1.0, "Decision ledger", fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    yy = 0.90
    ax.text(0.0, yy, f"{'domain':10s}{'verdict':9s}{'n':>9s}{'%':>6s}  reason",
            fontsize=6.5, family="monospace", va="top", transform=ax.transAxes)
    yy -= 0.06
    for d in order:
        rec = df[df.domain.astype(str) == d]
        if not len(rec):
            continue
        r = rec.iloc[0]
        ax.add_patch(Rectangle((0.0, yy - 0.028), 0.018, 0.03, facecolor=cmap.get(d, "0.6"),
                               edgecolor="none", transform=ax.transAxes))
        ax.add_patch(Rectangle((0.10, yy - 0.028), 0.018, 0.03, facecolor=st.flag_color(r["verdict"]),
                               edgecolor="none", transform=ax.transAxes))
        txt = f"{d:9s} {r['verdict']:8s} {int(r['n_cells']):>8,} {r['pct_cells']:>5.1f}  {str(r['reason'])[:74]}"
        ax.text(0.03, yy - 0.012, txt, fontsize=6, family="monospace", va="center", transform=ax.transAxes)
        yy -= 0.05
        if yy < 0.03:
            break


# ------------------------------------------------------------------ figures
def make_master(A, df, meta, out):
    ns.apply_style()
    order = domain_order(df); cmap = dom_colors(order)
    fig = plt.figure(figsize=(15, 15)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.25, 0.9],
                          hspace=0.32, wspace=0.30)
    ax_atlas = fig.add_subplot(gs[0:2, 0:2])
    ax_comp = fig.add_subplot(gs[0:2, 2])
    ax_cnt = fig.add_subplot(gs[2, 0])
    ax_area = fig.add_subplot(gs[2, 1])
    ax_score = fig.add_subplot(gs[2, 2])
    ax_led = fig.add_subplot(gs[3, :])
    panel_atlas(ax_atlas, A, order, cmap, meta)
    panel_composition(ax_comp, df, order)
    _violin(ax_cnt, A, df, order, "transcript_counts", meta, logscale=True,
            ref=meta.get("section_tx_median"), xlabel="log1p transcript_counts")
    _violin(ax_area, A, df, order, "cell_area", meta, logscale=False,
            ref=float(np.median(np.asarray(A.obs["cell_area"], float))), xlabel="cell_area (um^2)")
    panel_smearscore(ax_score, df)
    panel_ledger(ax_led, df, order, cmap)
    # shared cell-type legend under composition
    fig.legend(handles=st.celltype_handles(present=_present_types(df)),
               loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=7, frameon=False, fontsize=6.5)
    nrem = meta.get("n_remove", 0); nrev = meta.get("n_review", 0); nkeep = meta.get("n_keep", 0)
    fig.suptitle(_title(meta, f"KEEP {nkeep} / REVIEW {nrev} / REMOVE {nrem}  "
                              f"({meta.get('pct_cells_remove', 0)}% cells flagged remove)"),
                 fontsize=12, y=0.995)
    ns.save_both(fig, out)


def _present_types(df):
    present = set()
    for _, r in df.iterrows():
        fr = r["celltype_fracs"]
        if isinstance(fr, str):
            import json; fr = json.loads(fr.replace("'", '"'))
        present |= {k for k, v in fr.items() if v > 0}
    return present


def make_highlight(A, df, meta, out):
    """Per-domain spatial small-multiples: focal domain in colour over grey; frame = verdict."""
    ns.apply_style()
    order = domain_order(df); cmap = dom_colors(order)
    xy = np.asarray(A.obsm["spatial_orig"]); dom = A.obs[RES].astype(str).values
    nd = len(order); ncol = min(4, nd); nrow = int(np.ceil(nd / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    s_bg = _size(A.n_obs) * 0.7; s_fg = _size(A.n_obs) * 1.6
    for i, d in enumerate(order):
        ax = axes[i]; m = dom == d
        ax.scatter(xy[~m, 0], xy[~m, 1], s=s_bg, c="0.86", linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], s=s_fg, c=cmap[d], linewidths=0, rasterized=True, alpha=0.85)
        ns.despine_spatial(ax)
        rec = df[df.domain.astype(str) == d].iloc[0]
        fc = st.flag_color(rec["verdict"])
        for spn in ax.spines.values():
            spn.set_visible(True); spn.set_color(fc); spn.set_linewidth(2.2)
        ax.set_title(f"{d}  {rec['verdict']}\nscore {rec['smear_score']:.2f} | "
                     f"n={int(rec['n_cells']):,} ({rec['pct_cells']:.1f}%)", fontsize=7.5, color=fc)
        if i == 0:
            x0 = xy[:, 0].min() + 0.03 * np.ptp(xy[:, 0]); y0 = xy[:, 1].max() - 0.03 * np.ptp(xy[:, 1])
            ns.add_scalebar(ax, (x0, y0), 500, label="500 um", color="black", fontsize=6)
    for j in range(nd, len(axes)):
        axes[j].axis("off")
    fig.legend(handles=st.flag_handles(), loc="upper right", ncol=3, frameon=False, fontsize=7)
    fig.suptitle(_title(meta, "per-domain highlight (smear detector)"), fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ns.save_both(fig, out)


def make_markers(A, df, meta, rg, out, top_n=5):
    ns.apply_style()
    order = [d for d in domain_order(df) if d != "unassigned"]
    if rg is None or len(order) < 2:
        fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off")
        ax.text(0.5, 0.5, "marker dotplot unavailable (<2 domains)", ha="center")
        ns.save_both(fig, out); return
    # top genes per domain (by score), union preserving per-domain block order
    top = []
    for d in order:
        gg = rg[rg.group.astype(str) == d].sort_values("scores", ascending=False)
        for g in gg["names"].head(top_n):
            if g not in top:
                top.append(g)
    dom = A.obs[RES].astype(str).values
    X = _dense(A[:, top].X)
    C = _dense(A[:, top].layers["counts"]) if "counts" in A.layers else (X > 0)
    mean = np.zeros((len(order), len(top))); frac = np.zeros_like(mean)
    for i, d in enumerate(order):
        m = dom == d
        mean[i] = X[m].mean(0); frac[i] = (C[m] > 0).mean(0)
    z = (mean - mean.mean(0)) / (mean.std(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(7, len(top) * 0.32), max(3.2, len(order) * 0.5) + 1.2))
    xs, ys, ss, cs = [], [], [], []
    for i in range(len(order)):
        for j in range(len(top)):
            xs.append(j); ys.append(i); ss.append(8 + 220 * frac[i, j]); cs.append(z[i, j])
    scn = ax.scatter(xs, ys, s=ss, c=cs, cmap="RdBu_r", vmin=-2.5, vmax=2.5, edgecolors="0.3", linewidths=0.3)
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for tick, d in zip(ax.get_yticklabels(), order):
        rec = df[df.domain.astype(str) == d].iloc[0]
        tick.set_color(st.flag_color(rec["verdict"]))
    ax.set_ylim(-0.6, len(order) - 0.4); ax.invert_yaxis()
    cb = fig.colorbar(scn, ax=ax, fraction=0.025, pad=0.01); cb.set_label("mean expr z", fontsize=7)
    h = [plt.scatter([], [], s=8 + 220 * f, c="0.5", edgecolors="0.3", linewidths=0.3, label=f"{int(f*100)}%")
         for f in (0.25, 0.5, 1.0)]
    ax.legend(handles=h, title="frac expr", loc="center left", bbox_to_anchor=(1.05, 0.5),
              frameon=False, fontsize=6)
    ax.set_title(_title(meta, "top markers per domain (y-labels coloured by verdict)"), fontsize=9)
    ns.save_both(fig, out)


def make_atlas(A, df, meta, out):
    """Compact standalone domain atlas thumbnail for the cohort contact sheet."""
    ns.apply_style()
    order = domain_order(df); cmap = dom_colors(order)
    xy = np.asarray(A.obsm["spatial_orig"]); dom = A.obs[RES].astype(str).values
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    s = _size(A.n_obs)
    draw = (["unassigned"] if "unassigned" in order else []) + [d for d in order if d != "unassigned"]
    for d in draw:
        m = dom == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=cmap[d], linewidths=0, rasterized=True, alpha=0.8)
    ns.despine_spatial(ax)
    nrem = meta.get("n_remove", 0)
    ax.set_title(f"{meta['sample']} | {st.status_label(meta['status'])}\n{meta['level']} | "
                 f"{meta['seg_version']} | REMOVE {nrem}", fontsize=7,
                 color=st.status_color(meta["status"]))
    ns.save_both(fig, out)


def make_qcpanels(A, df, meta, out):
    """Morphology (nucleus-free %), negative-control ratio, and spatial-coherence bars per domain."""
    ns.apply_style()
    order = domain_order(df)
    d = df.set_index(df.domain.astype(str)).reindex(order)
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, 0.55 * len(order) + 1.5)))
    cols = [st.flag_color(v) for v in d["verdict"]]
    # (1) morphology: nucleus-free % + median nuclear frac
    ax = axes[0]
    ax.barh(y, d["pct_no_nucleus"], color=cols, edgecolor="0.3", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("% cells with no nucleus"); ax.set_title("Nucleus-free fraction (spillover tell)", fontsize=9)
    # (2) negative-control ratio (log2)
    ax = axes[1]
    nr = np.log2(np.clip(d["negctrl_ratio"].values.astype(float), 1e-3, None))
    ax.barh(y, nr, color=cols, edgecolor="0.3", linewidth=0.3)
    ax.axvline(0, ls="--", lw=0.7, color="0.4"); ax.axvline(np.log2(3), ls=":", lw=0.7, color="0.5")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("log2 neg-ctrl ratio vs section"); ax.set_title("Negative-control burden", fontsize=9)
    # (3) coherence triad: purity, largest-CC, Moran (grouped)
    ax = axes[2]
    w = 0.26
    ax.barh(y + w, d["nbhd_purity"], height=w, color="#0072B2", label="nbhd purity")
    ax.barh(y, d["largest_cc_frac"].astype(float), height=w, color="#009E73", label="largest-CC frac")
    ax.barh(y - w, d["morans_I"].astype(float).clip(lower=0), height=w, color="#CC79A7", label="Moran I")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7); ax.set_xlim(0, 1)
    ax.set_xlabel("coherence (higher = territory)"); ax.set_title("Spatial coherence triad", fontsize=9)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    for a in axes:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    fig.suptitle(_title(meta, "morphology / negative-control / spatial coherence"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ns.save_both(fig, out)
