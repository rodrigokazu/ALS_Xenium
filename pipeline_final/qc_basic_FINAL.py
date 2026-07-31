#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/qc_basic_FINAL.py
#
# The ordinary QC plots. The report was somehow missing them. Transcript counts, cell area,
# genes per cell and negative-control fraction had only ever appeared as small violins buried
# in the hero page, and the request was to break them out properly, per niche and per sample.
#
# Top row is the whole section. Bottom row is the same four metrics split by Novae niche at
# novae_domains_n6, ordered by descending cell count, using the same domain colours as the
# rest of that sample's chapter. A per-domain median table sits at the foot.
#
# It reads the same per_sample niches h5ad the rest of the dual-pass QC reads, through the
# same loader. That is the only reason the numbers on this page can be trusted against the
# numbers on every other page.
# ========================================================================================

"""qc_basic_FINAL.py -- classic per-sample QC page (_FINAL re-run), added on top of the existing
Dual-pass QC report at the user's request: "classic transcript count, cell area, and all the basic
QC plots, per niche, per sample" were not broken out as their own page (only small violins buried in
the hero page existed before). Reads the same per_sample niches h5ad the rest of dpqc reads, so the
numbers are guaranteed consistent with the rest of the report.

Panels: TOP row = whole-section distributions (transcript counts, cell area, genes detected per
cell, negative-control fraction). BOTTOM = the same 4 metrics broken out per Novae niche
(novae_domains_n6), ordered by descending cell count with the same domain colours as the rest of
this sample's chapter (dpqc_style_FINAL.domain_palette). A per-domain median table sits at the foot.
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ps_io_FINAL_fix as ps_io
import dpqc_style_FINAL as st
import final_config as cfg

OBS_COLS = ["novae_domains_n6", "status", "sample", "total_counts", "transcript_counts",
            "cell_area", "nucleus_area", "control_probe_counts", "control_codeword_counts",
            "is_MN", "xenium_cell_id"]


def _order_by_size(dom):
    vc = dom.value_counts()
    order = [d for d in vc.index if d != "unassigned"] + (["unassigned"] if "unassigned" in vc.index else [])
    return order, vc


def _violin(ax, groups, values, colors, xlabel, logscale=False, ref=None):
    data = [np.log1p(v) if logscale else v for v in values]
    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_edgecolor("0.3"); pc.set_linewidth(0.4); pc.set_alpha(0.85)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_edgecolor("0.25"); parts[key].set_linewidth(0.6)
    ax.set_yticks(range(1, len(groups) + 1)); ax.set_yticklabels(groups, fontsize=6.5)
    ax.set_xlabel(("log1p " + xlabel) if logscale else xlabel, fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    if ref is not None:
        rv = np.log1p(ref) if logscale else ref
        ax.axvline(rv, color="0.4", linestyle="--", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _hist(ax, values, xlabel, logscale=False, color="#4C78A8", bins=40):
    v = np.log1p(values) if logscale else values
    ax.hist(v, bins=bins, color=color, edgecolor="white", linewidth=0.2)
    ax.set_xlabel(("log1p " + xlabel) if logscale else xlabel, fontsize=7.5)
    ax.set_ylabel("cells", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    med = np.median(values)
    ax.axvline(np.log1p(med) if logscale else med, color="0.2", linestyle="--", linewidth=0.7)
    ax.text(0.98, 0.95, f"median={med:,.0f}" if not logscale else f"median={med:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def make_basicqc(sample, h5, out_pdf):
    A = ps_io.load_indep(h5, obs_cols=OBS_COLS)
    obs = A.obs
    n = A.n_obs
    dom = obs["novae_domains_n6"].astype(str)
    status = str(obs["status"].iloc[0]) if "status" in obs else "NA"

    tx = np.asarray(obs["transcript_counts"], float)
    area = np.asarray(obs["cell_area"], float)
    negfrac = (np.asarray(obs["control_probe_counts"], float) +
               np.asarray(obs["control_codeword_counts"], float)) / np.maximum(tx, 1.0)
    counts = A.layers["counts"] if "counts" in A.layers else A.X
    genes_per_cell = np.asarray((counts > 0).sum(axis=1)).ravel()

    order, vc = _order_by_size(dom)
    palette = st.domain_palette(order)
    colors = [palette[d] for d in order]

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{sample}  |  {status}  |  n={n:,} cells  |  basic QC metrics "
                 f"(whole section + per Novae niche, n6)", fontsize=11, y=0.985)

    # ---- top row: whole-section distributions ----
    ax1 = fig.add_axes([0.07, 0.72, 0.20, 0.19])
    _hist(ax1, tx, "transcript_counts", logscale=True)
    ax1.set_title("Transcript counts / cell", fontsize=8)
    ax2 = fig.add_axes([0.32, 0.72, 0.20, 0.19])
    _hist(ax2, area, "cell_area (um^2)", logscale=False, color="#59A14F")
    ax2.set_title("Cell area", fontsize=8)
    ax3 = fig.add_axes([0.57, 0.72, 0.20, 0.19])
    _hist(ax3, genes_per_cell, "genes_detected", logscale=False, color="#E15759")
    ax3.set_title("Genes detected / cell", fontsize=8)
    ax4 = fig.add_axes([0.82, 0.72, 0.15, 0.19])
    _hist(ax4, negfrac * 100, "neg-ctrl % of transcripts", logscale=False, color="#B07AA1", bins=30)
    ax4.set_title("Negative-control %", fontsize=8)

    # ---- bottom: per-niche breakdown ----
    groups_tx = [tx[dom == d] for d in order]
    groups_area = [area[dom == d] for d in order]
    groups_genes = [genes_per_cell[dom == d] for d in order]

    axv1 = fig.add_axes([0.07, 0.40, 0.27, 0.27])
    _violin(axv1, order, groups_tx, colors, "transcript_counts", logscale=True, ref=float(np.median(tx)))
    axv1.set_title("Transcript counts per niche", fontsize=8)
    axv2 = fig.add_axes([0.40, 0.40, 0.27, 0.27])
    _violin(axv2, order, groups_area, colors, "cell_area (um^2)", logscale=False, ref=float(np.median(area)))
    axv2.set_title("Cell area per niche", fontsize=8)
    axv3 = fig.add_axes([0.73, 0.40, 0.24, 0.27])
    _violin(axv3, order, groups_genes, colors, "genes_detected", logscale=False, ref=float(np.median(genes_per_cell)))
    axv3.set_title("Genes/cell per niche", fontsize=8)

    # ---- per-domain median table ----
    ax_t = fig.add_axes([0.07, 0.06, 0.90, 0.28]); ax_t.axis("off")
    ax_t.text(0.0, 1.0, "Per-niche basic QC medians", fontsize=9, fontweight="bold", va="top")
    header = f"{'domain':<12}{'n cells':>9}{'%':>7}{'tx_counts':>11}{'cell_area':>11}{'genes/cell':>11}{'neg-ctrl%':>11}"
    ax_t.text(0.0, 0.93, header, fontsize=7.3, family="monospace", va="top", fontweight="bold")
    y = 0.87
    for d in order:
        m = dom == d
        nc = int(m.sum())
        line = (f"{d:<12}{nc:>9,}{100*nc/n:>6.1f}%{np.median(tx[m]):>11,.0f}"
                f"{np.median(area[m]):>11,.0f}{np.median(genes_per_cell[m]):>11,.0f}"
                f"{100*np.mean(negfrac[m]):>10.2f}%")
        ax_t.text(0.0, y, line, fontsize=7.0, family="monospace", va="top")
        y -= 0.075
    ax_t.text(0.0, max(y - 0.02, 0.0),
              "Added basic-QC page: whole-section + per-niche transcript counts, cell area, genes "
              "detected, and negative-control fraction. Dashed line = whole-section median. Domain "
              "colours match this sample's other figures (assigned by descending cell count).",
              fontsize=6.3, color="0.35", va="top", wrap=True)

    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    make_basicqc(a.sample, a.h5, a.out)
    print(f"[{a.sample}] basic-QC page -> {a.out}")
