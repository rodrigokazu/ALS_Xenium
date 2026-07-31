#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_sweep.py
#
# Answers a question the main per-sample report does not: how much does the picture change
# with the number of niches you ask for?
#
# The independent run scanned Novae hierarchy levels per sample and kept the level nearest
# each target of 4, 6 and 10 domains. This report shows one sample across all three side by
# side, then quantifies the nesting between them with contingency tables, adjusted Rand index
# and mean purity, then checks whether the markers change.
#
# The reason to look is that a resolution choice can invent structure. If n4 to n6 to n10 nest
# cleanly then the finer solutions are subdividing real compartments; if they reshuffle, the
# domain count is doing the work and not the tissue.
#
# Four pages: facts and composition, spatial maps at all three resolutions, the nesting
# analysis, and marker dotplots at each.
# ========================================================================================

"""ps_sweep.py -- per-sample niche-NUMBER sweep report for the INDEPENDENT Novae run.

The INDEPENDENT run scanned Novae hierarchy levels per sample and kept the level nearest
each target domain count [4, 6, 10] -> obs columns novae_domains_n4 / n6 / n10. This report
shows, for ONE sample, how the spatial niches change with the target count:
  page 1  facts: cells/MN, realized domain counts, chosen Novae level per target, composition
  page 2  spatial niche maps at n4 | n6 | n10 (side by side)
  page 3  nesting: contingency n4->n6 and n6->n10 (row-normalised), ARI + mean purity
  page 4  markers: top-5 dotplot at each resolution (are coarse markers refined, not replaced?)
Also writes a per-sample sweep stats JSON for the cohort roll-up (ps_sweep_cohort.py).

Runs in module python 3.11.1 (scanpy/anndata/sklearn/matplotlib). Reads the h5ad via ps_io
(h5py build; avoids the /uns/log1p read crash). Spatial maps use spatial_orig -- no DAPI/bundle
needed, so this is light. DATA IS PRE-QC -> caveat stamped.
"""
import os, sys, argparse, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
VH = os.path.dirname(HERE)
for p in (HERE, VH):
    if p not in sys.path:
        sys.path.insert(0, p)
import nature_style as ns
import ps_io

RESES = ["novae_domains_n4", "novae_domains_n6", "novae_domains_n10"]
TAGS = {"novae_domains_n4": "n4", "novae_domains_n6": "n6", "novae_domains_n10": "n10"}
PREQC = ("PRE-QC / exploratory. Marcel Gaussian re-segmentation; INDEPENDENT per-sample Novae with a "
         "per-sample level scan targeting 4/6/10 domains (min_counts>=1 only). Unsupervised spatial "
         "niches (not annotated grey/white matter); is_MN unvalidated by CHAT/MNX1; N=1 section. "
         "Domain IDs are per-sample and per-resolution -- not comparable across samples.")


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def palette(labels):
    doms = sorted([str(x) for x in set(labels) if str(x) != "unassigned"])
    order = doms + (["unassigned"] if "unassigned" in set(map(str, labels)) else [])
    return order, ns.domain_colors(order)


def chosen_levels(h5, sample):
    """Per-target Novae level chosen for this sample + the level->count scan, read from the
    original run's summary manifest (<rundir>/summary/<sample>__manifest.json). Returns
    ({target: level}, {level: n_domains}). uns['novae_attrs'] holds hop params, NOT levels."""
    rundir = os.path.dirname(os.path.dirname(h5))           # .../Novae_persample_INDEPENDENT_MNcorrected
    mp = os.path.join(rundir, "summary", f"{sample}__manifest.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            return (m.get("chosen_level_per_target", {}) or {},
                    m.get("level_scan_counts", {}) or {})
        except Exception:
            pass
    return {}, {}


# ----------------------------------------------------------------------------- pages
def fig_spatial_triplet(A, sample, status, out_png):
    ns.apply_style()
    xy = np.asarray(A.obsm["spatial_orig"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, res in zip(axes, RESES):
        order, cmap = palette(A.obs[res].values)
        dom = A.obs[res].astype(str).values
        draw = (["unassigned"] if "unassigned" in order else []) + [d for d in order if d != "unassigned"]
        for d in draw:
            m = dom == d
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=cmap[d], linewidths=0, rasterized=True)
        ns.despine_spatial(ax)
        n = len([d for d in order if d != "unassigned"])
        ax.set_title(f"{TAGS[res]}  ({n} niches)", fontsize=11)
        h = [Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                    markerfacecolor=cmap[d], markeredgecolor='none', label=d) for d in order]
        ax.legend(handles=h, loc='upper right', fontsize=5.5, frameon=False, handletextpad=0.3,
                  labelspacing=0.25, ncol=1)
    fig.suptitle(f"{sample}  |  {status}  |  niche-number sweep  (n={A.n_obs:,} cells)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _contingency(A, coarse, fine):
    """Row-normalised contingency (rows=fine domains, cols=coarse), + mean purity + ARI.
    Excludes 'unassigned' from both."""
    c = A.obs[coarse].astype(str).values
    fdom = A.obs[fine].astype(str).values
    keep = (c != "unassigned") & (fdom != "unassigned")
    c, fdom = c[keep], fdom[keep]
    ct = pd.crosstab(pd.Series(fdom, name=fine), pd.Series(c, name=coarse))
    rn = ct.div(ct.sum(1), axis=0)                       # each fine domain -> distribution over coarse
    purity = float(rn.max(1).mean())                     # mean top fraction (1.0 = perfect nesting)
    ari = float(adjusted_rand_score(c, fdom))
    return rn, purity, ari


def fig_nesting(A, sample, out_png):
    ns.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    meta = {}
    for ax, (coarse, fine) in zip(axes, [("novae_domains_n4", "novae_domains_n6"),
                                         ("novae_domains_n6", "novae_domains_n10")]):
        rn, purity, ari = _contingency(A, coarse, fine)
        im = ax.imshow(rn.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(rn.shape[1])); ax.set_xticklabels(rn.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(rn.shape[0])); ax.set_yticklabels(rn.index, fontsize=7)
        ax.set_xlabel(f"coarser: {TAGS[coarse]}"); ax.set_ylabel(f"finer: {TAGS[fine]}")
        ax.set_title(f"{TAGS[fine]} into {TAGS[coarse]}\npurity={purity:.2f}  ARI={ari:.2f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        meta[f"{TAGS[coarse]}_{TAGS[fine]}"] = {"purity": round(purity, 3), "ari": round(ari, 3)}
    fig.suptitle(f"{sample}  |  do finer niches nest inside coarser ones?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return meta


def rank_res(A, res, top_n=5):
    keep = A.obs[res].astype(str) != "unassigned"
    sub = A[keep.values].copy()
    sub.obs[res] = pd.Categorical(sub.obs[res].astype(str),
                                  categories=sorted(sub.obs[res].astype(str).unique()))
    groups = list(sub.obs[res].cat.categories)
    if len(groups) < 2:
        return sub, [], groups, {}
    sc.tl.rank_genes_groups(sub, res, method="wilcoxon", n_genes=sub.n_vars, use_raw=False)
    df = sc.get.rank_genes_groups_df(sub, group=None)
    top = []
    per = {}
    for g in groups:
        gg = df[df.group == g].sort_values("scores", ascending=False)
        names = list(gg["names"].head(top_n))
        per[g] = names
        for nme in names:
            if nme not in top:
                top.append(nme)
    return sub, top, groups, per


def _mean_frac(sub, res, top, groups):
    Xtop = _dense(sub[:, top].X)
    Ctop = _dense(sub[:, top].layers["counts"]) if "counts" in sub.layers else (Xtop > 0)
    lab = sub.obs[res].astype(str).values
    mean = pd.DataFrame(index=groups, columns=top, dtype=float)
    frac = pd.DataFrame(index=groups, columns=top, dtype=float)
    for g in groups:
        m = lab == g
        mean.loc[g] = Xtop[m].mean(0); frac.loc[g] = (Ctop[m] > 0).mean(0)
    return mean, frac


def fig_marker_triplet(A, sample, out_png):
    ns.apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    topsets = {}
    for ax, res in zip(axes, RESES):
        sub, top, groups, per = rank_res(A, res, top_n=5)
        topsets[TAGS[res]] = top
        if not top:
            ax.text(0.5, 0.5, f"{TAGS[res]}: <2 niches", ha="center"); ax.axis("off"); continue
        order, cmap = palette(A.obs[res].values)
        mean, frac = _mean_frac(sub, res, top, groups)
        ng, ngr = len(top), len(groups)
        xs, ys, ss, cs = [], [], [], []
        for gi, gr in enumerate(groups):
            for gj, gn in enumerate(top):
                xs.append(gj); ys.append(gi)
                ss.append(6 + 150 * float(frac.loc[gr, gn])); cs.append(float(mean.loc[gr, gn]))
        scv = ax.scatter(xs, ys, s=ss, c=cs, cmap="viridis", edgecolors="0.3", linewidths=0.25)
        ax.set_xticks(range(ng)); ax.set_xticklabels(top, rotation=90, fontsize=5.5)
        ax.set_yticks(range(ngr)); ax.set_yticklabels(groups, fontsize=7)
        for tick, gr in zip(ax.get_yticklabels(), groups):
            tick.set_color(cmap.get(gr, "black"))
        ax.set_xlim(-0.6, ng - 0.4); ax.set_ylim(-0.6, ngr - 0.4); ax.invert_yaxis()
        ax.set_title(f"{TAGS[res]} top markers", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.colorbar(scv, ax=ax, fraction=0.015, pad=0.01)
    fig.suptitle(f"{sample}  |  markers across resolutions (dot size=frac expr, colour=mean log1p)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    # marker-set turnover (Jaccard of pooled top-5 sets between consecutive resolutions)
    def jac(a, b):
        a, b = set(a), set(b)
        return round(len(a & b) / max(len(a | b), 1), 3)
    turn = {"n4_n6": jac(topsets.get("n4", []), topsets.get("n6", [])),
            "n6_n10": jac(topsets.get("n6", []), topsets.get("n10", []))}
    return topsets, turn


def _text_page(pdf, lines, title=None, fontsize=10, family="monospace"):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.9]); ax.axis("off")
    y = 0.98
    if title:
        ax.text(0.0, y, title, fontsize=15, fontweight="bold", va="top", family="sans-serif"); y -= 0.045
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=fontsize, va="top", family=family); y -= 0.023
    pdf.savefig(fig); plt.close(fig)


def _img_page(pdf, png, caption=None):
    if not os.path.exists(png):
        _text_page(pdf, [f"[missing: {os.path.basename(png)}]"]); return
    img = plt.imread(png)
    fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white")   # landscape for triplets
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.86]); ax.axis("off"); ax.imshow(img)
    if caption:
        fig.text(0.5, 0.04, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    pdf.savefig(fig, dpi=200); plt.close(fig)


def run(h5, outroot, sample=None):
    import textwrap
    A = ps_io.load_indep(h5)
    sample = sample or str(A.obs["sample"].iloc[0])
    status = str(A.obs["status"].iloc[0])
    n_mn = int(pd.Series(A.obs["is_MN"]).map(lambda v: str(v).lower() in ("true", "1", "1.0")).sum())
    figdir = os.path.join(outroot, "figures", sample); pdfdir = os.path.join(outroot, "pdf")
    for d in (figdir, pdfdir, os.path.join(outroot, "stats")):
        os.makedirs(d, exist_ok=True)
    lv, scan = chosen_levels(h5, sample)
    counts = {TAGS[r]: len([d for d in A.obs[r].astype(str).unique() if d != "unassigned"]) for r in RESES}
    print(f"[{sample}] sweep counts {counts} chosen_levels {lv}", flush=True)

    f_sp = os.path.join(figdir, f"{sample}__sweep_spatial.png")
    f_ne = os.path.join(figdir, f"{sample}__sweep_nesting.png")
    f_mk = os.path.join(figdir, f"{sample}__sweep_markers.png")
    fig_spatial_triplet(A, sample, status, f_sp)
    nest = fig_nesting(A, sample, f_ne)
    topsets, turn = fig_marker_triplet(A, sample, f_mk)
    print(f"[{sample}] nesting {nest} turnover {turn}", flush=True)

    out_pdf = os.path.join(pdfdir, f"{sample}__sweep.pdf")
    with PdfPages(out_pdf) as pdf:
        head = [f"Sample: {sample}    Status: {status}",
                f"Cells: {A.n_obs:,}    Motor neurons (is_MN): {n_mn:,}", "",
                "Target niche counts scanned per sample: 4, 6, 10", "",
                "Realised domains:"]
        for r in RESES:
            head.append(f"    {TAGS[r]:4s} -> {counts[TAGS[r]]} niches")
        if lv:
            head += ["", "Chosen Novae hierarchy level per target count (per-sample scan):"]
            for tgt in ("4", "6", "10"):
                L = lv.get(tgt, lv.get(int(tgt)) if str(tgt).isdigit() else None)
                if L is not None:
                    head.append(f"    target {tgt} niches -> Novae level {L}")
        head += ["", "Nesting (do finer niches subdivide coarser ones?):"]
        for k, v in nest.items():
            head.append(f"    {k}: purity={v['purity']}  ARI={v['ari']}")
        head += ["", "Marker-set turnover (Jaccard of pooled top-5 markers):",
                 f"    n4->n6: {turn['n4_n6']}    n6->n10: {turn['n6_n10']}",
                 "", "Contents:  2 spatial maps (n4|n6|n10)   3 nesting   4 markers by resolution",
                 "", "-" * 92, "PROVENANCE / CAVEAT:"]
        for ln in textwrap.wrap(PREQC, 90):
            head.append("  " + ln)
        _text_page(pdf, head, title=f"Niche-number sweep  |  {sample}")
        _img_page(pdf, f_sp, f"Fig 1 | {sample}: spatial niches at target counts 4, 6, 10 (independent per-sample Novae).")
        _img_page(pdf, f_ne, f"Fig 2 | {sample}: nesting of finer niches within coarser (row-normalised contingency).")
        _img_page(pdf, f_mk, f"Fig 3 | {sample}: top-5 markers per niche at each resolution.")
    print(f"[{sample}] wrote sweep PDF -> {out_pdf}", flush=True)

    stats = {"sample": sample, "status": status, "n_cells": int(A.n_obs), "n_mn": n_mn,
             "realised_counts": counts,
             "chosen_levels": {str(k): (int(v) if str(v).lstrip('-').isdigit() else v) for k, v in lv.items()},
             "level_scan_counts": {str(k): int(v) for k, v in scan.items()} if scan else {},
             "nesting": nest, "marker_turnover": turn,
             "top_markers": {TAGS[r]: topsets.get(TAGS[r], []) for r in RESES},
             "composition": {TAGS[r]: {d: int(c) for d, c in A.obs[r].astype(str).value_counts().items()}
                             for r in RESES}}
    with open(os.path.join(outroot, "stats", f"{sample}__sweep_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[{sample}] wrote sweep stats JSON", flush=True)
    return out_pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=None); ap.add_argument("--sample", default=None)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--persample-dir",
                    default="/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
                            "Novae_persample_INDEPENDENT_MNcorrected/per_sample")
    ap.add_argument("--outroot",
                    default="/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
                            "Novae_persample_INDEPENDENT_MNcorrected/report/resolution_sweep")
    a = ap.parse_args()
    h5 = a.h5
    if h5 is None and a.idx is not None:
        h5 = sorted(glob.glob(os.path.join(a.persample_dir, "*niches_independent.h5ad")))[a.idx]
    if h5 is None and a.sample:
        hits = glob.glob(os.path.join(a.persample_dir, f"{a.sample}*niches_independent.h5ad"))
        h5 = hits[0] if hits else None
    if h5 is None:
        sys.exit("need --h5, --sample, or --idx")
    run(h5, a.outroot, sample=a.sample)
