#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_sweep_cohort.py
#
# Cohort roll-up of the resolution sweep, built from the per-sample stats JSONs.
#
# Four pages: a table of realised counts, chosen levels, nesting purity and ARI and marker
# turnover; a heatmap of which Novae level got chosen per target per sample; nesting quality
# bars; and marker turnover with composition entropy against resolution.
#
# The level heatmap is the interesting one. Because each sample was scanned independently, the
# level that yields six domains varies between sections, and that variation is itself a
# statement about how comparable the sections are.
#
# Stamped pre-QC.
# ========================================================================================

"""ps_sweep_cohort.py -- cohort roll-up of the per-sample niche-number sweep.
Reads the per-sample sweep stats JSONs (ps_sweep.py). Pages:
  1 cohort table (realised counts, chosen levels, nesting purity/ARI, marker turnover)
  2 chosen Novae level per target x sample (heatmap -- shows the per-sample level variation)
  3 nesting quality: purity + ARI (n4->n6, n6->n10) per sample (bars)
  4 marker-set turnover per sample (bars) + composition entropy vs resolution
PRE-QC caveat stamped.
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PREQC = ("PRE-QC / exploratory. INDEPENDENT per-sample Novae, per-sample level scan targeting 4/6/10 "
         "domains. Domain IDs are per-sample & per-resolution (not label-comparable across samples); "
         "we compare resolutions by nesting quality and marker turnover, not by ID. N=1 per section.")


def load(statsdir):
    S = []
    for p in sorted(glob.glob(os.path.join(statsdir, "*__sweep_stats.json"))):
        S.append(json.load(open(p)))
    return S


def entropy(comp):
    v = np.array([c for d, c in comp.items() if d != "unassigned"], float)
    p = v / v.sum() if v.sum() else v
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def is_ctrl(s):
    return str(s).lower().startswith("control")


def _text_page(pdf, lines, title=None, fs=8):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.05, 0.04, 0.92, 0.92]); ax.axis("off"); y = 0.99
    if title:
        ax.text(0.0, y, title, fontsize=14, fontweight="bold", va="top", family="sans-serif"); y -= 0.04
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=fs, va="top", family="monospace"); y -= 0.02
    pdf.savefig(fig); plt.close(fig)


def page_table(pdf, S):
    L = [f"{'sample':13s}{'status':9s}{'n4':>3s}{'n6':>3s}{'n10':>4s}  "
         f"{'pur46':>6s}{'ari46':>6s}{'pur610':>7s}{'ari610':>7s}  {'J46':>5s}{'J610':>5s}", "-" * 78]
    for s in S:
        rc = s["realised_counts"]; ne = s["nesting"]; tu = s["marker_turnover"]
        a = ne.get("n4_n6", {}); b = ne.get("n6_n10", {})
        L.append(f"{s['sample']:13s}{s['status']:9s}{rc['n4']:>3d}{rc['n6']:>3d}{rc['n10']:>4d}  "
                 f"{a.get('purity',0):>6.2f}{a.get('ari',0):>6.2f}{b.get('purity',0):>7.2f}{b.get('ari',0):>7.2f}  "
                 f"{tu.get('n4_n6',0):>5.2f}{tu.get('n6_n10',0):>5.2f}")
    L += ["", "pur=nesting purity (1.0=finer niches sit cleanly inside coarser); ARI=agreement;",
          "J=Jaccard overlap of pooled top-5 markers between the two resolutions."]
    _text_page(pdf, L, title="Niche-number sweep -- cohort table")


def page_levels(pdf, S):
    tgts = ["n4", "n6", "n10"]
    names = [s["sample"] for s in S]
    M = np.full((len(S), 3), np.nan)
    for i, s in enumerate(S):
        cl = s.get("chosen_levels", {})
        for j, t in enumerate(tgts):
            # keys may be like 'n4'/'level_n4'/'4'; try a few
            for key in (t, f"level_{t}", t.replace("n", ""), f"target_{t}"):
                if key in cl and str(cl[key]).lstrip("-").isdigit():
                    M[i, j] = int(cl[key]); break
    fig, ax = plt.subplots(figsize=(6, 9))
    if np.isnan(M).all():
        ax.text(0.5, 0.5, "chosen-level metadata not present in stats\n(uns['novae_attrs'] had no level keys)",
                ha="center", va="center"); ax.axis("off")
    else:
        im = ax.imshow(M, aspect="auto", cmap="cividis")
        ax.set_xticks(range(3)); ax.set_xticklabels(tgts)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7, color="w")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Novae hierarchy level chosen")
        ax.set_title("Chosen Novae level per target count (per-sample scan)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_nesting(pdf, S):
    names = [s["sample"] for s in S]
    x = np.arange(len(names))
    pur46 = [s["nesting"].get("n4_n6", {}).get("purity", np.nan) for s in S]
    pur610 = [s["nesting"].get("n6_n10", {}).get("purity", np.nan) for s in S]
    ari46 = [s["nesting"].get("n4_n6", {}).get("ari", np.nan) for s in S]
    ari610 = [s["nesting"].get("n6_n10", {}).get("ari", np.nan) for s in S]
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    axes[0].bar(x - 0.2, pur46, 0.4, label="n4->n6", color="#4C72B0")
    axes[0].bar(x + 0.2, pur610, 0.4, label="n6->n10", color="#DD8452")
    axes[0].set_ylabel("nesting purity"); axes[0].set_ylim(0, 1.02); axes[0].legend(frameon=False)
    axes[0].set_title("How cleanly finer niches nest inside coarser ones (1.0 = perfect)")
    axes[1].bar(x - 0.2, ari46, 0.4, label="n4->n6", color="#4C72B0")
    axes[1].bar(x + 0.2, ari610, 0.4, label="n6->n10", color="#DD8452")
    axes[1].set_ylabel("adjusted Rand index"); axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_turnover(pdf, S):
    names = [s["sample"] for s in S]; x = np.arange(len(names))
    j46 = [s["marker_turnover"].get("n4_n6", np.nan) for s in S]
    j610 = [s["marker_turnover"].get("n6_n10", np.nan) for s in S]
    ent = {t: [entropy(s["composition"][t]) for s in S] for t in ("n4", "n6", "n10")}
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    axes[0].bar(x - 0.2, j46, 0.4, label="n4->n6", color="#55A868")
    axes[0].bar(x + 0.2, j610, 0.4, label="n6->n10", color="#C44E52")
    axes[0].set_ylabel("marker Jaccard overlap"); axes[0].set_ylim(0, 1.02); axes[0].legend(frameon=False)
    axes[0].set_title("Top-marker set overlap between resolutions (high = coarse markers persist)")
    for t, c in zip(("n4", "n6", "n10"), ("#8172B3", "#937860", "#DA8BC3")):
        axes[1].plot(x, ent[t], marker="o", ms=3, label=t, color=c)
    axes[1].set_ylabel("composition entropy (bits)"); axes[1].legend(frameon=False, title="resolution")
    axes[1].set_title("Niche-composition entropy rises with more niches")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main(statsdir, out_pdf):
    import textwrap
    S = load(statsdir)
    if not S:
        sys.exit(f"no sweep stats in {statsdir}")
    with PdfPages(out_pdf) as pdf:
        page_table(pdf, S)
        page_levels(pdf, S)
        page_nesting(pdf, S)
        page_turnover(pdf, S)
        _text_page(pdf, [""] + textwrap.wrap(PREQC, 88), title="Provenance / caveat")
    print(f"wrote cohort sweep PDF -> {out_pdf} ({len(S)} samples)")


if __name__ == "__main__":
    base = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_MNcorrected/report/resolution_sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--statsdir", default=base + "/stats")
    ap.add_argument("--out", default=base + "/pdf/ZZ_resolution_sweep.pdf")
    a = ap.parse_args()
    main(a.statsdir, a.out)
