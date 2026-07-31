#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_cohort_FINAL.py
#
# The roll-up across all 20 sections, built from the per-sample stats JSONs and thumbnails.
#
# Six views: a contact sheet of every section, the keep/remove decision matrix, how the smear
# scores spread out, removal rates with confounds, the cell-type composition of what was
# dropped against what was kept, and a confound panel. It closes with a recommendation page.
#
# The dropped-versus-kept composition is the one to look at hardest. If removal is
# preferentially deleting a cell type rather than a spatial artefact, that shows up here and
# nowhere else, and it would quietly bias every downstream comparison.
#
# Confound text reflects the uniform _final segmentation, so the old two-control
# re-segmentation batch effect no longer applies. Spinal level against disease still does.
# ========================================================================================

"""dpqc_cohort_FINAL.py -- cohort roll-up for the Dual-pass QC (_FINAL re-run). Reads the per-sample
stats JSONs (and the per-sample atlas thumbnails) and builds:
  CO1 contact sheet of all sections   CO2 keep/remove decision matrix   CO3 smear-score landscape
  CO4 removal-rate bars + confounds   CO5 dropped-vs-kept cell-type composition   CO6 confound panel
then assembles a cohort PDF with a recommendation page. Colours from dpqc_style_FINAL. PRE-QC caveat
stamped. Logic identical to the proven _gausss dpqc_cohort; the confound text now reflects the UNIFORM
_final segmentation (the 2-control reseg-batch confound is resolved) + the is_MN protection state.
"""
import os, sys, glob, json, argparse, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dpqc_style_FINAL as st
import nature_style as ns

STATUS_ORDER = {"control": 0, "sporadic": 1, "c9": 2}
CAVEAT = ("PRE-QC / exploratory. Novae run independently per sample -> domain IDs are per-sample and "
          "NOT comparable across samples (domain colours are per-sample). Region/level confounded with "
          "disease (controls cervical, ALS lumbar). Segmentation is UNIFORM across the cohort (Marcel "
          "_final), so the earlier 2-control reseg-batch confound is resolved. Smear removal is a "
          "provisional, reversible, WITHIN-sample QC choice, not a biological or disease claim.")


def load(statsdir):
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(statsdir, "*__dpqc.json")))]
    S.sort(key=lambda s: (STATUS_ORDER.get(str(s["status"]).lower(), 9), s["level"], s["sample"]))
    return S


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def co1_contact(pdf, S, figroot):
    ncol = 5; nrow = int(np.ceil(len(S) / ncol))
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    fig.suptitle("CO1 | Domain atlas contact sheet (novae_domains_n6, per-sample colours)", fontsize=11, y=0.99)
    for i, s in enumerate(S):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        png = os.path.join(figroot, s["sample"], f"{s['sample']}__atlas.png")
        if os.path.exists(png):
            ax.imshow(plt.imread(png))
        ax.axis("off")
    fig.text(0.5, 0.02, "Domain colours are per-sample and not comparable across panels.",
             ha="center", fontsize=7, color="0.4")
    pdf.savefig(fig, dpi=200); plt.close(fig)


def co2_matrix(pdf, S):
    maxd = max(len(s["domains_detail"]) for s in S)
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    y = np.arange(len(S))[::-1]
    for yi, s in zip(y, S):
        dd = sorted(s["domains_detail"], key=lambda r: -r["n_cells"])
        for j, r in enumerate(dd):
            ax.add_patch(Rectangle((j, yi - 0.4), 0.92, 0.8, facecolor=st.flag_color(r["verdict"]),
                                   edgecolor="white", lw=0.5))
            ax.text(j + 0.46, yi, str(r["domain"]).replace("D", ""), ha="center", va="center",
                    fontsize=5, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s['sample']}  ({st.status_label(s['status'])[:4]}/{s['level'][:4]})" for s in S], fontsize=6)
    ax.set_xlim(-0.2, maxd); ax.set_xlabel("domain slot (ranked by size; ragged)")
    ax.set_title("CO2 | KEEP / REVIEW / REMOVE decision matrix", fontsize=11)
    ax.set_yticks(list(y), minor=False)
    for yi, s in zip(y, S):
        ax.get_yticklabels()[list(y).index(yi)].set_color(st.status_color(s["status"]))
    ax.legend(handles=st.flag_handles(), loc="lower right", frameon=False, fontsize=7, ncol=3)
    _despine(ax)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def co3_landscape(pdf, S):
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white")
    x = 0; xt, xl = [], []
    for s in S:
        for r in s["domains_detail"]:
            jitter = (hash(str(r["domain"])) % 100 - 50) / 400.0
            ax.scatter(x + jitter, r["smear_score"], s=8 + r["n_cells"] / 400.0,
                       c=st.flag_color(r["verdict"]), edgecolors="0.3", linewidths=0.2, alpha=0.85)
        xt.append(x); xl.append(s["sample"]); x += 1
    ax.axhline(0.62, ls="--", lw=0.8, color="0.3"); ax.axhline(0.42, ls=":", lw=0.8, color="0.5")
    ax.text(len(S) - 0.5, 0.63, "REMOVE thr", fontsize=6, color="0.3")
    ax.text(len(S) - 0.5, 0.43, "REVIEW thr", fontsize=6, color="0.5")
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=90, fontsize=6)
    for t, s in zip(ax.get_xticklabels(), S):
        t.set_color(st.status_color(s["status"]))
    ax.set_ylabel("composite smear score"); ax.set_ylim(0, 1)
    ax.set_title("CO3 | Smear-score landscape (point size = domain cell count; x-labels coloured by status)", fontsize=11)
    ax.legend(handles=st.flag_handles() + st.status_handles(), loc="upper left", frameon=False, fontsize=6, ncol=2)
    _despine(ax); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def co4_removal(pdf, S):
    S2 = sorted(S, key=lambda s: s.get("pct_cells_remove", 0), reverse=True)
    y = np.arange(len(S2))[::-1]
    fig, ax = plt.subplots(figsize=(8.27, 9)); fig.patch.set_facecolor("white")
    ax.barh(y, [s.get("pct_cells_remove", 0) for s in S2],
            color=[st.status_color(s["status"]) for s in S2], edgecolor="0.3", linewidth=0.3)
    for yi, s in zip(y, S2):
        ax.text(s.get("pct_cells_remove", 0) + 0.2, yi,
                f"{s.get('n_remove',0)}R/{s.get('n_review',0)}?  {s['level'][:4]}",
                va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels([s["sample"] for s in S2], fontsize=7)
    ax.set_xlabel("% of section cells flagged REMOVE")
    ax.set_title("CO4 | Removal rate per section (bar = status; label = #REMOVE/#REVIEW, level; seg uniform)", fontsize=10)
    ax.legend(handles=st.status_handles(), loc="lower right", frameon=False, fontsize=7)
    _despine(ax); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def co5_dropkeep(pdf, S):
    keep = {}; rem = {}
    for s in S:
        for r in s["domains_detail"]:
            tgt = rem if r["verdict"] == "REMOVE" else keep
            fr = r.get("celltype_fracs", {})
            if isinstance(fr, str):
                fr = json.loads(fr.replace("'", '"'))
            for ct, f in fr.items():
                tgt[ct] = tgt.get(ct, 0.0) + f * r["n_cells"]
    fig, ax = plt.subplots(figsize=(8.27, 6)); fig.patch.set_facecolor("white")
    for i, (lab, dic) in enumerate([("KEEP+REVIEW cells", keep), ("REMOVED cells", rem)]):
        tot = sum(dic.values()) or 1
        left = 0
        for ct in st.CELLTYPE_ORDER:
            v = dic.get(ct, 0) / tot
            if v <= 0:
                continue
            ax.barh(i, v, left=left, color=st.CELLTYPE_COLORS[ct], edgecolor="none")
            left += v
        ax.text(1.01, i, f"n={int(tot):,}", va="center", fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["KEEP+REVIEW", "REMOVED"]); ax.set_xlim(0, 1)
    ax.set_xlabel("cell-type fraction"); ax.set_title("CO5 | What are we removing? Composition of kept vs removed cells", fontsize=10)
    ax.legend(handles=st.celltype_handles(), loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False, fontsize=6)
    _despine(ax); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def co6_confound(pdf, S):
    df = pd.DataFrame([{"status": st.status_label(s["status"]), "level": s["level"],
                        "seg": s["seg_version"]} for s in S])
    fig, ax = plt.subplots(figsize=(8.27, 6)); fig.patch.set_facecolor("white"); ax.axis("off")
    ax.text(0.0, 1.0, "CO6 | Confound honesty panel", fontsize=13, fontweight="bold", va="top")
    tab = pd.crosstab(df["status"], df["level"])
    lines = ["Sample counts, disease status x spinal level:", ""]
    lines.append(f"{'':16s}" + "".join(f"{c:>12s}" for c in tab.columns))
    for idx, row in tab.iterrows():
        lines.append(f"{idx:16s}" + "".join(f"{int(v):>12d}" for v in row.values))
    seg_versions = sorted(str(x) for x in df["seg"].unique())
    lines += ["", f"Segmentation: UNIFORM across the cohort ({', '.join(seg_versions)}) -- the earlier "
                  f"2-control reseg-batch confound is RESOLVED in the _final run.",
              "", "Read every keep/remove pattern through this table: controls are cervical, ALS is",
              "mostly lumbar, so region and disease cannot be separated in this cohort."]
    y = 0.86
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=10, family="monospace", va="top"); y -= 0.05
    pdf.savefig(fig); plt.close(fig)


def recommendation(pdf, S):
    nrem = sum(s.get("n_remove", 0) for s in S); nrev = sum(s.get("n_review", 0) for s in S)
    nkeep = sum(s.get("n_keep", 0) for s in S)
    worst = sorted(S, key=lambda s: s.get("pct_cells_remove", 0), reverse=True)[:5]
    mn_active = sum(1 for s in S if s.get("mn_protection_active"))
    mn_line = (f"MN protection active in {mn_active}/{len(S)} sections (is_MN present; still unvalidated "
               f"vs CHAT/MNX1)."
               if mn_active else
               f"MN protection INACTIVE in all {len(S)} sections: Marcel's _final is_MN annotation was "
               f"pending, so no domain was protected by MN enrichment -- re-run once it lands.")
    L = [f"Across {len(S)} sections at novae_domains_n6: {nkeep} KEEP, {nrev} REVIEW, {nrem} REMOVE.",
         f"Mean cells flagged REMOVE per section: {np.mean([s.get('pct_cells_remove',0) for s in S]):.1f}%.",
         mn_line, ""]
    L += ["Sections with the most cells flagged (inspect first with the collaborator):"]
    for s in worst:
        L.append(f"  {s['sample']:12s} {st.status_label(s['status']):14s} {s['level']:9s} "
                 f"REMOVE {s.get('n_remove',0)} domains, {s.get('pct_cells_remove',0):.1f}% cells")
    L += ["", "Recommended workflow:",
          "  1  Accept KEEP domains as the working spatial compartments.",
          "  2  Walk every REVIEW domain with the collaborator against its highlight map.",
          "  3  Confirm REMOVE domains, then re-check them at n4 and n10; keep only stable calls.",
          "  4  Drop or reassign removed cells; rebuild cleaned domains; compare ONLY region-matched samples.",
          "", "-" * 92, "CAVEAT:"] + ["  " + x for x in textwrap.wrap(CAVEAT, 90)]
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.86, 0.9]); ax.axis("off")
    ax.text(0.0, 0.99, "Cohort recommendation", fontsize=15, fontweight="bold", va="top", family="sans-serif")
    y = 0.93
    for ln in L:
        ax.text(0.0, y, ln, fontsize=9.5, family="monospace", va="top"); y -= 0.0205
    pdf.savefig(fig); plt.close(fig)


def main(statsdir, figroot, out_pdf):
    S = load(statsdir)
    if not S:
        sys.exit(f"no stats in {statsdir}")
    with PdfPages(out_pdf) as pdf:
        co1_contact(pdf, S, figroot)
        co2_matrix(pdf, S)
        co3_landscape(pdf, S)
        co4_removal(pdf, S)
        co5_dropkeep(pdf, S)
        co6_confound(pdf, S)
        recommendation(pdf, S)
    print(f"wrote cohort PDF -> {out_pdf} ({len(S)} samples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statsdir", required=True)
    ap.add_argument("--figroot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.statsdir, a.figroot, a.out)
