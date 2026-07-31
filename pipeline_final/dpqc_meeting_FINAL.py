#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_meeting_FINAL.py
#
# The same evidence rearranged for a conversation. An intro on how to run the meeting, one
# page per section ordered control then sporadic then C9, each with the atlas, the highlight
# thumbnails, the ledger and concrete talking points, then a closing checklist.
#
# Kept apart from the cohort PDF because the audiences differ. The cohort document argues a
# position; this one walks somebody through twenty sections without losing them, and the
# ordering by disease group is what makes the pattern visible as you turn pages.
# ========================================================================================

"""dpqc_meeting_FINAL.py -- the sample-by-sample MEETING GUIDE for walking a collaborator through the
Dual-pass QC (_FINAL re-run). An intro on how to run the meeting, then ONE page per section (ordered
control -> sporadic -> C9): the domain atlas + per-domain highlight grid thumbnails, the decision
ledger, and concrete talking points. Closing checklist. Logic identical to the proven _gausss
dpqc_meeting; confound text updated for the uniform _final segmentation + the is_MN state.
"""
import os, sys, glob, json, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dpqc_style_FINAL as st

STATUS_ORDER = {"control": 0, "sporadic": 1, "c9": 2}


def load(statsdir):
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(statsdir, "*__dpqc.json")))]
    S.sort(key=lambda s: (STATUS_ORDER.get(str(s["status"]).lower(), 9), s["level"], s["sample"]))
    return S


def intro_page(pdf, S):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.86, 0.9]); ax.axis("off")
    ax.text(0.0, 0.99, "Dual-pass QC -- sample-by-sample meeting guide", fontsize=16,
            fontweight="bold", va="top")
    nrem = sum(s.get("n_remove", 0) for s in S); nrev = sum(s.get("n_review", 0) for s in S)
    mn_active = sum(1 for s in S if s.get("mn_protection_active"))
    mn_line = ("MN protection is active (is_MN present, still unvalidated vs CHAT/MNX1)."
               if mn_active else
               "MN protection is INACTIVE this run: Marcel's _final is_MN annotation was pending, so no "
               "domain was protected by motor-neuron enrichment -- re-run once it lands.")
    body = [
        f"This guide walks {len(S)} spinal-cord sections one at a time so you and the collaborator can "
        "sign off on which Novae domains to keep and which to remove before downstream analysis.",
        "",
        "How to run each section (about two to three minutes each):",
        "  1  ATLAS: does the domain map look like a spinal cord? Note the gross layout.",
        "  2  HIGHLIGHT GRID: for each domain ask -- territory or pepper? A real compartment is a",
        "     connected patch; a smear scatters across the whole slice. The frame colour is the verdict.",
        "  3  LEDGER: accept the KEEP rows; DECIDE the REVIEW rows together; CONFIRM the REMOVE rows",
        "     against the highlight grid. If you disagree with a call, note it -- the rule is transparent.",
        "  4  RECORD the decision. The per-sample decisions CSV keeps the figures and pipeline in step.",
        "",
        f"Provisional totals to confirm: {nrem} domains proposed REMOVE, {nrev} proposed REVIEW across "
        "the cohort.",
        "",
        "Keep the confounds in view the whole time: domain IDs are per-sample (not comparable across",
        "sections); controls are cervical and ALS mostly lumbar (no disease read-out); segmentation is",
        "uniform across the cohort (Marcel _final), so the earlier reseg-batch confound is resolved.",
        mn_line,
        "Spend your time on the REVIEW domains -- that is where your judgement matters most.",
        "Everything here is pre-QC and reversible.",
        "",
        "Verdict legend:  KEEP = coherent territory with a real identity;  REVIEW = mixed signals /",
        "protected / under-powered;  REMOVE = spatially incoherent AND identity/quality failure.",
    ]
    y = 0.93
    for ln in body:
        ax.text(0.0, y, ln, fontsize=10, va="top", family="monospace"); y -= 0.023
    # verdict swatches
    x = 0.0
    for f in ("KEEP", "REVIEW", "REMOVE"):
        ax.add_patch(Rectangle((x, y - 0.02), 0.03, 0.02, facecolor=st.flag_color(f), transform=ax.transAxes))
        ax.text(x + 0.04, y - 0.01, f, fontsize=9, va="center"); x += 0.18
    pdf.savefig(fig); plt.close(fig)


def sample_page(pdf, s, figroot):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    samp = s["sample"]
    fig.suptitle(f"{samp}  |  {st.status_label(s['status'])}  |  {s['level']}  |  {s['seg_version']}  |  "
                 f"{s['n_cells']:,} cells", fontsize=12, y=0.985, color=st.status_color(s["status"]))
    # atlas (top-left) + highlight (top-right)
    axa = fig.add_axes([0.04, 0.60, 0.44, 0.34]); axa.axis("off")
    pa = os.path.join(figroot, samp, f"{samp}__atlas.png")
    if os.path.exists(pa): axa.imshow(plt.imread(pa))
    axa.set_title("domain atlas", fontsize=8)
    axh = fig.add_axes([0.50, 0.60, 0.46, 0.34]); axh.axis("off")
    ph = os.path.join(figroot, samp, f"{samp}__highlight.png")
    if os.path.exists(ph): axh.imshow(plt.imread(ph))
    axh.set_title("per-domain highlights (territory vs pepper)", fontsize=8)
    # ledger
    axl = fig.add_axes([0.06, 0.06, 0.88, 0.50]); axl.axis("off")
    dd = sorted(s["domains_detail"], key=lambda r: (-{"REMOVE": 2, "REVIEW": 1, "KEEP": 0}[r["verdict"]], -r["n_cells"]))
    axl.text(0.0, 1.0, "Decision ledger & talking points", fontsize=11, fontweight="bold", va="top")
    y = 0.94
    hdr = f"{'domain':8s}{'verdict':8s}{'n':>8s}{'%':>6s} {'dominant':16s} reason"
    axl.text(0.0, y, hdr, fontsize=7.5, family="monospace", va="top"); y -= 0.035
    for r in dd:
        axl.add_patch(Rectangle((0.0, y - 0.018), 0.012, 0.02, facecolor=st.flag_color(r["verdict"]),
                                transform=axl.transAxes))
        line = (f"{str(r['domain']):7s} {r['verdict']:8s}{int(r['n_cells']):>8,}{r['pct_cells']:>5.1f} "
                f"{str(r.get('dominant_celltype','?'))[:15]:16s} {str(r['reason'])[:60]}")
        axl.text(0.02, y - 0.008, line, fontsize=7, family="monospace", va="center")
        y -= 0.032
        if y < 0.18:
            break
    # talking points
    rem = [r for r in dd if r["verdict"] == "REMOVE"]; rev = [r for r in dd if r["verdict"] == "REVIEW"]
    tps = []
    if rem:
        tps.append("CONFIRM removal of " + ", ".join(str(r["domain"]) for r in rem) +
                   " against the highlight grid (should read as pepper).")
    if rev:
        tps.append("DECIDE together on " + ", ".join(str(r["domain"]) for r in rev) +
                   " -- check whether the marker signature is real before dropping.")
    if not rem and not rev:
        tps.append("All domains read as clean territories -- accept as-is.")
    tps.append(f"About {s.get('pct_cells_remove',0)}% of cells would be dropped; verify this does not delete a whole cell type.")
    y = max(y - 0.02, 0.14)
    axl.text(0.0, y, "Talking points:", fontsize=8.5, fontweight="bold", va="top"); y -= 0.028
    for tp in tps:
        for ln in textwrap.wrap("- " + tp, 96):
            axl.text(0.0, y, ln, fontsize=8, va="top"); y -= 0.022
    pdf.savefig(fig); plt.close(fig)


def closing_page(pdf, S):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.86, 0.9]); ax.axis("off")
    ax.text(0.0, 0.99, "After the meeting -- checklist", fontsize=15, fontweight="bold", va="top")
    body = [
        "1  Finalise the per-sample keep/remove decisions in the decisions CSVs.",
        "2  Re-run the REMOVE calls at n4 and n10; keep only the calls that agree across resolutions.",
        "3  Validate is_MN against CHAT / MNX1 before trusting any motor-neuron-bearing domain.",
        "4  Decide: drop removed cells, or reassign them to the nearest coherent domain (latent + space).",
        "5  Rebuild cleaned per-sample domains from the surviving cells.",
        "6  Only then, and only within region-matched samples, begin the biological comparison.",
        "",
        "Every decision here is provisional, within-sample, and reversible. Nothing in this guide is a",
        "disease finding -- region, segmentation batch and pre-QC status are all confounded with status.",
    ]
    y = 0.92
    for ln in body:
        ax.text(0.0, y, ln, fontsize=10.5, va="top", family="monospace"); y -= 0.026
    pdf.savefig(fig); plt.close(fig)


def main(statsdir, figroot, out_pdf):
    S = load(statsdir)
    if not S:
        sys.exit(f"no stats in {statsdir}")
    with PdfPages(out_pdf) as pdf:
        intro_page(pdf, S)
        for s in S:
            sample_page(pdf, s, figroot)
        closing_page(pdf, S)
    print(f"wrote meeting guide -> {out_pdf} ({len(S)} sections)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statsdir", required=True); ap.add_argument("--figroot", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.statsdir, a.figroot, a.out)
