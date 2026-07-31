#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_report_FINAL.py
#
# Assembles one sample's PDF chapter: cover, the four figures, the shared colour key, the
# decision ledger, a plain-language recommendation, and a methods and caveats box.
#
# One figure per page with generous margins. The report is meant to be read by a collaborator
# in a meeting, not mined by whoever wrote it, so legibility beats density.
#
# The caveats box is not decoration. It states that this is pre-QC, that domain ids are
# per-sample and not comparable, and what the motor-neuron protection status was. Those
# qualifications need to travel with the figures, because pages get screenshotted and pasted
# into slides without their context.
# ========================================================================================

"""dpqc_report_FINAL.py -- assemble the per-sample visual PDF for the Dual-pass QC (_FINAL re-run).
Embeds the Nature-tier figures (master / highlight / markers / qcpanels) + a cover, the shared
colour key, a decision ledger, a plain-language recommendation, and a methods/caveats box.
Minimalist: one figure per page, generous margins, consistent typography. Logic identical to the
proven _gausss dpqc_report; caveats updated for the uniform _final segmentation + the is_MN state.
"""
import os, sys, textwrap, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dpqc_style_FINAL as st


def caveat_text(meta):
    """Per-sample caveat; the is_MN clause reflects whether MN protection was active in this run."""
    mn = ("The is_MN motor-neuron protector was ACTIVE (is_MN present) but remains UNVALIDATED "
          "(no CHAT/MNX1 confirmation) and was used only as a weak protective prior."
          if meta.get("mn_protection_active")
          else "The is_MN motor-neuron protector was INACTIVE for this run because Marcel's is_MN "
               "annotation on the _final segmentation was still pending, so no domain was protected "
               "by MN enrichment (all other verdicts are unaffected).")
    return ("PRE-QC / EXPLORATORY. Marcel's improved _final segmentation, applied UNIFORMLY across "
            "all sections (the earlier per-batch segmentation confound is resolved). Novae run "
            "INDEPENDENTLY per sample (min_counts>=1 only), so domain IDs are sample-specific and "
            "NOT comparable across samples. Region/spinal level is confounded with disease (controls "
            "cervical, ALS mostly lumbar). Cell-type labels cover only the higher-count cells (the "
            "untyped fraction is shown, never hidden). " + mn + " Smear removal is a provisional, "
            "reversible QC choice, not a biological claim.")

METHODS = ("Each domain is scored WITHIN its own section (immune to the cross-sample confounds) on "
           "three axes. Spatial coherence: Moran's I of the domain indicator, k-nearest-neighbour "
           "same-domain purity, PAS (percentage of abnormal spots), CHAOS (within-domain 1-NN micron "
           "distance), largest connected-component fraction, and convex-hull spread. Expression "
           "identity: number of strong specific positive markers (log fold-change over 1 and adjusted "
           "p below 1e-10), the top-10 marker sign balance (the 'one gene propping it up' test), and "
           "novae_latent silhouette. Transcript/QC: median transcript counts vs the section median, "
           "transcript density, negative-control fraction, nucleus-free fraction, and untyped fraction. "
           "VERDICT is a two-gate rule: a domain is flagged REMOVE only if it is BOTH spatially "
           "incoherent AND fails identity/QC; a strong specific marker signature or motor-neuron "
           "enrichment protects a rare-but-real domain (downgraded to REVIEW). KEEP = coherent + real "
           "identity; REVIEW = mixed signals or under-powered (n<50).")


def _text_page(pdf, lines, title=None, fontsize=10, family="monospace"):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.86, 0.9]); ax.axis("off")
    y = 0.99
    if title:
        ax.text(0.0, y, title, fontsize=15, fontweight="bold", va="top", family="sans-serif")
        y -= 0.045
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=fontsize, va="top", family=family)
        y -= 0.0195
        if y < 0.03:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.08, 0.05, 0.86, 0.9]); ax.axis("off"); y = 0.99
    pdf.savefig(fig); plt.close(fig)


def _img_page(pdf, png, caption=None):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.88]); ax.axis("off")
    if os.path.exists(png):
        ax.imshow(plt.imread(png))
    else:
        ax.text(0.5, 0.5, f"[missing: {os.path.basename(png)}]", ha="center")
    if caption:
        fig.text(0.5, 0.035, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    pdf.savefig(fig, dpi=200); plt.close(fig)


def _wrap(s, w=92):
    return textwrap.wrap(s, w) or [""]


def recommendation_lines(meta, df):
    d = df.copy()
    d["dom"] = d["domain"].astype(str)
    rem = d[d.verdict == "REMOVE"]; rev = d[d.verdict == "REVIEW"]; kep = d[d.verdict == "KEEP"]
    L = []
    L += _wrap(f"Recommendation for {meta['sample']} ({st.status_label(meta['status'])}, "
               f"{meta['level']}, {meta['seg_version']}). Of {len(d)} domains at novae_domains_n6: "
               f"{meta.get('n_keep',0)} KEEP, {meta.get('n_review',0)} REVIEW, "
               f"{meta.get('n_remove',0)} REMOVE. Removing the flagged domains drops "
               f"{meta.get('pct_cells_remove',0)}% of the section's cells.")
    L += [""]
    if len(rem):
        L += ["REMOVE (drop before downstream domain analysis):"]
        for _, r in rem.sort_values("smear_score", ascending=False).iterrows():
            L += _wrap(f"  - {r['dom']}: n={int(r['n_cells']):,} ({r['pct_cells']:.1f}%), "
                       f"smear {r['smear_score']:.2f}. {r['reason']}", 88)
        L += [""]
    if len(rev):
        L += ["REVIEW (inspect with the collaborator; do not auto-drop):"]
        for _, r in rev.sort_values("smear_score", ascending=False).iterrows():
            L += _wrap(f"  - {r['dom']}: n={int(r['n_cells']):,} ({r['pct_cells']:.1f}%), "
                       f"dominant {r['dominant_celltype']}. {r['reason']}", 88)
        L += [""]
    if len(kep):
        L += ["KEEP (coherent territories with a real identity):"]
        for _, r in kep.sort_values("n_cells", ascending=False).iterrows():
            mk = r.get("top_marker") or "-"
            L += _wrap(f"  - {r['dom']}: n={int(r['n_cells']):,} ({r['pct_cells']:.1f}%), "
                       f"{r['dominant_celltype']} (top marker {mk}).", 88)
    return L


def build_persample_pdf(sample, meta, df, figdir, colourkey_png, out_pdf):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    v = meta
    with PdfPages(out_pdf) as pdf:
        # cover
        head = [
            f"Sample: {sample}",
            f"Disease status: {st.status_label(meta['status'])}",
            f"Spinal level: {meta['level']}    Segmentation: {meta['seg_version']} (uniform across cohort)",
            f"Cells: {meta['n_cells']:,}    Panel genes: {meta['n_genes']}    "
            f"Typed fraction: {meta['typed_frac_sample']*100:.0f}%",
            f"Novae resolution: {meta['resolution']}",
            f"MN protection: {'ACTIVE' if meta.get('mn_protection_active') else 'INACTIVE (is_MN annotation pending)'}"
            f"   (is_MN source: {meta.get('is_MN_source', 'n/a')})",
            "",
            f"Domain verdicts:  KEEP {v.get('n_keep',0)}   REVIEW {v.get('n_review',0)}   "
            f"REMOVE {v.get('n_remove',0)}",
            f"Cells flagged REMOVE: {v.get('pct_cells_remove',0)}%",
            "",
            "Contents:",
            "  1  Colour key",
            "  2  Hero page: domain atlas + composition + count/area + smear score + ledger",
            "  3  Per-domain spatial highlights (the smear detector)",
            "  4  Marker signatures per domain",
            "  5  Morphology / negative-control / spatial-coherence panels",
            "  6  Decision ledger and recommendation",
            "  7  Methods and caveats",
            "", "-" * 92, "CAVEAT:",
        ]
        for ln in _wrap(caveat_text(meta)):
            head.append("  " + ln)
        _text_page(pdf, head, title=f"Dual-pass QC report  |  {sample}", family="monospace")
        _img_page(pdf, colourkey_png, "Shared colour key: cell types (global), status, verdict; domain colours are per-sample.")
        _img_page(pdf, os.path.join(figdir, f"{sample}__master.png"),
                  f"Fig 1 | {sample}: hero page. Domain atlas, per-domain cell-type composition, "
                  "transcript-count and cell-area violins, ranked smear score, and the decision ledger.")
        _img_page(pdf, os.path.join(figdir, f"{sample}__highlight.png"),
                  f"Fig 2 | {sample}: each domain isolated over the section in grey. A real compartment "
                  "lights up as a connected territory; a smear scatters as diffuse pepper. Frame colour = verdict.")
        _img_page(pdf, os.path.join(figdir, f"{sample}__markers.png"),
                  f"Fig 3 | {sample}: top markers per domain (dot size = fraction expressing, colour = "
                  "mean z-expression). A domain with no coherent positive programme is a smear.")
        _img_page(pdf, os.path.join(figdir, f"{sample}__qcpanels.png"),
                  f"Fig 4 | {sample}: nucleus-free fraction, negative-control burden, and the spatial-coherence triad.")
        _text_page(pdf, recommendation_lines(meta, df), title="Decision & recommendation", family="monospace", fontsize=9)
        _text_page(pdf, _wrap(METHODS) + [""] + ["CAVEATS:"] + ["  " + x for x in _wrap(caveat_text(meta))],
                   title="Methods & caveats", family="sans-serif", fontsize=10)
    return out_pdf
