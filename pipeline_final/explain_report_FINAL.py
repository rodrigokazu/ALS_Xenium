#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/explain_report_FINAL.py
#
# A standalone glossary answering the question that came back after the first report went out:
# what is a smear score and what were these analyses actually doing.
#
# The design decision worth preserving is that it reads its numbers and thresholds directly
# out of dpqc_compute_FINAL.py instead of restating them. Documentation that paraphrases code
# goes stale silently and then misleads with authority. This cannot drift, but it does have to
# be re-run after any change to dpqc_compute.
#
# Covers the pipeline overview, the two gates, how the score is built, a glossary, and a page
# on how to read the report.
# ========================================================================================

"""explain_report_FINAL.py -- standalone "how to read this report" / glossary PDF for the
ALS SC-Xenium _FINAL Novae + Dual-pass QC comprehensive report. Built at the user's explicit
request ("explanation pdf that tells me the analyses done and what it is calling smear score").
Every number/threshold here is read directly out of dpqc_compute_FINAL.py, not paraphrased from
memory, so it stays correct if that file changes -- re-run this script after any dpqc_compute edit.
"""
import os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/dualpass_qc/merged/Explanation_and_Glossary.pdf"

PAGE = (8.27, 11.69)
MUTE = "0.3"


def _page():
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.09, 0.05, 0.84, 0.90]); ax.axis("off")
    return fig, ax


def _wrap(t, width=100):
    return "\n".join(textwrap.wrap(t, width=width))


def _para(ax, y, text, fontsize=9.5, width=98, gap=0.018, color="black", family="sans-serif"):
    wrapped = _wrap(text, width)
    n = wrapped.count("\n") + 1
    ax.text(0.0, y, wrapped, fontsize=fontsize, va="top", family=family, color=color)
    return y - gap * n - 0.012


def _h1(ax, y, text):
    ax.text(0.0, y, text, fontsize=14, fontweight="bold", va="top", family="sans-serif")
    ax.axhline(y - 0.012, xmin=0.0, xmax=1.0, color="0.7", linewidth=0.7)
    return y - 0.045


def _h2(ax, y, text):
    ax.text(0.0, y, text, fontsize=11.5, fontweight="bold", va="top", family="sans-serif", color="#1B4F72")
    return y - 0.032


def cover(pdf):
    fig, ax = _page()
    ax.text(0.0, 0.72, "How to read this report", fontsize=24, fontweight="bold", va="top")
    ax.text(0.0, 0.665, "Explanation and glossary for the ALS SC-Xenium _FINAL", fontsize=13, va="top", color=MUTE)
    ax.text(0.0, 0.635, "Novae per-sample niches + Dual-pass QC comprehensive report", fontsize=13, va="top", color=MUTE)
    ax.axhline(0.60, xmin=0.0, xmax=0.35, color="#1B4F72", linewidth=2.5)
    ax.text(0.0, 0.55, "ALS spinal-cord Xenium cohort  |  Cooper-Knock Lab", fontsize=10, va="top")
    ax.text(0.0, 0.52, "Companion to: ALS_SCXenium_FINAL_Novae_DualPassQC_Comprehensive.pdf (168+ pages)",
            fontsize=9.5, va="top", color=MUTE)
    ax.text(0.0, 0.10, "This document explains, in plain terms, what analysis produced every number and\n"
                       "figure in the comprehensive report, and precisely defines \"smear score\" and every\n"
                       "other metric it is built from. All thresholds and formulas below are read directly\n"
                       "from the pipeline code (dpqc_compute_FINAL.py), not paraphrased.",
            fontsize=9, va="top", color=MUTE)
    ax.text(0.0, 0.02, "Pre-QC / exploratory. Not for statistical or disease inference.", fontsize=8.5, va="top", color="0.45")
    pdf.savefig(fig); plt.close(fig)


def page_overview(pdf):
    fig, ax = _page()
    y = _h1(ax, 0.97, "1. What analysis was run")
    y = _para(ax, y, "The starting point is Marcel's improved \"_final\" cell segmentation, applied "
                     "UNIFORMLY to all 20 sections of the ALS spinal-cord Xenium cohort (10 control, "
                     "6 sporadic ALS, 4 C9orf72 ALS; a 480-gene panel). This resolves an earlier confound "
                     "where two batches used different segmentations.")
    y = _h2(ax, y, "Step 1 -- Novae spatial niches (per sample, INDEPENDENT)")
    y = _para(ax, y, "Novae (a graph neural network for spatial niche discovery) is run separately on "
                     "each of the 20 sections -- a fresh model per sample, with no information shared "
                     "across sections. This means niche identities (\"D1014\", \"D1015\", ...) are "
                     "PER-SAMPLE labels, assigned by descending cell count, and are NOT the same biology "
                     "from one sample to the next -- only marker identity lets you compare across samples. "
                     "Graph: Delaunay triangulation on true micron coordinates, 99th-percentile edge "
                     "pruning. Cells are filtered to min_counts>=1 only (pre-QC). Four resolutions are "
                     "scanned per sample and matched to target niche counts of 4, 6, 8 and 10 (6 is the "
                     "primary resolution used throughout the report).")
    y = _h2(ax, y, "Step 2 -- Dual-pass Quality Control (this report's core)")
    y = _para(ax, y, "Some Novae niches are real anatomy (white matter, grey matter, vasculature, "
                     "meninges). Others are \"smear\" domains: pepper-like scatters of low-quality, "
                     "ambient, or segmentation-spillover cells that Novae groups together because they "
                     "share a lack of real signal, not because they share biology. Dual-pass QC scores "
                     "every niche in every sample on two independent axes -- spatial coherence and "
                     "molecular/QC identity -- and hands down one of three verdicts: KEEP, REVIEW, or "
                     "REMOVE. This is the \"dual-pass\" in the name: a niche is only removed if it fails "
                     "BOTH passes at once (see section 3).")
    y = _h2(ax, y, "Step 3 -- Differential expression (DEGs) per niche")
    y = _para(ax, y, "For every niche with >=2 groups in a sample, scanpy's rank_genes_groups (Wilcoxon "
                     "rank-sum test, one niche vs. the rest of that same sample) ranks every gene. A gene "
                     "counts as a \"strong marker\" for a niche if log-fold-change > 1 AND "
                     "Benjamini-Hochberg adjusted p < 1e-10. The marker dotplot/heatmap page in each "
                     "sample chapter shows the top-ranked genes per niche; the glossary (section 5) "
                     "explains how the marker COUNT feeds into the QC verdict.")
    y = _h2(ax, y, "Step 4 -- Spatial autocorrelation (Moran's I) per niche")
    y = _para(ax, y, "Moran's I measures whether a niche's cells sit next to each other (spatially "
                     "clustered, I close to 1) or are scattered at random through the section (I close "
                     "to 0 or negative). It is computed once per niche, on that niche's binary "
                     "membership indicator over the same spatial neighbour graph Novae used, and is one "
                     "of the three GATE-1 (spatial coherence) checks in section 3.")
    y = _h2(ax, y, "Step 5 -- Basic QC metrics (added, this page's companion)")
    y = _para(ax, y, "Each sample chapter also carries a dedicated basic-QC page: whole-section AND "
                     "per-niche distributions of transcript counts per cell, cell area, genes detected "
                     "per cell, and negative-control fraction -- the classic Xenium QC quantities, broken "
                     "out per niche so a smear niche's degraded profile is visible directly, not just "
                     "inferred from the composite score.")
    pdf.savefig(fig); plt.close(fig)


def page_gates(pdf):
    fig, ax = _page()
    y = _h1(ax, 0.97, "2. The KEEP / REVIEW / REMOVE decision")
    ax.text(0.0, y + 0.008, "the two-gate rule", fontsize=9.5, va="top", color=MUTE, style="italic")
    y -= 0.012
    y = _para(ax, y, "A niche is REMOVED only if it fails BOTH of two independent gates at once. Failing "
                     "only one gate, or being protected (see below), downgrades the call to REVIEW "
                     "instead of REMOVE. This two-gate design exists so that a niche which is merely "
                     "small, or merely diffuse -- motor neurons, blood vessels, meninges and microglia "
                     "are all genuinely sparse/spread-out but real -- is never auto-removed on one signal "
                     "alone.")
    y = _h2(ax, y, "GATE 1 -- spatial coherence (fails if ANY of):")
    for line in [
        "Moran's I < 0.10  (the niche's cells are not spatially clustered)",
        "largest connected component < 0.20 of the niche's cells  (it is not one contiguous patch)",
        "PAS (percent abnormal spots -- fraction of a cell's neighbours in a DIFFERENT niche) > 0.50",
    ]:
        ax.text(0.03, y, "- " + line, fontsize=9.3, va="top", family="monospace"); y -= 0.021
    y -= 0.012
    y = _h2(ax, y, "GATE 2 -- molecular identity / QC (fails if ANY of):")
    for line in [
        "<= 1 strong marker gene (logFC>1, padj<1e-10 vs. rest of section)",
        "negative-control fraction > 3.0x the section median",
        "median transcript count < 0.6x the section median",
        "untyped-cell fraction > 1.5x the sample's baseline untyped fraction",
    ]:
        ax.text(0.03, y, "- " + line, fontsize=9.3, va="top", family="monospace"); y -= 0.021
    y -= 0.012
    y = _h2(ax, y, "PROTECTION (overrides a REMOVE down to REVIEW):")
    y = _para(ax, y, ">= 3 strong marker genes, OR motor-neuron fraction more than 2x the sample's "
                     "overall MN fraction (and > 1% of the niche) -- a real-but-rare compartment is "
                     "never silently deleted.", width=95)
    y -= 0.01
    y = _h2(ax, y, "The special \"unassigned\" niche")
    y = _para(ax, y, "Cells Novae could not confidently place in ANY niche (graph-invalid / "
                     "under-connected) are pooled into \"unassigned\" and always verdict REMOVE -- this "
                     "is a bookkeeping bucket, not a scored niche.", width=95)
    y -= 0.02
    ax.axhline(y, xmin=0, xmax=1, color="0.85", linewidth=0.6); y -= 0.03
    y = _para(ax, y, "IMPORTANT: this KEEP/REVIEW/REMOVE verdict is decided directly from the raw gate "
                     "metrics above -- it is NOT computed by thresholding the composite \"smear score\" "
                     "described next. The smear score is a separate, continuous 0-1 ranking used for the "
                     "bar-chart visualisations and for sorting domains worst-to-best at a glance; the two "
                     "are highly correlated in practice but are not the same calculation.",
                fontsize=9.3, color="#7B241C")
    pdf.savefig(fig); plt.close(fig)


def page_score(pdf):
    fig, ax = _page()
    y = _h1(ax, 0.97, "3. The composite \"smear score\"")
    ax.text(0.0, y + 0.008, "0 = clean, 1 = maximally smear-like", fontsize=9.5, va="top", color=MUTE, style="italic")
    y -= 0.012
    y = _para(ax, y, "A single continuous number per niche, built as a weighted average of four "
                     "sub-scores, each itself an average of 0-1-normalised metrics oriented so that "
                     "HIGHER always means MORE smear-like:")
    y -= 0.01
    ax.text(0.0, y, "smear_score = 0.45 x coherence + 0.25 x identity + 0.15 x qc + 0.15 x composition",
            fontsize=10.5, family="monospace", fontweight="bold", va="top")
    y -= 0.045
    rows = [
        ("coherence (45%)", "1-purity, 1-Moran's I, 1-largest_cc_frac, PAS, scaled CHAOS ratio, "
                              "scaled convex-hull spread ratio -- averaged"),
        ("identity (25%)", "inverted strong-marker count, inverted Novae latent silhouette -- averaged"),
        ("qc (15%)", "inverted transcript-count ratio vs. section median, negative-control ratio -- averaged"),
        ("composition (15%)", "untyped-cell fraction (scaled), cell-type Shannon entropy -- averaged"),
    ]
    for name, desc in rows:
        ax.text(0.0, y, name, fontsize=9.5, fontweight="bold", va="top", family="monospace")
        y -= 0.019
        y = _para(ax, y, desc, fontsize=8.7, width=100, color="0.25")
        y -= 0.006
    y -= 0.015
    y = _h2(ax, y, "Why a composite score AND a two-gate verdict?")
    y = _para(ax, y, "The gate rule (section 2) is what actually decides KEEP/REVIEW/REMOVE -- it is "
                     "auditable, threshold-by-threshold, and defensible in a methods section. The "
                     "composite score exists purely so a human scanning the cohort-level \"smear-score "
                     "landscape\" or the per-sample ranked bar chart can see AT A GLANCE which niches are "
                     "worst, and by how much, without re-deriving six separate metrics in their head. "
                     "Treat the score as a ranking aid, and the verdict as the actual decision.")
    y -= 0.015
    y = _h2(ax, y, "Where you see it in the report")
    y = _para(ax, y, "Cohort page CO3 (\"smear-score landscape\", one dot per niche per sample) and the "
                     "\"Smear score + verdict\" ranked bar chart on every sample's hero page (page 2 of "
                     "each chapter).", width=98)
    pdf.savefig(fig); plt.close(fig)


def page_glossary(pdf):
    fig, ax = _page()
    y = _h1(ax, 0.97, "4. Metric glossary")
    entries = [
        ("Moran's I", "Spatial autocorrelation of a niche's binary membership over the section's spatial "
                       "neighbour graph. ~1 = tightly clustered territory; ~0/negative = scattered pepper."),
        ("largest_cc_frac", "Fraction of a niche's cells that belong to its single largest spatially-"
                             "connected component. Low = the niche is fragmented into many small islands."),
        ("PAS (percent abnormal spots)", "Mean, over a niche's cells, of the fraction of each cell's "
                                          "spatial k-nearest-neighbours that belong to a DIFFERENT niche. "
                                          "High = the niche's cells are surrounded by other niches, not "
                                          "each other."),
        ("nbhd_purity", "Mean fraction of a cell's spatial neighbours sharing its own niche label (the "
                         "complement of PAS in spirit, used inside the coherence sub-score)."),
        ("CHAOS / chaos_ratio", "Mean within-niche 1-nearest-neighbour micron distance, divided by the "
                                 "section-wide equivalent. High = cells nominally 'in' the niche are far "
                                 "apart from their nearest same-niche neighbour -- diffuse, not packed."),
        ("hull_area_frac / spread_ratio", "The niche's convex-hull area as a fraction of the section's "
                                           "hull, compared to the niche's cell-count fraction. A niche "
                                           "spread over disproportionately more area than its cell share "
                                           "warrants scores worse."),
        ("n_strong_markers", "Count of genes with logFC>1 and adjusted p<1e-10 for this niche vs. the "
                              "rest of the section (Wilcoxon rank_genes_groups). The molecular-identity "
                              "backbone of GATE 2."),
        ("latent_silhouette", "Silhouette score of the niche's cells in Novae's learned latent embedding "
                               "space -- how well-separated the niche is from others in the space Novae "
                               "itself uses to define niches."),
        ("count_ratio", "Niche's median transcript count divided by the section-wide median. <1 = this "
                         "niche's cells carry less signal than a typical cell in the section."),
        ("negctrl_ratio", "Niche's mean negative-control-probe fraction divided by the section median. "
                           ">1 = elevated background/noise relative to the rest of the section."),
        ("untyped_ratio", "Niche's %-untyped-cells (no confident coarse cell-type call) divided by the "
                           "sample's overall untyped baseline. >1 = this niche is disproportionately "
                           "made of cells too low-quality to type at all."),
        ("celltype_entropy", "Shannon entropy of the niche's coarse cell-type composition (excluding "
                              "Untyped). High = no single cell type dominates -- a mixed bag rather than "
                              "a coherent tissue compartment."),
        ("is_MN / MN fraction", "Motor-neuron calls. On the _FINAL segmentation, Marcel's is_MN "
                                 "annotation is still pending, so is_MN is all-False everywhere in this "
                                 "report and MN protection is INACTIVE -- this is stated on every sample "
                                 "chapter's cover and does not affect any other verdict."),
    ]
    for term, desc in entries:
        ax.text(0.0, y, term, fontsize=9.3, fontweight="bold", va="top", family="sans-serif")
        y -= 0.017
        y = _para(ax, y, desc, fontsize=8.3, width=104, color="0.2", gap=0.0155)
        y -= 0.004
    pdf.savefig(fig); plt.close(fig)


def page_howtoread(pdf):
    fig, ax = _page()
    y = _h1(ax, 0.97, "5. How the comprehensive PDF is laid out")
    y = _para(ax, y, "ALS_SCXenium_FINAL_Novae_DualPassQC_Comprehensive.pdf follows this structure:")
    y -= 0.008
    items = [
        ("Index / TOC (page 1)", "every sample, its disease status, cell count, KEEP/REVIEW/REMOVE "
                                  "niche counts, %cells flagged REMOVE, and its chapter's starting page."),
        ("Cohort overview", "20-section domain-atlas contact sheet, keep/remove decision matrix, "
                             "smear-score landscape, removal-rate bars with confounds, and dropped-vs-"
                             "kept cell-type composition."),
        ("Per-sample chapter (repeated 20x)", "1 colour key -- 2 hero page (domain atlas, cell-type "
                                               "composition, transcript-count/cell-area violins, ranked "
                                               "smear score, decision ledger) -- 3 per-domain spatial "
                                               "highlights (the smear detector) -- 4 marker "
                                               "dotplot/heatmap (DEGs) -- 5 morphology / negative-control "
                                               "/ spatial-coherence panels -- 6 basic-QC page (whole-"
                                               "section + per-niche transcript counts, cell area, genes "
                                               "detected, negative-control %) -- 7 decision ledger + "
                                               "recommendation -- 8 methods and caveats."),
    ]
    for name, desc in items:
        ax.text(0.0, y, name, fontsize=9.8, fontweight="bold", va="top", family="sans-serif", color="#1B4F72")
        y -= 0.02
        y = _para(ax, y, desc, fontsize=8.6, width=102, color="0.2")
        y -= 0.012
    y -= 0.01
    y = _h2(ax, y, "Companion documents (same folder / cohort/ subfolder on OAK)")
    for line in [
        "DualPassQC_Cohort.pdf -- the cohort-level figures reproduced inside the comprehensive PDF.",
        "DualPassQC_Strategy.pdf -- the original methods-and-rationale strategy note (7pp).",
        "DualPassQC_MeetingGuide.pdf -- a section-by-section walkthrough for a sign-off meeting.",
        "*_Audio_RKS.pdf/.txt -- listening.io audio editions of the strategy and cohort summary.",
    ]:
        ax.text(0.02, y, "- " + line, fontsize=8.6, va="top", family="sans-serif"); y -= 0.021
    y -= 0.02
    ax.axhline(y, xmin=0, xmax=1, color="0.85", linewidth=0.6); y -= 0.03
    y = _para(ax, y, "PRE-QC / EXPLORATORY throughout. Novae domain IDs are per-sample and not "
                     "comparable across samples. Spinal level is confounded with disease (controls "
                     "cervical, ALS mostly lumbar). Segmentation is uniform across the cohort (Marcel's "
                     "_final). is_MN annotation is pending, so MN protection is inactive everywhere. "
                     "Smear removal is a provisional, reversible, within-sample QC choice -- not a "
                     "biological or disease claim.", fontsize=8.8, color="0.3")
    pdf.savefig(fig); plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        cover(pdf)
        page_overview(pdf)
        page_gates(pdf)
        page_score(pdf)
        page_glossary(pdf)
        page_howtoread(pdf)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
