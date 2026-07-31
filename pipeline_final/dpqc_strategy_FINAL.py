#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_strategy_FINAL.py
#
# The white paper behind the method: why domains are scored the way they are, what the two
# gates mean, and what the verdicts are and are not evidence of. Extensive in content,
# deliberately plain in styling.
#
# Three outputs: a visual PDF with a two-gate schematic, the listening.io audio edition, and
# the raw TTS text. Pass --statsdir to fold real cohort numbers into the prose instead of
# describing the method in the abstract.
#
# This is the document to hand a reviewer who asks why a domain was dropped. The justification
# needs to exist independently of the figures, because "the script said REMOVE" is not a
# defensible answer.
# ========================================================================================

"""dpqc_strategy_FINAL.py -- the STRATEGY documents for the Dual-pass QC of the per-sample Novae
domains (_FINAL re-run). Extensive in content, minimalist in style. Produces:
  DualPassQC_Strategy.pdf            clean visual white paper (sections + a two-gate schematic)
  DualPassQC_Strategy_Audio_RKS.pdf  listening.io audio edition (flowing ASCII prose)
  DualPassQC_Strategy_Audio_RKS.txt  the raw TTS text
Optionally folds in cohort numbers if a dpqc stats dir is given (--statsdir). Method text identical to
the proven _gausss dpqc_strategy; the data/confound passages updated for the uniform _final
segmentation + the is_MN protection state.
"""
import os, sys, glob, json, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    import dpqc_style_FINAL as st
    FLAG = st.FLAG_COLORS
except Exception:
    FLAG = {"KEEP": "#44AA99", "REVIEW": "#E69F00", "REMOVE": "#AA4499"}

INK = "#1a1a1a"; MUTE = "#6b6b6b"; RULE = "#c9c9c9"; ACCENT = "#0072B2"

# ------------------------------------------------------------------ visual white paper content
SECTIONS = [
    ("1  Purpose",
     ["This document explains the strategy behind the dual-pass quality-control characterisation of "
      "the per-sample Novae spatial domains in the amyotrophic lateral sclerosis (ALS) spinal-cord "
      "Xenium cohort. It states what we are deciding, on what evidence, and with what safeguards, so "
      "that a collaborator can follow and sign off on the removal of low-quality (\"smear\") domains "
      "sample by sample.",
      "It is a methods-and-rationale companion to the per-sample figure reports, the cohort summary, "
      "and the sample-by-sample meeting guide."]),
    ("2  The problem",
     ["Novae was run independently on each of the twenty sections, discovering spatial domains from "
      "the local neighbourhood of gene expression. Most domains carve the tissue into recognisable "
      "anatomical territories -- white matter, grey matter, vasculature, meninges. But some domains "
      "do not form a territory at all: they scatter across the whole section like pepper and tend to "
      "collect low-count, ambient, or segmentation-spillover cells. We call these \"smear\" domains. "
      "Left in, they contaminate every downstream domain-level analysis. The task is to identify and "
      "remove them, defensibly and reversibly, before going further."]),
    ("3  What a smear domain is",
     ["A smear has two signatures at once. Spatially, it is incoherent: its cells are not neighbours "
      "of one another, so it fails every measure of spatial continuity. Molecularly, it has no real "
      "identity: it carries no coherent set of positive marker genes, its transcript counts sit below "
      "the section's, its negative-control burden is elevated, and it is enriched for cells too "
      "low-quality to have been typed at all.",
      "A domain that is merely small, or merely diffuse, is NOT automatically a smear. Motor neurons, "
      "vessels, meninges and microglia are genuinely sparse or spread out yet biologically real. The "
      "whole strategy is built to separate a real-but-sparse compartment from a true smear."]),
    ("4  The data and its confounds",
     ["The object is the independent per-sample Novae run on Marcel's improved _final segmentation, "
      "applied uniformly to all twenty sections: ten controls (mostly cervical), six sporadic ALS and "
      "four C9orf72 ALS (mostly lumbar), a four-hundred-and-eighty gene panel, three domain resolutions "
      "per sample (four, six and ten domains; six is primary). The data is pre-quality-control: only "
      "cells with at least one transcript were kept.",
      "Two confounds are load-bearing and never leave the page. Domain identifiers are per-sample and "
      "are NOT comparable across samples. Spinal level is confounded with disease (controls cervical, "
      "ALS lumbar), so no disease conclusion is valid here. Unlike the earlier run, the segmentation is "
      "now uniform across the whole cohort, so the previous 2-control segmentation-batch confound is "
      "resolved.",
      "One operational note: the motor-neuron protector (is_MN) is active only once Marcel's is_MN "
      "annotation on the _final segmentation is joined in. While that annotation is pending, no domain "
      "is protected by motor-neuron enrichment and all other verdicts are unaffected; re-run once it "
      "lands to switch the protector on."]),
    ("5  The strategy in one line",
     ["Score every domain WITHIN its own section on three axes of evidence, and remove a domain only "
      "when it fails on two independent fronts at once -- spatial incoherence AND a molecular / quality "
      "failure -- while a genuine marker identity or motor-neuron enrichment always protects a "
      "rare-but-real domain from deletion.",
      "Working within each section makes the decision immune to the cross-sample confounds: every "
      "threshold is relative to that section's own domains, medians and nulls, never to a global cutoff."]),
    ("6  The three axes of evidence",
     ["Spatial coherence -- does the domain form a territory? Moran's I of domain membership, "
      "k-nearest-neighbour same-domain purity, the percentage of abnormal spots (PAS), the CHAOS "
      "dispersion score, the largest connected-component fraction, and convex-hull spread. These are "
      "the field-standard acceptance metrics for a spatial-domain solution.",
      "Expression identity -- does the domain mean something? The number of strong, specific positive "
      "markers; the balance of positive versus negative genes among the top ten (the test that catches "
      "a domain propped up by a single ambient gene); and the domain's silhouette in the Novae "
      "embedding.",
      "Transcript, quality and composition -- is it built from good cells? Median transcript counts "
      "against the section, transcript density, negative-control fraction, nucleus-free fraction, the "
      "dominant cell type and its purity, and the fraction of cells too low-quality to be typed."]),
    ("7  The verdict: a two-gate rule",
     ["KEEP: the domain is spatially coherent AND has a real positive identity, with a concordant "
      "dominant cell type. These are the anatomical territories.",
      "REMOVE: the domain fails BOTH the spatial gate AND the identity/quality gate. Requiring both is "
      "what stops a real-but-diffuse compartment being deleted on one metric alone. Removal excludes "
      "the domain's cells from downstream domain analysis; it is reversible and is not a biological claim.",
      "REVIEW: the signals conflict, or the domain is under-powered (fewer than fifty cells), or a "
      "strong marker signature / motor-neuron enrichment has protected it from removal. These go to the "
      "collaborator for a human decision.",
      "Every verdict ships with the numbers behind it and a one-clause reason, so the call is auditable, "
      "not a black box."]),
    ("8  The colour system",
     ["Four fixed registries keep a colour meaning one thing. Cell types get a single global, "
      "biologically mnemonic palette (the only labels comparable across samples). Disease status is a "
      "grey-baseline plus a warm ALS pair. The verdict uses a colourblind-safe teal / amber / purple, "
      "deliberately avoiding red-green. Domain colours are assigned per sample by descending size and "
      "never travel between samples, because the domains themselves do not.",
      "Cell-type colour never means a domain; verdict colour never means a status. The key is printed "
      "once and reused, so every figure and every page agrees."]),
    ("9  What you get",
     ["Per sample: a hero page (domain atlas, cell-type composition, transcript-count and cell-area "
      "distributions, a ranked smear score, and a decision ledger), a per-domain spatial highlight "
      "grid that makes a smear visible to the eye, a marker-signature dotplot, a morphology / "
      "negative-control / coherence panel, and an assembled PDF with a recommendation.",
      "Across the cohort: a twenty-section contact sheet, a keep/remove decision matrix, a smear-score "
      "landscape, a per-sample removal-rate bar with the confounds shown alongside, and a check that "
      "removal is dropping low-quality cells rather than a whole biological cell type.",
      "Plus this strategy note, an audio edition of both the strategy and the cohort summary for "
      "listening on the move, and a sample-by-sample meeting guide."]),
    ("10  How the meeting runs",
     ["Go section by section. For each, read the hero page top-left to bottom-right: does the atlas "
      "look like a spinal cord? Then the highlight grid: which domains are territories and which are "
      "pepper? Then the ledger: accept the KEEP calls, decide the REVIEW calls together, confirm the "
      "REMOVE calls against the highlight grid. Record the decision; the machine-readable decisions "
      "file keeps figure and pipeline in agreement.",
      "Spend the time on the REVIEW domains -- that is where human judgement adds what the metrics cannot."]),
    ("11  What not to conclude",
     ["No disease comparison: it is fully confounded with region, segmentation batch and pre-QC status. "
      "No cross-sample domain counting: the identifiers are arbitrary per section. A domain is not "
      "unreal because it is small or under-typed. A smear is not \"no biology\" -- it is cells too poorly "
      "captured to place. And is_MN is not ground truth until it is validated against CHAT and MNX1.",
      "These figures are exploratory quality control, not statistics and not paper evidence."]),
    ("12  Next steps",
     ["Validate the motor-neuron flag against canonical markers before trusting any motor-neuron-bearing "
      "domain. Re-run the smear calls at the four- and ten-domain resolutions and keep only the calls "
      "that agree, marking the rest uncertain. For confirmed removals, decide whether to drop the cells "
      "or to reassign them to the nearest coherent domain by embedding and space. Then rebuild the "
      "cleaned domains and, only within region-matched samples, begin the biological read-out."]),
]

TWO_GATE_NOTE = ("Two-gate decision: a domain is removed only when BOTH gates fail. One gate failing, "
                 "a protected identity, or too few cells sends it to REVIEW.")


def _titlepage(pdf):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(Rectangle((0.10, 0.62), 0.36, 0.006, color=ACCENT, transform=ax.transAxes))
    ax.text(0.10, 0.70, "Dual-pass QC of per-sample Novae domains", fontsize=23, fontweight="bold",
            va="bottom", color=INK)
    ax.text(0.10, 0.655, "Strategy for identifying and removing smear domains", fontsize=13, color=MUTE)
    ax.text(0.10, 0.55, "ALS spinal-cord Xenium cohort  |  Cooper-Knock Lab", fontsize=11, color=INK)
    ax.text(0.10, 0.52, "Independent per-sample Novae niches  |  Marcel _final segmentation (uniform)",
            fontsize=10, color=MUTE)
    ax.text(0.10, 0.14, "Extensive in content, minimalist in form.", fontsize=10, style="italic", color=MUTE)
    ax.text(0.10, 0.11, "Pre-QC / exploratory. Not for statistical or disease inference.", fontsize=9, color=MUTE)
    pdf.savefig(fig); plt.close(fig)


def _section_page(pdf, blocks):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = 0.94
    for (title, paras) in blocks:
        ax.text(0.10, y, title, fontsize=14, fontweight="bold", color=INK, va="top")
        ax.add_patch(Rectangle((0.10, y - 0.018), 0.80, 0.0015, color=RULE))
        y -= 0.042
        for p in paras:
            for ln in textwrap.wrap(p, 92):
                ax.text(0.10, y, ln, fontsize=9.7, color=INK, va="top")
                y -= 0.0182
            y -= 0.010
        y -= 0.016
    pdf.savefig(fig); plt.close(fig)


def _schematic_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.10, 0.93, "The two-gate decision", fontsize=16, fontweight="bold", color=INK, va="top")
    ax.add_patch(Rectangle((0.10, 0.905), 0.80, 0.0015, color=RULE))

    def box(x, y, w, h, text, fc="#ffffff", ec=INK, fs=9.5, tc=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=1.1, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1, label=None, color=INK):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
        if label:
            ax.text((x0 + x1) / 2 + 0.02, (y0 + y1) / 2, label, fontsize=8, color=color, va="center")

    box(0.34, 0.82, 0.32, 0.05, "A domain in a section", fc="#f3f6fa", bold=True)
    box(0.10, 0.68, 0.34, 0.07, "GATE 1 -- spatial coherence\nMoran's I, PAS, largest-CC,\nCHAOS, purity, hull spread")
    box(0.56, 0.68, 0.34, 0.07, "GATE 2 -- identity / quality\nstrong markers, neg-control,\ntranscript counts, untyped")
    arrow(0.46, 0.82, 0.30, 0.755); arrow(0.54, 0.82, 0.70, 0.755)
    box(0.30, 0.52, 0.40, 0.06, "Both gates FAIL ?", fc="#faf3f6", bold=True)
    arrow(0.27, 0.68, 0.40, 0.585); arrow(0.73, 0.68, 0.60, 0.585)
    box(0.08, 0.36, 0.26, 0.06, "REMOVE", fc=FLAG["REMOVE"], ec=FLAG["REMOVE"], tc="white", bold=True, fs=12)
    box(0.37, 0.36, 0.26, 0.06, "REVIEW", fc=FLAG["REVIEW"], ec=FLAG["REVIEW"], tc="white", bold=True, fs=12)
    box(0.66, 0.36, 0.26, 0.06, "KEEP", fc=FLAG["KEEP"], ec=FLAG["KEEP"], tc="white", bold=True, fs=12)
    arrow(0.40, 0.52, 0.21, 0.425, "yes,\nnot protected", FLAG["REMOVE"])
    arrow(0.50, 0.52, 0.50, 0.425, "one gate,\nor protected,\nor n<50", FLAG["REVIEW"])
    arrow(0.60, 0.52, 0.79, 0.425, "neither\ngate fails", FLAG["KEEP"])
    box(0.20, 0.16, 0.60, 0.07,
        "PROTECT: a strong specific marker signature, or motor-neuron enrichment,\n"
        "never lets a rare-but-real domain be REMOVED -- it is downgraded to REVIEW.",
        fc="#f7f7f0", ec=MUTE, fs=9)
    for ln in textwrap.wrap(TWO_GATE_NOTE, 96):
        ax.text(0.10, 0.10, ln, fontsize=9, color=MUTE, va="top"); break
    ax.text(0.10, 0.075, "  ".join(f"{k}" for k in ("KEEP", "REVIEW", "REMOVE")), fontsize=0, color="white")
    pdf.savefig(fig); plt.close(fig)


def build_visual(out_pdf, cohort_line=None):
    with PdfPages(out_pdf) as pdf:
        _titlepage(pdf)
        # 3 section pages, balanced
        _section_page(pdf, SECTIONS[0:4])
        _section_page(pdf, SECTIONS[4:7])
        _schematic_page(pdf)
        _section_page(pdf, SECTIONS[7:10])
        _section_page(pdf, SECTIONS[10:12])
        if cohort_line:
            _section_page(pdf, [("Cohort snapshot", [cohort_line])])
    print("wrote", out_pdf)


# ------------------------------------------------------------------ audio edition
AUDIO = [
 ("cover",
  "This is the audio edition of the dual pass quality control strategy for the amyotrophic lateral "
  "sclerosis spinal cord Xenium study. Xenium is an imaging based spatial transcriptomics platform. "
  "This recording explains, in plain language, how we decide which of the automatically discovered "
  "spatial domains to keep and which to throw away before any further analysis, and why the method is "
  "built the way it is. It is a companion to the per sample figure reports and to the sample by sample "
  "meeting guide."),
 ("the problem",
  "We used a method called Novae, a graph neural network that finds spatial domains from the local "
  "neighbourhood of gene expression, and we ran it separately on each of the twenty tissue sections. "
  "Most of the domains it finds are real pieces of anatomy, such as white matter, grey matter, blood "
  "vessels and the meninges. But some domains are not a place at all. Instead of forming one connected "
  "patch of tissue, they scatter across the whole slice like pepper, and they tend to collect cells "
  "that are low in signal, contaminated by ambient molecules, or the product of a segmentation error. "
  "We call these smear domains. If we leave them in, they poison every later analysis. Our job is to "
  "find them and remove them, carefully, and in a way we can undo."),
 ("what a smear is",
  "A smear domain fails on two fronts at the same time. In space, it is incoherent, meaning its cells "
  "are not close to one another, so it scores badly on every measure of spatial continuity. In "
  "molecular terms, it has no identity, meaning it carries no consistent set of marker genes that are "
  "switched on, its transcript counts sit below the rest of the section, its negative control signal "
  "is high, and it is full of cells so poor in quality that they could not even be assigned a cell "
  "type. The important subtlety is that a domain which is merely small, or merely spread out, is not "
  "automatically a smear. Motor neurons, blood vessels, the meninges and microglia are all genuinely "
  "sparse or scattered, yet completely real. The entire strategy exists to tell a real but sparse "
  "population apart from a true smear."),
 ("the data and its confounds",
  "The material is the independent per sample Novae run on Marcel's improved final segmentation, "
  "applied uniformly to all twenty sections. There are twenty sections. Ten are controls, taken mostly "
  "from the neck, that is the cervical cord. Six are sporadic amyotrophic lateral sclerosis and four "
  "carry the C nine or f seventy two expansion, and these disease cases come mostly from the lower "
  "back, that is the lumbar cord. The gene panel has four hundred and eighty genes. For each section "
  "Novae was asked for three different numbers of domains, namely four, six and ten, and we treat six "
  "as the primary view. The data is pre quality control, meaning the only filter so far is that a cell "
  "must carry at least one transcript. Two confounds must be kept in mind at all times. First, the "
  "domain names are local to each section and cannot be compared by name from one section to another. "
  "Second, the spinal level is tangled with the disease, because controls are from the neck and the "
  "disease cases are from the lower back, so no comparison of disease versus control is valid here. "
  "Unlike the earlier run, the segmentation is now uniform across the whole cohort, so the previous "
  "concern that two of the control sections came from a separate segmentation batch no longer applies. "
  "One more caution: the motor neuron protector is active only once Marcel's motor neuron annotation on "
  "the final segmentation is joined in; while it is pending, no domain is protected by motor neuron "
  "enrichment, and all other verdicts are unaffected."),
 ("the strategy in one line",
  "The strategy is this. Score every domain within its own section, on three axes of evidence, and "
  "remove a domain only when it fails on two independent fronts at once, that is, spatial incoherence "
  "together with a molecular or quality failure. At the same time, a genuine marker identity, or an "
  "enrichment of motor neurons, always protects a rare but real domain from deletion. Working within "
  "each section, rather than across sections, is deliberate. It makes every threshold relative to that "
  "section's own domains and its own typical values, which is exactly what neutralises the confounds "
  "of region, batch and disease."),
 ("the three axes of evidence",
  "The first axis is spatial coherence, which asks whether the domain forms a territory. We measure "
  "Moran's I, a spatial autocorrelation of domain membership; the fraction of a cell's nearest spatial "
  "neighbours that share its domain; the percentage of abnormal spots; a dispersion score called "
  "CHAOS; the fraction of the domain sitting in its single largest connected blob; and how far its "
  "convex hull spreads across the section. The second axis is expression identity, which asks whether "
  "the domain means anything. We count the strong, specific marker genes it switches on; we test the "
  "balance of positive against negative genes among its top ten markers, which catches a domain that "
  "is propped up by a single leaky gene; and we measure how tight and separable it is in the Novae "
  "embedding. The third axis is transcript quality and composition, which asks whether the domain is "
  "built from good cells. We look at its median transcript count against the section, its transcript "
  "density, its negative control burden, the share of cells with no nucleus, its dominant cell type "
  "and how pure that is, and the fraction of its cells too poor to be typed at all."),
 ("how a verdict is reached",
  "Each domain receives one of three verdicts. Keep means the domain is spatially coherent and has a "
  "real positive identity with a matching dominant cell type. These are the anatomical territories, and "
  "they are accepted. Remove means the domain fails both the spatial gate and the identity and quality "
  "gate. Requiring both to fail is the heart of the method, because it prevents a real but diffuse "
  "compartment being deleted on the strength of a single unhappy number. Removing a domain simply takes "
  "its cells out of the later domain level analysis; it is fully reversible and it is not a biological "
  "claim. Review means the signals disagree, or the domain has fewer than fifty cells and is therefore "
  "under powered, or a strong marker signature or a cluster of motor neurons has protected it from "
  "removal. These are handed to the collaborator for a human decision. Every verdict comes with the "
  "numbers behind it and a short reason, so nothing is a black box."),
 ("what you get",
  "For each section you get a hero page that shows the domain map, the cell type make up of every "
  "domain, the spread of transcript counts and cell sizes, a ranked smear score, and a decision ledger. "
  "You get a grid that isolates each domain over a grey background, so a smear is visible to the naked "
  "eye as a scatter of pepper while a real domain lights up as a solid patch. You get a marker dot plot "
  "and a panel of quality measures. And you get an assembled document with a written recommendation. "
  "Across the whole cohort you get a contact sheet of all twenty sections, a keep and remove decision "
  "matrix, a landscape of every domain's smear score, a bar chart of how much each section loses to "
  "removal with the confounds shown beside it, and a safety check confirming that removal is dropping "
  "low quality cells rather than an entire biological population. There is also this strategy, an audio "
  "cohort summary, and a meeting guide."),
 ("how the meeting runs",
  "In the meeting, go section by section. For each one, read the hero page from the top left to the "
  "bottom right. First ask whether the domain map even looks like a spinal cord. Then study the "
  "highlight grid and ask, for each domain, is this a territory or is this pepper. Then read the ledger, "
  "accept the keep calls, decide the review calls together, and confirm each remove call against the "
  "highlight grid. Record every decision. The machine readable decisions file keeps the figures and the "
  "pipeline in agreement. Spend most of the time on the review domains, because that is where human "
  "judgement adds what the numbers cannot."),
 ("what not to conclude",
  "Be disciplined about the limits. Do not draw any disease conclusion, because disease is completely "
  "confounded with region, with segmentation batch, and with the fact that the data is pre quality "
  "control. Do not count how many sections contain a given domain, because the domain names are "
  "arbitrary from one section to the next. Do not decide a domain is unreal just because it is small or "
  "because many of its cells are untyped. Do not equate a smear with an absence of biology; a smear is "
  "made of cells that are too poorly captured to place. And do not treat the motor neuron flag as ground "
  "truth until it has been checked against the classical markers, choline acetyltransferase and M N X "
  "one. These pictures are exploratory quality control, not statistics, and not evidence for a paper."),
 ("next steps and close",
  "The next steps are clear. Validate the motor neuron flag against the canonical markers before "
  "trusting any domain that claims to hold motor neurons. Repeat the smear calls at the four domain and "
  "ten domain resolutions, and keep only the calls that agree, marking the rest as uncertain. For the "
  "confirmed removals, decide whether to delete the cells outright or to reassign them to their nearest "
  "coherent domain in space and in the embedding. Then rebuild the cleaned domains, and only within "
  "region matched sections, begin to read out the biology. That is the strategy: score within each "
  "section, demand two independent failures before deleting anything, protect the rare but real, keep "
  "an audit trail, and never let a confound masquerade as a finding. Thank you for listening."),
]


def build_audio(out_pdf, out_txt):
    paras = [p for _, p in AUDIO]
    body = "\n\n".join(paras).encode("ascii", "ignore").decode("ascii")
    with open(out_txt, "w") as f:
        f.write(body + "\n")
    lines = []
    for p in paras:
        lines += textwrap.wrap(p, 96); lines.append("")
    with PdfPages(out_pdf) as pdf:
        per = 44
        for i in range(0, len(lines), per):
            fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.10, 0.05, 0.80, 0.9]); ax.axis("off")
            y = 0.99
            if i == 0:
                ax.text(0.0, y, "Dual-pass QC strategy -- audio edition", fontsize=15,
                        fontweight="bold", va="top", family="sans-serif")
                ax.text(0.0, y - 0.035, "Built for Listening.io. Flowing prose, no tables.",
                        fontsize=9, va="top", color="0.4"); y -= 0.075
            for ln in lines[i:i + per]:
                ax.text(0.0, y, ln, fontsize=11, va="top", family="serif"); y -= 0.0205
            pdf.savefig(fig); plt.close(fig)
    print("wrote", out_pdf, "and", out_txt)


def cohort_snapshot(statsdir):
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(statsdir, "*__dpqc.json")))]
    if not S:
        return None
    nrem = sum(s.get("n_remove", 0) for s in S); nrev = sum(s.get("n_review", 0) for s in S)
    nkeep = sum(s.get("n_keep", 0) for s in S)
    pct = np.mean([s.get("pct_cells_remove", 0) for s in S])
    return (f"Across {len(S)} sections at the six-domain resolution, the pipeline marks {nkeep} domains "
            f"KEEP, {nrev} REVIEW and {nrem} REMOVE, dropping on average {pct:.1f}% of cells per section. "
            f"These are provisional, within-sample calls for the collaborator to confirm.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--statsdir", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cl = cohort_snapshot(a.statsdir) if a.statsdir and os.path.isdir(a.statsdir) else None
    build_visual(os.path.join(a.outdir, "DualPassQC_Strategy.pdf"), cohort_line=cl)
    build_audio(os.path.join(a.outdir, "DualPassQC_Strategy_Audio_RKS.pdf"),
                os.path.join(a.outdir, "DualPassQC_Strategy_Audio_RKS.txt"))
