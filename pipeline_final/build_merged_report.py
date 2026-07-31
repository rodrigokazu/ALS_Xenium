#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/build_merged_report.py
#
# Merges the cohort PDF and all 20 per-sample PDFs into one document with a table of contents
# at the front.
#
# It shells out to pdfunite. That is not laziness; no Python PDF merge library is installed in
# any of the SCG environments, and adding one to a working scientific stack to concatenate
# files was not worth the risk.
#
# Index page numbers come from real pdfinfo page counts and not assumed chapter lengths, so
# the contents entries actually land on the right pages in the merged file.
#
# Chapters are ordered control, then sporadic, then C9.
# ========================================================================================

"""build_merged_report.py -- merge the _FINAL Dual-pass QC cohort PDF + all 20 per-sample PDFs
into ONE comprehensive PDF with a leading index/TOC page, using pdfunite (poppler-utils; no
python PDF-merge library is installed anywhere in the SCG envs). Page numbers on the index are
computed from real pdfinfo page counts so they land correctly in the merged file.
"""
import json, glob, os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPQC_ROOT = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_FINAL/dualpass_qc"
DPQC = os.path.join(DPQC_ROOT, "extracted")
STATS_DIR = os.path.join(DPQC, "stats")
PDF_DIR = os.path.join(DPQC, "pdf")
BASICQC_DIR = os.path.join(DPQC_ROOT, "basicqc")
COHORT_PDF = os.path.join(DPQC, "cohort", "DualPassQC_Cohort.pdf")
OUT_DIR = os.path.join(DPQC_ROOT, "merged")
CHAPTERS_DIR = os.path.join(OUT_DIR, "chapters")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHAPTERS_DIR, exist_ok=True)

STATUS_ORDER = {"control": 0, "sporadic": 1, "c9": 2}


def npages(pdf):
    out = subprocess.check_output(["pdfinfo", pdf]).decode()
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"no Pages: line in pdfinfo output for {pdf}")


def load_samples():
    rows = []
    for f in sorted(glob.glob(os.path.join(STATS_DIR, "*__dpqc.json"))):
        d = json.load(open(f))
        rows.append(d)
    rows.sort(key=lambda d: (STATUS_ORDER.get(str(d.get("status", "")).lower(), 9), d["sample"]))
    return rows


def main():
    rows = load_samples()
    assert len(rows) == 20, f"expected 20 samples, found {len(rows)}"

    cohort_n = npages(COHORT_PDF)
    sample_pdfs = []
    for d in rows:
        orig = os.path.join(PDF_DIR, f"{d['sample']}__DualPassQC.pdf")
        basicqc = os.path.join(BASICQC_DIR, f"{d['sample']}__basicqc.pdf")
        assert os.path.exists(orig), f"missing {orig}"
        assert os.path.exists(basicqc), f"missing {basicqc}"
        # per-sample chapter = original Dual-pass QC report + the added basic-QC page, concatenated
        # once per sample so the master merge below just treats each chapter as one file.
        chapter = os.path.join(CHAPTERS_DIR, f"{d['sample']}__chapter.pdf")
        subprocess.run(["pdfunite", orig, basicqc, chapter], check=True)
        d["_pdf"] = chapter
        d["_npages"] = npages(chapter)
        sample_pdfs.append(chapter)

    # index is built as 1 page first (fits 20 rows fine); if it overflows we regenerate as 2.
    INDEX_PAGES = 1
    start = INDEX_PAGES + cohort_n + 1  # first sample's first page (1-indexed)
    for d in rows:
        d["_start_page"] = start
        start += d["_npages"]

    # ---- render the index page ----
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.06, 0.04, 0.9, 0.92])
    ax.axis("off")
    y = 0.99
    ax.text(0.0, y, "ALS SC-Xenium _FINAL segmentation", fontsize=17, fontweight="bold",
            va="top", family="sans-serif")
    y -= 0.028
    ax.text(0.0, y, "Novae per-sample INDEPENDENT niches + Dual-pass QC -- comprehensive report",
            fontsize=12, va="top", family="sans-serif")
    y -= 0.022
    ax.text(0.0, y, "Marcel's improved uniform segmentation; per-sample QC metrics, domain\n"
                    "characterisation (Moran's I, DEG markers), KEEP/REVIEW/REMOVE verdicts.",
            fontsize=8.5, va="top", family="sans-serif", color="0.25")
    y -= 0.046
    ax.text(0.0, y, f"Cohort overview  ................................................  "
                    f"page {INDEX_PAGES + 1}", fontsize=9.5, va="top", family="monospace",
            fontweight="bold")
    y -= 0.020
    ax.text(0.0, y, "See the companion Explanation_and_Glossary.pdf (same folder) for what analysis "
                    "was run and what \"smear score\" means.", fontsize=8, va="top",
            family="sans-serif", color="#7B241C", fontweight="bold")
    y -= 0.026
    ax.text(0.0, y, "SAMPLE INDEX", fontsize=11, fontweight="bold", va="top", family="sans-serif")
    y -= 0.006
    ax.axhline(y, xmin=0.0, xmax=1.0, color="0.6", linewidth=0.6)
    y -= 0.018
    header = f"{'#':<3}{'Sample':<13}{'Status':<11}{'Cells':>9}{'K/R/X':>10}{'Remove%':>10}{'Page':>7}"
    ax.text(0.0, y, header, fontsize=8.3, va="top", family="monospace", fontweight="bold")
    y -= 0.017
    ax.axhline(y + 0.006, xmin=0.0, xmax=1.0, color="0.8", linewidth=0.4)
    for i, d in enumerate(rows, 1):
        krx = f"{d.get('n_keep', 0)}/{d.get('n_review', 0)}/{d.get('n_remove', 0)}"
        line = (f"{i:<3}{d['sample']:<13}{str(d.get('status','')):<11}"
                f"{d.get('n_cells', 0):>9,}{krx:>10}{d.get('pct_cells_remove', 0):>9.2f}%"
                f"{d['_start_page']:>7}")
        ax.text(0.0, y, line, fontsize=8.0, va="top", family="monospace")
        y -= 0.0145
    y -= 0.012
    ax.text(0.0, y, "K/R/X = domains KEEP/REVIEW/REMOVE (dual-pass QC verdict, per sample). "
                    "Remove% = share of cells in REMOVE-verdict domains.",
            fontsize=7, va="top", family="sans-serif", color="0.35")
    y -= 0.014
    ax.text(0.0, y, "PRE-QC / exploratory. Novae run independently per sample -> domain IDs and "
                    "colours are per-sample and NOT comparable across samples. Segmentation is "
                    "uniform across the cohort (Marcel's _final). Region/level confounded with "
                    "disease (controls cervical, ALS mostly lumbar). is_MN annotation pending on "
                    "_final -> MN-protection inactive in all verdicts below.",
            fontsize=7, va="top", family="sans-serif", color="0.35", wrap=True)

    index_pdf = os.path.join(OUT_DIR, "_index.pdf")
    fig.savefig(index_pdf)
    plt.close(fig)
    actual_index_pages = npages(index_pdf)
    print(f"[index] rendered {actual_index_pages} page(s) (assumed {INDEX_PAGES})")
    if actual_index_pages != INDEX_PAGES:
        print("[index] WARNING page count mismatch -- page numbers in the index text may be off "
              "by this many pages; rerun with INDEX_PAGES updated if so.")

    merged = os.path.join(OUT_DIR, "ALS_SCXenium_FINAL_Novae_DualPassQC_Comprehensive.pdf")
    cmd = ["pdfunite", index_pdf, COHORT_PDF] + sample_pdfs + [merged]
    subprocess.run(cmd, check=True)
    total = npages(merged)
    print(f"[merge] wrote {merged} ({total} pages; expected "
          f"{actual_index_pages + cohort_n + sum(d['_npages'] for d in rows)})")

    with open(os.path.join(OUT_DIR, "index_manifest.json"), "w") as fh:
        json.dump({"cohort_pages": cohort_n, "index_pages": actual_index_pages,
                    "samples": [{"sample": d["sample"], "status": d["status"],
                                 "n_cells": d.get("n_cells"), "n_keep": d.get("n_keep"),
                                 "n_review": d.get("n_review"), "n_remove": d.get("n_remove"),
                                 "pct_cells_remove": d.get("pct_cells_remove"),
                                 "start_page": d["_start_page"], "n_pages": d["_npages"]}
                                for d in rows],
                    "total_pages": total}, fh, indent=2)
    print("[manifest] wrote index_manifest.json")


if __name__ == "__main__":
    main()
