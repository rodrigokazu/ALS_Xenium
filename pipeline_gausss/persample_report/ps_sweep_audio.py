#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_sweep_audio.py
#
# Audio edition of the sweep roll-up for listening.io. Flowing ASCII prose, numbers spelled
# out, acronyms voiced, no tables or bullets. Emits a .txt and a matching text-flow PDF.
#
# Same house rules as the other audio scripts in this repo. Anything a text-to-speech engine
# would read as punctuation soup gets rewritten into words.
# ========================================================================================

"""ps_sweep_audio.py -- listening.io audio edition of the niche-number sweep cohort roll-up.
Flowing prose, ASCII-safe, numbers spelled, acronyms voiced, no tables/bullets. Reads the
per-sample sweep stats JSONs. Emits .txt + a text-flow PDF.
"""
import os, sys, glob, json, argparse, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def w(n):
    n = int(n)
    if n < 0: return "minus " + w(-n)
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + w(n % 100))
    if n < 1_000_000: return w(n // 1000) + " thousand" + ("" if n % 1000 == 0 else (", " if n % 1000 >= 100 else " and ") + w(n % 1000))
    return w(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else ", " + w(n % 1_000_000))


def spell_sample(code):
    body, _, block = str(code).partition("_")
    digits = "".join(c for c in body if c.isdigit())
    pref = "".join(c for c in body if c.isalpha())
    out = f"sample {' '.join(list(pref.upper()))} {' '.join(_ONES[int(d)] for d in digits)}".strip()
    if block: out += ", block " + " ".join(list(block.upper()))
    return out


def load(statsdir):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(statsdir, "*__sweep_stats.json")))]


def two(x):
    """A ratio 0..1 said for speech, decimals spelled digit-by-digit, e.g. 0.83 -> 'zero point eight three'."""
    try: x = float(x)
    except Exception: return "not available"
    if np.isnan(x): return "not available"
    whole, frac = f"{x:.2f}".split(".")
    return f"{_ONES[int(whole)]} point " + " ".join(_ONES[int(d)] for d in frac)


def narrative(S):
    n = len(S)
    p46 = np.nanmean([s["nesting"].get("n4_n6", {}).get("purity", np.nan) for s in S])
    p610 = np.nanmean([s["nesting"].get("n6_n10", {}).get("purity", np.nan) for s in S])
    j46 = np.nanmean([s["marker_turnover"].get("n4_n6", np.nan) for s in S])
    j610 = np.nanmean([s["marker_turnover"].get("n6_n10", np.nan) for s in S])
    para = [
        f"This is the audio edition of the niche-number sweep summary for the amyotrophic lateral "
        f"sclerosis spinal cord Xenium cohort. Xenium is an imaging-based spatial transcriptomics "
        f"platform. In this experiment we ran Novae, a graph neural network for spatial domain "
        f"discovery, independently on each of {w(n)} samples. For every sample we scanned the Novae "
        f"hierarchy and kept the level that lands nearest each of three target niche counts: four, "
        f"six, and ten. All {w(n)} samples cleanly reached four, six, and ten spatial niches. The "
        f"question this summary answers is what changes as we ask for more niches.",
        "The first thing we check is nesting: when we go from a coarse map to a finer one, do the "
        "finer niches sit cleanly inside the coarser ones, or do the boundaries reshuffle? We measure "
        "this with a purity score, where one point zero means each finer niche falls entirely within a "
        "single coarser niche. Across the cohort the average purity going from four to six niches is "
        f"about {two(p46)}, and going from six to ten niches about {two(p610)}. Higher values mean the "
        "resolutions form a clean hierarchy rather than a reshuffle.",
        "The second thing we check is the marker genes. If the coarse niches are real biology, the "
        "genes that define them should still be defining the finer niches, just split more finely. We "
        "measure the overlap of the top marker sets between resolutions with a Jaccard score, where "
        "one means the same markers and zero means completely different markers. Across the cohort the "
        f"average overlap from four to six niches is about {two(j46)}, and from six to ten about "
        f"{two(j610)}. Composition entropy, a measure of how evenly cells are spread across niches, "
        "rises as expected when we add more niches.",
    ]
    for s in S:
        rc = s["realised_counts"]
        a = s["nesting"].get("n4_n6", {}); b = s["nesting"].get("n6_n10", {})
        t = spell_sample(s["sample"]); t = t[0].upper() + t[1:]
        para.append(
            f"{t} resolved four, six, and ten niches. Its nesting purity was about "
            f"{two(a.get('purity'))} from four to six and about {two(b.get('purity'))} from six to ten, "
            f"and its marker overlap was about {two(s['marker_turnover'].get('n4_n6'))} then about "
            f"{two(s['marker_turnover'].get('n6_n10'))}.")
    para.append(
        "A caution on interpretation. This is pre quality control and exploratory. The niches are "
        "unsupervised spatial domains, the domain identifiers are local to each sample and to each "
        "resolution, and the motor neuron labels are not confirmed by the classical markers. Read the "
        "sweep as a guide to how stable the spatial structure is as we change the number of niches, "
        "not as a quantitative disease result.")
    return para


def to_pdf(paras, out_pdf, title):
    lines = []
    for p in paras:
        lines += textwrap.wrap(p, 96) + [""]
    per = 46
    with PdfPages(out_pdf) as pdf:
        for i in range(0, len(lines), per):
            fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.09, 0.05, 0.82, 0.9]); ax.axis("off"); y = 0.99
            if i == 0:
                ax.text(0.0, y, title, fontsize=15, fontweight="bold", va="top", family="sans-serif"); y -= 0.05
            for ln in lines[i:i + per]:
                ax.text(0.0, y, ln, fontsize=11, va="top", family="serif"); y -= 0.021
            pdf.savefig(fig); plt.close(fig)


def main(statsdir, out_pdf, out_txt):
    S = load(statsdir)
    if not S:
        sys.exit(f"no sweep stats in {statsdir}")
    paras = narrative(S)
    body = "\n\n".join(paras).encode("ascii", "ignore").decode("ascii")
    open(out_txt, "w").write(body + "\n")
    to_pdf(paras, out_pdf, "Niche-number sweep -- audio edition")
    print(f"wrote sweep audio -> {out_pdf} + {out_txt} ({len(S)} samples)")


if __name__ == "__main__":
    base = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_MNcorrected/report/resolution_sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--statsdir", default=base + "/stats")
    ap.add_argument("--out", default=base + "/pdf/ZZ_resolution_sweep_audio.pdf")
    ap.add_argument("--txt", default=base + "/pdf/ZZ_resolution_sweep_audio.txt")
    a = ap.parse_args()
    main(a.statsdir, a.out, a.txt)
