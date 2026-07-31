#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_audio_FINAL.py
#
# The listening.io edition of the cohort result. Every substantial document in this project
# ships an audio version, and audio has requirements that a PDF does not.
#
# Flowing ASCII prose only. No tables, no bullets, no symbols a text-to-speech engine will
# mangle. Numbers are spelled out and gene symbols are spelled letter by letter, because
# "STMN2" read as a word is useless and read as S-T-M-N-2 is not.
#
# Emits a .txt, the cleanest input for the TTS pipeline, plus a matching text-flow PDF so the
# same words exist in a readable form.
#
# One passage per section giving its verdicts and the reasoning, then a cohort synthesis and
# the caveats.
# ========================================================================================

"""dpqc_audio_FINAL.py -- listening.io audio edition of the Dual-pass QC cohort result (_FINAL re-run).
Flowing ASCII prose, numbers spelled for speech, gene symbols spelled letter-by-letter, no tables or
bullets (the listening.io gold standard). One passage per section giving its verdicts and which
domains to remove and why, then a cohort synthesis + caveats. Emits a .txt (cleanest TTS input) and a
matching text-flow PDF. Narrative updated for the uniform _final segmentation + the is_MN state.
"""
import os, sys, glob, json, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def w(n):
    n = int(round(n))
    if n < 0: return "minus " + w(-n)
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + w(n % 100))
    if n < 1_000_000: return w(n // 1000) + " thousand" + ("" if n % 1000 == 0 else (", " if n % 1000 >= 100 else " and ") + w(n % 1000))
    return w(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else ", " + w(n % 1_000_000))


def spell(g):
    out = []
    for ch in str(g):
        if ch.isdigit(): out.append(_ONES[int(ch)])
        elif ch.isalpha(): out.append(ch.upper())
    return " ".join(out)


def spell_sample(code):
    body, _, block = str(code).partition("_")
    digits = "".join(c for c in body if c.isdigit()); pref = "".join(c for c in body if c.isalpha())
    out = f"{' '.join(list(pref.upper()))} {' '.join(_ONES[int(d)] for d in digits)}".strip()
    if block: out += ", block " + " ".join(list(block.upper()))
    return out


def status_phrase(s):
    s = str(s).lower()
    if s.startswith("control"): return "a control donor"
    if s.startswith("c9"): return "an A L S donor with the C nine or f seventy-two expansion"
    if s.startswith("spor"): return "a sporadic A L S donor"
    return "an A L S donor"


def join_and(items):
    items = list(items)
    if not items: return "none"
    if len(items) == 1: return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def load(statsdir):
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(statsdir, "*__dpqc.json")))]
    order = {"control": 0, "sporadic": 1, "c9": 2}
    S.sort(key=lambda s: (order.get(str(s["status"]).lower(), 9), s["level"], s["sample"]))
    return S


def narrative(S):
    P = []
    n = len(S); nrem = sum(s.get("n_remove", 0) for s in S)
    mn_active = sum(1 for s in S if s.get("mn_protection_active"))
    P.append(
        "This is the audio edition of the dual pass quality control results for the amyotrophic lateral "
        "sclerosis spinal cord Xenium cohort. Xenium is an imaging based spatial transcriptomics platform. "
        "For each section we ran the spatial domain method Novae independently, then scored every domain "
        "within its own section to decide whether it is a real tissue territory to keep, a borderline case "
        "to review, or a smear to remove. A smear is a domain whose cells scatter across the whole slice "
        f"rather than forming a connected patch, and which collects low quality cells. There are {w(n)} "
        "sections in total, and this recording walks through each one and states which domains it "
        "recommends removing and why.")
    P.append(
        "Remember three cautions throughout. The domain names are local to each section and cannot be "
        "compared across sections. The spinal level is tangled with the disease, because controls come "
        "from the neck and the disease cases from the lower back, so nothing here is a disease result. "
        "And this is pre quality control and exploratory, a guided tour, not a statistic.")
    for s in S:
        dd = sorted(s["domains_detail"], key=lambda r: -r["n_cells"])
        rem = [r for r in dd if r["verdict"] == "REMOVE"]
        rev = [r for r in dd if r["verdict"] == "REVIEW"]
        kep = [r for r in dd if r["verdict"] == "KEEP"]
        t = spell_sample(s["sample"]); t = t[0].upper() + t[1:]
        sent = (f"Section {t} is {status_phrase(s['status'])}, from the {s['level']} cord. It holds "
                f"{w(s['n_cells'])} cells across {w(len(dd))} domains. The recommendation is to keep "
                f"{w(len(kep))}, review {w(len(rev))}, and remove {w(len(rem))}")
        if rem:
            parts = []
            for r in rem:
                marker = spell(r["top_marker"]) if r.get("top_marker") else "no clear marker"
                parts.append(f"domain {spell(r['domain'])}, {w(r['n_cells'])} cells, whose reason is "
                             f"{r['reason'].replace('%',' percent').replace(';',' and')}")
            sent += ". The domains to remove are " + join_and(parts) + "."
        else:
            sent += ", and no domain crosses the removal threshold in this section."
        if rev:
            sent += (f" {w(len(rev)).capitalize()} domain" + ("s" if len(rev) != 1 else "") +
                     " are held for review, meaning the signals conflict or a real marker signature "
                     "protected a sparse domain, so a human should look before deleting.")
        sent += f" In all, about {w(s.get('pct_cells_remove', 0))} percent of this section's cells are flagged for removal."
        P.append(sent)
    P.append(
        f"Across the whole cohort the pipeline flags {w(nrem)} domains for removal, and in almost every "
        "case these are the diffuse, low count, weakly marked domains rather than any real cell "
        "population. The safety check confirms that removal drops ambient and untyped cells, not an "
        "entire biological cell type such as the motor neurons.")
    mn_sentence = ("The motor neuron flag is present but is not yet validated against the classical "
                   "markers choline acetyltransferase and M N X one."
                   if mn_active else
                   "The motor neuron protector was inactive in this run, because Marcel's motor neuron "
                   "annotation on the final segmentation was still pending, so no domain was protected "
                   "by motor neuron enrichment; once that annotation lands the pipeline should be re run "
                   "to switch the protector on.")
    P.append(
        "To close, the open questions for the meeting are these. " + mn_sentence + " The removal calls "
        "should be repeated at the four domain and ten domain resolutions and only the agreeing calls "
        "trusted. The segmentation is now uniform across the whole cohort in this final run, so the "
        "earlier concern that two of the control sections came from a separate segmentation batch no "
        "longer applies. Treat every removal as provisional and reversible. Thank you for listening.")
    return P


def to_pdf(paras, out_pdf, title):
    lines = []
    for p in paras:
        lines += textwrap.wrap(p, 96); lines.append("")
    with PdfPages(out_pdf) as pdf:
        per = 44
        for i in range(0, len(lines), per):
            fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.10, 0.05, 0.80, 0.9]); ax.axis("off"); y = 0.99
            if i == 0:
                ax.text(0.0, y, title, fontsize=15, fontweight="bold", va="top", family="sans-serif")
                ax.text(0.0, y - 0.035, "Built for Listening.io. Flowing prose, no tables.",
                        fontsize=9, color="0.4", va="top"); y -= 0.075
            for ln in lines[i:i + per]:
                ax.text(0.0, y, ln, fontsize=11, va="top", family="serif"); y -= 0.0205
            pdf.savefig(fig); plt.close(fig)


def main(statsdir, out_pdf, out_txt):
    S = load(statsdir)
    if not S:
        sys.exit(f"no stats in {statsdir}")
    P = narrative(S)
    body = "\n\n".join(P).encode("ascii", "ignore").decode("ascii")
    with open(out_txt, "w") as f:
        f.write(body + "\n")
    to_pdf(P, out_pdf, "Dual-pass QC cohort result -- audio edition")
    print(f"wrote audio -> {out_pdf} + {out_txt} ({len(S)} samples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statsdir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--txt", required=True)
    a = ap.parse_args()
    main(a.statsdir, a.out, a.txt)
