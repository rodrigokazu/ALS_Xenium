#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/gene_coloc_compare_fig_FINAL.py
#
# Puts MNX1, BCL6 and STMN2 side by side across all sections, reading the three stats CSVs
# written by gene_coloc_hero_FINAL and producing a three-panel distribution figure with one
# point per section, plus a merged summary of medians.
#
# Two corrections over the _gausss version, both about not lying with defaults. The low
# quality section list is no longer hardcoded to the gauss-era SD01620_BI call; it has to be
# passed with --lowq and defaults to none with a loud reminder, because the degraded set on
# _final must come from the _final dual-pass QC rather than from memory. And n in the titles
# is counted from the data instead of being written as a literal.
#
# It also detects the is_MN-pending state. If every section reports n_MN == 0 the comparison
# is meaningless, and the script refuses rather than drawing three identical empty
# distributions.
# ========================================================================================

"""gene_coloc_compare_fig_FINAL.py -- compare MNX1 vs BCL6 vs STMN2 motor-neuron
co-localisation across all _final sections. Reads the three *_MN_colocalisation_stats.csv
(written by gene_coloc_hero_FINAL.py), builds a 3-panel distribution figure (each point =
one section) and a merged summary CSV of medians.

_final deltas vs gene_coloc_compare_fig.py:
  * LOWQ (sections shown hollow + excluded from medians) is NO LONGER hardcoded to the
    gauss-era SD01620_BI (4,388-cell) call. On _final the degraded set must come from the
    _final dual-pass QC (Novae_persample_INDEPENDENT_FINAL/dualpass_qc). Pass it via
    --lowq SD..,SD.. ; default = none, with a loud reminder.
  * n in titles/prints is computed from the data, not the literal "n=20 / n=19".
  * is_MN GATE: if the stats are is_MN-pending (every section n_MN==0, i.e. the annotation
    had not landed when the stats were generated), the comparison is meaningless -> the
    script prints a clear pending message and writes nothing.
No bundle/parquet access -- pure CSV plotting. Env: xenium_vistools (or any with mpl)."""
import os, sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GENES = ["MNX1", "BCL6", "STMN2"]
GC = {"MNX1": "#00C2A8", "BCL6": "#FFB000", "STMN2": "#E45756"}


def load(d):
    frames = {}
    for g in GENES:
        p = os.path.join(d, f"{g}_MN_colocalisation_stats.csv")
        if not os.path.exists(p):
            p = os.path.join(d, f"{g.lower()}_MN_colocalisation_stats.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing stats CSV for {g} in {d} (run gene_coloc_hero_FINAL.py first)")
        frames[g] = pd.read_csv(p).set_index("sample")
    return frames


def _is_pending(F):
    """True if the stats reflect is_MN not-yet-available (all sections zero MN)."""
    for g in GENES:
        df = F[g]
        col = "n_MN" if "n_MN" in df.columns else None
        if col is None or df[col].sum() > 0:
            return False
    return True


def main(d, out, lowq=None):
    lowq = set(lowq or [])
    F = load(d)
    if _is_pending(F):
        print("[compare] stats are is_MN-PENDING (every section has n_MN==0) -> the "
              "MNX1/BCL6/STMN2 MN co-localisation comparison is is_MN-gated and cannot be "
              "drawn yet. Re-run gene_coloc_hero_FINAL.py after Marcel's _final MN annotation "
              "lands, then re-run this. Nothing written."); return
    if not lowq:
        print("[compare] NOTE: --lowq empty -> no section excluded from medians. Set --lowq "
              "from the _final dual-pass QC REMOVE/REVIEW list before using medians in a "
              "figure (the gauss-era SD01620_BI exclusion does NOT auto-carry to _final).", flush=True)

    metrics = [
        ("frac_MN_pos", "% of motor neurons positive", 100.0, False),
        ("enrich_MN_vs_nonMN", "enrichment (per-cell MN vs non-MN)", 1.0, True),
        ("mean_per_MN", "mean transcripts per motor neuron", 1.0, True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    rows = []
    n_keep_ref = None
    for ax, (col, label, mult, logy) in zip(axes, metrics):
        for i, g in enumerate(GENES):
            s = F[g][col].astype(float) * mult
            keep = s[[ix not in lowq for ix in s.index]]
            drop = s[[ix in lowq for ix in s.index]]
            n_keep_ref = len(keep)
            x = np.full(len(keep), i) + (np.linspace(-0.16, 0.16, len(keep)))
            ax.scatter(x, keep.values, s=34, color=GC[g], alpha=0.8, edgecolors="black", linewidths=0.4, zorder=3)
            if len(drop):
                ax.scatter(np.full(len(drop), i), drop.values, s=34, facecolors="none", edgecolors=GC[g], linewidths=1.1, zorder=3)
            med = float(np.median(keep.values)) if len(keep) else float("nan")
            ax.plot([i-0.28, i+0.28], [med, med], color=GC[g], lw=3, zorder=4)
            ax.annotate(f"{med:.1f}" if not logy else f"{med:.1f}x" if col != "mean_per_MN" else f"{med:.2f}",
                        (i, med), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=10, fontweight="bold", color=GC[g])
            rows.append(dict(gene=g, metric=col, median=med,
                             min=float(keep.min()) if len(keep) else float("nan"),
                             max=float(keep.max()) if len(keep) else float("nan")))
        if logy: ax.set_yscale("log")
        ax.set_xticks(range(len(GENES))); ax.set_xticklabels(GENES, fontsize=12)
        ax.set_title(label, fontsize=12); ax.grid(axis="y", alpha=0.25)
        if col == "enrich_MN_vs_nonMN":
            ax.axhline(1.0, color="0.5", ls="--", lw=1); ax.set_ylabel("fold (log scale)")
    n_total = len(F[GENES[0]])
    excl = f"; hollow = excluded ({sorted(lowq)})" if lowq else "; no sections excluded"
    fig.suptitle("Motor-neuron co-localisation: canonical MN markers (MNX1, STMN2) vs BCL6\n"
                 f"each point = one section (n={n_total}{excl})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ("png", "pdf"):
        fig.savefig(f"{out}/gene_MN_coloc_comparison.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{out}/gene_MN_coloc_comparison_summary.csv", index=False)
    print(f"=== median across sections (n_kept={n_keep_ref}, excluded={sorted(lowq)}) ===")
    piv = summ.pivot(index="gene", columns="metric", values="median").reindex(GENES)
    print(piv.to_string())
    print(f"\nwrote {out}/gene_MN_coloc_comparison.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("d", help="dir with the *_MN_colocalisation_stats.csv files")
    ap.add_argument("out", nargs="?", default=None, help="output dir (default = d)")
    ap.add_argument("--lowq", default="", help="comma-sep sample labels to show hollow + exclude from medians")
    a = ap.parse_args()
    out = a.out or a.d
    lowq = [s for s in a.lowq.split(",") if s.strip()]
    main(a.d, out, lowq=lowq)
