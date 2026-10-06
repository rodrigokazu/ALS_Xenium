#!/usr/bin/env python3
"""Coverage vs mixing for every method on the three Atera crops.

x: % of QC transcripts assigned to a cell (higher = more covered); y: MECR with cells cut to 1000 transcripts
(lower = cleaner cells). Ours come from <compare>/option_*/<spot>/metrics.json, the baselines and 10x from
<compare>/baselines_scores.csv (same splitmerge.metrics.score, same QC transcripts).
python plot_pareto.py <compare root>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = Path(sys.argv[1])
SPOTS = [("spot1_purkinje_left", "spot 1: Purkinje / granule edge"), ("spot2_gl_wm_centre", "spot 2: granule layer / white matter"),
         ("spot3_purkinje_right", "spot 3: Purkinje layer, right folium")]
OURS = [("option_A_18s", "A: 18S"), ("option_B_density", "B untuned"), ("option_Bgrow28", "grow 28, DAPI mask"),
        ("option_Bgrow77", "grow 77, DAPI mask"), ("option_Bt60", "Bt60"), ("option_Bt28", "Bt28"), ("option_Bt10", "Bt10")]
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"
base = pd.read_csv(C / "baselines_scores.csv")
rows = []
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), dpi=170)
for ax, (spot, title) in zip(axs, SPOTS):
    b = base[base.crop == spot]
    t = b[b.method == "xenium_original"].iloc[0]
    rows.append({"crop": spot, "method": "10x", "family": "10x", "coverage": t.pct_assigned, "mecr_n1000": t.mecr_n1000, "n_cells": t.n_cells, "median_tx": t.median_tx_per_cell})
    for _, r in b[b.method.isin(["voronoi", "watershed", "cellpose"])].iterrows():
        rows.append({"crop": spot, "method": r.method, "family": "baseline", "coverage": r.pct_assigned, "mecr_n1000": r.mecr_n1000, "n_cells": r.n_cells, "median_tx": r.median_tx_per_cell})
    for key, lab in OURS:
        f = C / key / spot / "metrics.json"
        if f.exists():
            s = json.load(open(f))["benchmark"]["new"]
            rows.append({"crop": spot, "method": lab, "family": "ours", "coverage": s["pct_assigned"], "mecr_n1000": s["mecr_n1000"], "n_cells": s["n_cells"], "median_tx": s["median_tx_per_cell"]})
    d = pd.DataFrame([r for r in rows if r["crop"] == spot])
    for fam, col, sz, z in (("baseline", GREY, 46, 2), ("ours", BLUE, 52, 3), ("10x", ORANGE, 90, 4)):
        s = d[d.family == fam]
        ax.scatter(s.coverage, s.mecr_n1000, s=sz, color=col, zorder=z, edgecolor="white", linewidth=0.8)
    curve = d[d.method.isin(["Bt60", "Bt28", "Bt10"])].sort_values("coverage")
    ax.plot(curve.coverage, curve.mecr_n1000, color=BLUE, lw=1.1, zorder=1, ls=(0, (3, 2)))
    placed = []
    for _, r in d.iterrows():
        dx, dy = (0.35, 0.0002)
        yo = 0.0
        for (px_, py_, yo_) in placed:   # stagger labels of near-identical points so none overprints another
            if abs(px_ - r.coverage) < 1.2 and abs(py_ - r.mecr_n1000) < 0.0004 and abs(yo_ - yo) < 1.2:
                yo += 2.6
        placed.append((r.coverage, r.mecr_n1000, yo))
        dy = -0.9 * yo if yo else 0.0
        dx = 0.35
        ax.annotate(r.method if r.family != "baseline" else r.method.capitalize(), (r.coverage, r.mecr_n1000), xytext=(dx, dy), textcoords="offset fontsize",
                    fontsize=7.5, color={"10x": ORANGE, "ours": BLUE, "baseline": "#52514e"}[r.family], fontweight="bold" if r.family == "10x" else "normal")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("transcripts assigned to a cell (%)  →  more covered")
    ax.grid(color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors="#898781", labelsize=8)
axs[0].set_ylabel("MECR at 1000 tx / cell  ↓  cleaner cells")
fig.suptitle("Atera cerebellum, three 400 µm crops: coverage against mixing (10x orange, baselines grey, ours blue; dashed = Bt density cutoff 60 → 28 → 10)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(C / "pareto_coverage_vs_mecr1000.png", facecolor="white")
df = pd.DataFrame(rows); df.to_csv(C / "scores_all_methods.csv", index=False)
pd.set_option("display.width", 200)
print(df[["crop", "method", "coverage", "mecr_n1000", "n_cells", "median_tx"]].to_string(index=False))
