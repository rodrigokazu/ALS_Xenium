#!/usr/bin/env python3
"""Can the image of a nucleus tell the cell type that the transcripts assign, and the size of its cell?

Labels: transcript annotation (annotation/cells_annotated.parquet) of the final cell under each nucleus centroid; types with
>= 150 nuclei, "mixed" calls left out. Features: nucfeat.py (shape, DAPI, 18S / membrane rings, crowding), image only.
Cross-validation: 5 spatial folds = vertical strips of the section (quantiles of x), so test nuclei are never neighbours of
training nuclei. Model: histogram gradient boosting, balanced class weights. Reported: balanced accuracy, per-type recall and
precision, confusion, the same with feature groups alone, and for cell size (log area of the transcript cell) R2 within type.
python nucmodel.py
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, matplotlib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, r2_score
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
O = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full"); M = O / "nucleus_model"
f = pd.read_parquet(M / "nucleus_features.parquet"); a = pd.read_parquet(O / "annotation/cells_annotated.parquet")
a.index = a.index.astype(str); f["key"] = f.final_label.astype(str)
cells = pd.read_parquet(O / "cells.parquet"); area = dict(zip(cells.cell.astype(str), cells.area_um2))
d = f.merge(a[["cell_type"]], left_on="key", right_index=True); d = d[~d.cell_type.astype(str).str.startswith("mixed")]
d["cell_area"] = d.key.map(area); vc = d.cell_type.value_counts(); d = d[d.cell_type.isin(vc[vc >= 150].index)].reset_index(drop=True)
GROUPS = {"shape": ["area_um2", "perimeter_um", "eccentricity", "solidity", "major_um", "minor_um"],
          "DAPI": ["dapi_mean", "dapi_sd", "dapi_cv", "dapi_p10", "dapi_p50", "dapi_p90", "dapi_bright_frac", "dapi_edge_centre", "dapi_texture"],
          "18S / membrane rings": ["s18_nuc", "s18_ring1", "s18_ring2", "mem_nuc", "mem_ring1", "mem_ring2", "dapi_ring1"],
          "crowding": ["nn_dist_um", "n_within_10um"]}
ALL = sum(GROUPS.values(), []); y = d.cell_type.astype(str).to_numpy(); types = sorted(set(y), key=lambda t: -(y == t).sum())
fold = pd.qcut(d.x_um, 5, labels=False).to_numpy()
def cv(cols):
    pred = np.empty(len(d), dtype=object)
    for k in range(5):
        tr, te = fold != k, fold == k
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=0).fit(d.loc[tr, cols], y[tr])
        pred[te] = m.predict(d.loc[te, cols])
    return pred
res = {"n_nuclei": len(d), "types": {t: int((y == t).sum()) for t in types}}
pred = cv(ALL); res["balanced_accuracy_all"] = balanced_accuracy_score(y, pred)
for g, cols in GROUPS.items(): res[f"balanced_accuracy_{g}"] = balanced_accuracy_score(y, cv(cols))
pr, rc, _, _ = precision_recall_fscore_support(y, pred, labels=types, zero_division=0)
res["per_type"] = {t: {"recall": round(float(r), 3), "precision": round(float(p), 3)} for t, p, r in zip(types, pr, rc)}
gg = d.cell_type.isin(["Granule", "Golgi"]).to_numpy()
if gg.sum() and (y == "Golgi").any():
    sub = d[gg]; ys = y[gg]; fs = fold[gg]; pg = np.empty(len(sub), dtype=object)
    for k in range(5):
        m = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0).fit(sub.loc[fs != k, ALL], ys[fs != k]); pg[fs == k] = m.predict(sub.loc[fs == k, ALL])
    res["golgi_vs_granule_balanced_accuracy"] = balanced_accuracy_score(ys, pg)
# cell size from the nucleus image (log area of the transcript-defined cell), within each type
r2 = {}
for t in types:
    s = d[(y == t) & d.cell_area.notna()]
    if len(s) < 300: continue
    yt = np.log(s.cell_area.to_numpy()); pt = np.empty(len(s)); fk = pd.qcut(s.x_um, 5, labels=False).to_numpy()
    for k in range(5):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(s.loc[fk != k, ALL], yt[fk != k]); pt[fk == k] = m.predict(s.loc[fk == k, ALL])
    r2[t] = round(float(r2_score(yt, pt)), 3)
res["cell_size_r2_within_type"] = r2
json.dump(res, open(M / "nucleus_model.json", "w"), indent=1); print(json.dumps(res, indent=1))
# figure: confusion (row-normalised, recall) | balanced accuracy by feature group | size R2
cmx = confusion_matrix(y, pred, labels=types, normalize="true")
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), dpi=150, gridspec_kw=dict(width_ratios=[1.35, 0.8, 0.8], wspace=0.45))
im = axs[0].imshow(cmx, cmap="Blues", vmin=0, vmax=1)
axs[0].set_xticks(range(len(types)), types, rotation=60, ha="right"); axs[0].set_yticks(range(len(types)), [f"{t} ({(y == t).sum():,})" for t in types])
for i in range(len(types)):
    for j in range(len(types)):
        if cmx[i, j] >= 0.1: axs[0].text(j, i, f"{cmx[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmx[i, j] > 0.55 else "#0b0b0b")
axs[0].set_xlabel("predicted from the nucleus image"); axs[0].set_ylabel("type from transcripts")
axs[0].set_title(f"Nucleus image → cell type, spatial 5-fold CV\nbalanced accuracy {res['balanced_accuracy_all']:.2f} (chance {1 / len(types):.2f})", loc="left", fontsize=10.5)
plt.colorbar(im, ax=axs[0], fraction=0.04, pad=0.02, label="share of the row (recall)")
gk = ["all"] + list(GROUPS); vals = [res["balanced_accuracy_all"]] + [res[f"balanced_accuracy_{g}"] for g in GROUPS]
axs[1].barh(range(len(gk)), vals, color="#2a78d6", height=0.6); axs[1].axvline(1 / len(types), color="#898781", ls="--", lw=1)
axs[1].set_yticks(range(len(gk)), ["all features"] + list(GROUPS)); axs[1].invert_yaxis(); axs[1].set_xlim(0, 1)
for i, v in enumerate(vals): axs[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
axs[1].set_title("Which image features carry the type", loc="left", fontsize=10.5); axs[1].set_xlabel("balanced accuracy (dashed = chance)")
rk = list(r2); axs[2].barh(range(len(rk)), [r2[k] for k in rk], color="#2a78d6", height=0.6); axs[2].set_yticks(range(len(rk)), rk); axs[2].invert_yaxis()
axs[2].axvline(0, color="#0b0b0b", lw=0.8)
for i, k in enumerate(rk): axs[2].text(max(r2[k], 0) + 0.01, i, f"{r2[k]:.2f}", va="center", fontsize=9)
axs[2].set_title("Nucleus image → size of its cell\n(R², log area, within type)", loc="left", fontsize=10.5); axs[2].set_xlim(min(-0.05, min(r2.values()) - 0.02), 1)
fig.savefig(M / "nucleus_model.png", facecolor="white", bbox_inches="tight"); print("saved")
