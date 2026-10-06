#!/usr/bin/env python3
"""Large Purkinje-marker blobs captured by one cell, on every window of the benchmark registries.
Blob = connected region where the sigma-3 um smoothed density of Purkinje-marker QC transcripts exceeds T per um^2 (area >= 150 um^2,
>= 40 marker transcripts). Per blob and method: share of the blob's marker transcripts in its largest single cell.
Test: exact McNemar on 'captured' (>= 70% in one cell) ours vs 10x, and paired Wilcoxon on the share.
python purkinje_blobs2.py <registry1.json> [<registry2.json> ...]  (env T=1.5)"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy import ndimage as ndi
from scipy.stats import wilcoxon, binomtest

UM = 0.2125; MARK = ["CALB1", "PCP2", "PCP4", "ITPR1", "CA8", "GRID2"]; T = float(os.environ.get("T", 1.5)); SIG = 3.0; MIN_UM2 = 150.0
rows = []
for rf in sys.argv[1:]:
    reg = json.load(open(rf))
    for wn, w in reg["windows"].items():
        x0, y0, x1, y1 = w["box"]; H, W = y1 - y0, x1 - x0
        t = ds.dataset(Path(w["bundle"]) / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "qv", "cell_id"],
            filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
            & pc.field("feature_name").isin(MARK) & (pc.field("qv") >= 20)).to_pandas()
        col = np.clip((t.x_location / UM).astype(int) - x0, 0, W - 1); row = np.clip((t.y_location / UM).astype(int) - y0, 0, H - 1)
        g = 4; cnt = np.zeros((H // g + 1, W // g + 1)); np.add.at(cnt, (row // g, col // g), 1)
        dens = ndi.gaussian_filter(cnt, SIG / (g * UM)) / (g * UM) ** 2
        lb, nb = ndi.label(dens > T); area = np.bincount(lb.ravel()) * (g * UM) ** 2
        keep = [i for i in range(1, nb + 1) if area[i] >= MIN_UM2]; blob = lb[row // g, col // g]
        t["blob"] = np.where(np.isin(blob, keep), blob, 0)
        lab = {}
        for mn, src in reg["methods"][wn].items():
            if src[0] == "10x":
                v = t.cell_id.astype(str).to_numpy(); lab[mn] = np.where(np.isin(v, ["UNASSIGNED", "-1", ""]), "", v)
            else:
                L = (np.load(src[1])["labels"] if src[0] == "npz" else np.load(src[1] + "/final_labels.npy", mmap_mode="r")[y0:y1, x0:x1] if src[0] == "run" else np.load(src[1]))
                lab[mn] = np.asarray(L)[row, col].astype(str); lab[mn] = np.where(lab[mn] == "0", "", lab[mn])
        for i in keep:
            s = (t.blob == i).to_numpy()
            if s.sum() < 40: continue
            rec = {"window": wn, "area_um2": round(area[i]), "marker_tx": int(s.sum())}
            for mn, v in lab.items():
                vs = v[s]; asg = vs[vs != ""]; top = pd.Series(asg).value_counts().iloc[0] / s.sum() if len(asg) else 0.0
                rec[mn] = round(float(top), 3)
            rows.append(rec)
df = pd.DataFrame(rows); df.to_csv("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare/purkinje_blobs_final.csv", index=False)
print(f"{len(df)} blobs on {df.window.nunique()} windows (density > {T}/um2 at sigma {SIG} um, area >= {MIN_UM2:.0f} um2)")
ms = [c for c in df.columns if c not in ("window", "area_um2", "marker_tx")]
for m in ms:
    d = df[m]; print(f"  {m:22s} captured by one cell (>=70%): {int((d >= .7).sum()):2d}/{len(d)} | median share in largest cell {d.median():.2f}")
if "ours" in df and "10x" in df:
    a, b = df["ours"] >= .7, df["10x"] >= .7; n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    print(f"ours vs 10x, captured: ours only {n10}, 10x only {n01}, exact McNemar p = {p:.4f}")
    dd = (df["ours"] - df["10x"]); print(f"share in largest cell, ours − 10x: median {dd.median():+.2f}; ours higher in {int((dd > 0).sum())}/{len(dd)} blobs; Wilcoxon p = {wilcoxon(dd).pvalue:.4f}")
