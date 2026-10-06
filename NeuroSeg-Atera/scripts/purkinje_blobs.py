#!/usr/bin/env python3
"""Do the methods capture large Purkinje-marker blobs as one cell?
Blobs = connected regions where the smoothed density of Purkinje-marker transcripts (CALB1 PCP2 PCP4 ITPR1 CA8 GRID2, QC) exceeds T.
Per blob and per method: fraction of the blob's marker transcripts in its largest single cell, fraction unassigned, number of
cells holding >= 10%. Methods: 10x (transcript cell_id), and each label image given. python purkinje_blobs.py <compare root>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy import ndimage as ndi

UM = 0.2125; C = Path(sys.argv[1]); B = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/work/local_bundle")
MARK = ["CALB1", "PCP2", "PCP4", "ITPR1", "CA8", "GRID2"]
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5; SIG = 3.0; MIN_UM2 = 150.0          # marker transcripts per um^2 after sigma 3 um smoothing; min blob area
METHODS = {"original 18S": "option_A500", "combined": "option_C500", "phase 1 (2/6)": "option_V2d_500", "phase 2": "option_P2_500", "phase 2 held-out genes": "option_P2held_500", "phase 2 held-out strict": "option_P2heldstrict_500"}
out = []
for spot in ("spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"):
    z = np.load(C / "option_C500" / spot / "crop.npz"); x0, y0, x1, y1 = z["window"]; H, W = z["labels"].shape
    t = ds.dataset(B / "transcripts.parquet").to_table(
        columns=["x_location", "y_location", "feature_name", "qv", "cell_id"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
        & pc.field("feature_name").isin(MARK) & (pc.field("qv") >= 20)).to_pandas()
    col = np.clip((t.x_location / UM).astype(int) - x0, 0, W - 1); row = np.clip((t.y_location / UM).astype(int) - y0, 0, H - 1)
    # 1 um grid density
    g = 4                                    # px per bin (0.85 um)
    cnt = np.zeros((H // g + 1, W // g + 1)); np.add.at(cnt, (row // g, col // g), 1)
    dens = ndi.gaussian_filter(cnt, SIG / (g * UM)) / (g * UM) ** 2
    lab_b, nb = ndi.label(dens > T)
    area = np.bincount(lab_b.ravel()) * (g * UM) ** 2
    keep = [i for i in range(1, nb + 1) if area[i] >= MIN_UM2]
    blob_of = lab_b[row // g, col // g]
    t["blob"] = np.where(np.isin(blob_of, keep), blob_of, 0)
    tenx = t.cell_id.astype(str).replace({"UNASSIGNED": "", "-1": ""})
    labs = {}
    for name, d in METHODS.items():
        f = C / d / spot / "crop.npz"
        if f.exists(): labs[name] = np.load(f)["labels"][row, col]
    for i in keep:
        s = t.blob == i
        if s.sum() < 40: continue
        rec = {"window": spot[:5], "blob": i, "area_um2": round(area[i]), "marker_tx": int(s.sum()), "x_um": round(float(t.x_location[s].mean())), "y_um": round(float(t.y_location[s].mean()))}
        for name, v in (("10x", tenx[s].to_numpy()), *[(k, l[s.to_numpy()]) for k, l in labs.items()]):
            un = (v == "") if v.dtype == object else (v == 0)
            ids = pd.Series(v[~un]); vc = ids.value_counts()
            rec[f"{name}|top"] = round(vc.iloc[0] / s.sum(), 2) if len(vc) else 0.0
            rec[f"{name}|none"] = round(float(un.mean()), 2)
            rec[f"{name}|cells"] = int((vc / s.sum() >= 0.10).sum())
        out.append(rec)
df = pd.DataFrame(out); df.to_csv(C / f"purkinje_blobs_T{T:g}.csv", index=False)
print(f"{len(df)} blobs (>= {MIN_UM2:.0f} um2, density > {T}/um2 at sigma {SIG} um, >= 40 marker transcripts)")
names = ["10x"] + list(METHODS)
for w, d in df.groupby("window"):
    print(f"\n{w}: {len(d)} blobs")
    for n in names:
        if f"{n}|top" not in d: continue
        print(f"  {n:14s} captured by ONE cell (>=70%): {int((d[f'{n}|top'] >= .7).sum()):2d} | split (<40% in any cell): {int((d[f'{n}|top'] < .4).sum()):2d} | mostly unassigned (>50%): {int((d[f'{n}|none'] > .5).sum()):2d} | median in largest cell: {d[f'{n}|top'].median():.2f}")
