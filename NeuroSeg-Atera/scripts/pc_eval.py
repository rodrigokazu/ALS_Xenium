#!/usr/bin/env python3
"""Purkinje cells only: find every Purkinje soma and score how each segmentation handles it.

Purkinje genes: log2FC >= 3 and mean >= 1 in 10x cluster 33 (Purkinje) and mean <= 0.3 in the granule clusters, split at
random into set A (finds the somata) and set B (scores them), so the score does not reuse the genes that defined the soma.
Contaminant genes: granule = mean over granule clusters >= 0.5 and >= 10x the Purkinje mean (FAT2, CADPS2, GABRA6, CBLN3, ...); glia = log2FC >= 3, mean >= 0.5
in astro cluster 13/16/19 and mean <= 0.2 in 33.
Soma: Gaussian-smoothed (sigma 1.5 um) density of set-A transcripts above one threshold for all nine windows (Otsu on the
pooled log density of tissue pixels), >= 60 um2, opened by 1 um; soma region = soma dilated by 1 um.
Per soma and method: n_cells (cells holding >= 10% of the region's set-B transcripts), capture (share of set-B transcripts in
the single best cell), orphan (share unassigned), and the best cell's contamination (granule + glia genes / all its transcripts).
python pc_eval.py <out dir> [windows...]
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WINS = sys.argv[2:] or ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]
METHODS = {"10x": None, "old 18S (A500)": "crops_A500", "18S + density (C500)": "crops_C500", "PF": "crops_PF_500", "I2": "crops_I2_500", "PF + Purkinje repair": "crops_PCR_PF_500", "I2 + Purkinje repair": "crops_PCR_I2_500", "Hybrid A": "crops_HYBA_500", "Hybrid B": "crops_HYBB_500", "Hybrid C": "crops_HYBC_500"}
GRAN = [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11]; GLIA = [13, 16, 19]
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
mc = lambda k: de[f"Cluster {k} Mean Counts"]; lf = lambda k: de[f"Cluster {k} Log2 fold change"]
gmax = np.max(np.column_stack([mc(k) for k in GRAN]), 1)
pcg = de["Feature Name"][(lf(33) >= 3) & (mc(33) >= 1) & (gmax <= 0.3)].tolist()
rng = np.random.default_rng(0); perm = rng.permutation(pcg); setA, setB = set(perm[::2]), set(perm[1::2])
gmean = np.column_stack([mc(k) for k in GRAN]).mean(1)
gran = set(de["Feature Name"][(gmean >= 0.5) & ((gmean + 0.01) / (mc(33) + 0.01) >= 10)])
glia = set(de["Feature Name"][np.any(np.column_stack([(lf(k) >= 3) & (mc(k) >= 0.5) for k in GLIA]), 1) & (mc(33) <= 0.2)])
json.dump({"purkinje_A": sorted(setA), "purkinje_B": sorted(setB), "granule": sorted(gran), "glia": sorted(glia)}, open(OUT / "gene_sets.json", "w"), indent=1)
print(f"Purkinje genes {len(pcg)} (A {len(setA)}, B {len(setB)}), granule {len(gran)}, glia {len(glia)}", flush=True)

def window(win):
    st = json.load(open(RUNS / "crops_PF_500" / win / "stats.json")); x0, y0, x1, y1 = st["window"]
    bundle = SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas(strings_to_categorical=True)
    gc, gr = (tx.x_location.to_numpy() / UM).astype(int), (tx.y_location.to_numpy() / UM).astype(int)
    H, W = y1 - y0, x1 - x0; r, c = np.clip(gr - y0, 0, H - 1), np.clip(gc - x0, 0, W - 1)
    g = tx.feature_name.astype(str).to_numpy()
    def dmap(mask, s):
        a = np.bincount(r[mask] * W + c[mask], minlength=H * W).reshape(H, W).astype(np.float32)
        return ndi.gaussian_filter(a, s / UM) / UM ** 2
    return dict(win=win, x0=x0, y0=y0, H=H, W=W, r=r, c=c, gc=gc, gr=gr, genes=g, cid=tx.cell_id.astype(str).to_numpy(),
                isA=np.isin(g, list(setA)), isB=np.isin(g, list(setB)), isG=np.isin(g, list(gran)), isL=np.isin(g, list(glia)),
                dA=dmap(np.isin(g, list(setA)), 1.5), dall=dmap(np.ones(len(g), bool), 1.0))

D = {w: window(w) for w in WINS}
pool = np.concatenate([np.log1p(d["dA"][d["dall"] > 20]).ravel()[::7] for d in D.values()])
thr = float(np.expm1(threshold_otsu(pool))); print("soma threshold (set-A tx/um2)", round(thr, 3), flush=True)
rows, somas = [], []
for w, d in D.items():
    m = ndi.binary_opening((d["dA"] >= thr) & (d["dall"] > 20), iterations=int(1 / UM)); lab, n = ndi.label(m)
    area = ndi.sum(m, lab, np.arange(1, n + 1)) * UM ** 2; keep = np.flatnonzero(area >= 60) + 1
    soma = np.where(np.isin(lab, keep), lab, 0)
    reg = ndi.grey_dilation(soma, size=int(2 / UM) + 1)
    rid = reg[d["r"], d["c"]]
    labs = {"10x": np.where(np.isin(d["cid"], ["", "UNASSIGNED"]), "0", d["cid"])}
    for name, sub in METHODS.items():
        if sub is None: continue
        f = RUNS / sub / w / "final_labels.npy"
        if f.exists(): labs[name] = np.asarray(np.load(f, mmap_mode="r")[d["gr"], d["gc"]]).astype(np.int64).astype(str)
    for k in keep:
        inr = rid == k; b = inr & d["isB"]; nb = int(b.sum())
        cy, cx = ndi.center_of_mass(soma == k)
        somas.append(dict(window=w, soma=int(k), area_um2=float(area[k - 1]), cx_px=d["x0"] + cx, cy_px=d["y0"] + cy, n_setB=nb))
        if nb < 20: continue
        for name, lb in labs.items():
            cells = pd.Series(lb[b]); cells = cells[cells != "0"]; vc = cells.value_counts()
            best = vc.index[0] if len(vc) else None
            cont = np.nan
            if best is not None:
                inc = lb == best; cont = float((d["isG"][inc] | d["isL"][inc]).sum() / inc.sum())
            rows.append(dict(window=w, soma=int(k), method=name, n_setB=nb, n_cells=int((vc / nb >= 0.10).sum()),
                             capture=float(vc.iloc[0] / nb) if len(vc) else 0.0, orphan=float(1 - len(cells) / nb), contamination=cont))
    print(w, "somata", len(keep), flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "pc_scores.csv", index=False); pd.DataFrame(somas).to_csv(OUT / "pc_somata.csv", index=False)
pd.set_option("display.width", 200)
s = df.groupby("method").agg(somata=("soma", "size"), capture=("capture", "mean"), orphan=("orphan", "mean"), split_2plus=("n_cells", lambda v: (v >= 2).mean()),
                             none_10pct=("n_cells", lambda v: (v == 0).mean()), contamination=("contamination", "median"))
print(s.round(3).to_string())
