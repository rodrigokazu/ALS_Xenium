#!/usr/bin/env python3
"""Stage 2b of PLAN.md: our own nucleus detection from DAPI plus a transcript nuclear score.

Transcript nuclear score: every gene gets a shrunken log-odds weight w_g = n/(n+500) * (logit(f_g) - logit(f_all)), where f_g
is the fraction of that gene's in-cell transcripts that lie in a 10x nucleus (learned on the spot1 window only). The score
map is the Gaussian-smoothed (sigma 1 um) mean weight of the transcripts around each point, so it does not depend on DAPI.
Pixel model: logistic regression on (DAPI normalised by whole-sample percentiles, transcript score, transcript density),
fitted on spot1 pixels against the 10x nucleus mask, then frozen. Nuclei: probability >= 0.5, split by distance-transform
peaks (>= 2 um apart), 8 to 250 um2. Output per window: own nucleus labels on the padded window grid, matched against 10x.
python nuclei_detect.py <window> [<window> ...]
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); OUT = SSD / "compare/nuclei"; OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
PADUM, BIN = 6.0, 0.5; PCT_DAPI = (3.0, 321.0); TRAIN = "spot1_purkinje_left"
rng = np.random.default_rng(0)

def load(win):
    bundle = SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / win / "stats.json"))["window"]
    pad = int(round(PADUM / UM)); X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad; H, W = Y1 - Y0, X1 - X0
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < X1 * UM) & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < Y1 * UM)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas(strings_to_categorical=True)
    tf = tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif"); z = zarr.open(tf.aszarr(level=0, series=0), mode="r")
    dapi = ndi.gaussian_filter(np.asarray(z[0, Y0:Y1, X0:X1]).astype(np.float32), 0.5 / UM)
    nb = ds.dataset(bundle / "nucleus_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 15) & (pc.field("vertex_x") < X1 * UM + 15)
                                                                     & (pc.field("vertex_y") >= Y0 * UM - 15) & (pc.field("vertex_y") < Y1 * UM + 15)).to_pandas()
    nuc = np.zeros((H, W), np.int32)
    for k, (_, g) in enumerate(nb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(H, W)); nuc[rr, cc] = k
    return dict(win=win, tx=tx, dapi=dapi, nuc10x=nuc, X0=X0, Y0=Y0, H=H, W=W, pad=pad, window=(x0, y0, x1, y1))

def gene_weights(tx):
    t = tx[tx.cell_id.astype(str).str.len() > 2]
    g = t.groupby("feature_name", observed=True).overlaps_nucleus.agg(["sum", "count"])
    fa = t.overlaps_nucleus.mean(); lo = np.log((g["sum"] + 1) / (g["count"] - g["sum"] + 1)) - np.log(fa / (1 - fa))
    return (lo * g["count"] / (g["count"] + 500)).to_dict()

def score_map(d, w):
    tx = d["tx"]; xs, ys = tx.x_location.to_numpy() / UM - d["X0"], tx.y_location.to_numpy() / UM - d["Y0"]
    f = BIN / UM; nby, nbx = int(d["H"] / f) + 1, int(d["W"] / f) + 1
    xb, yb = np.clip((xs / f).astype(int), 0, nbx - 1), np.clip((ys / f).astype(int), 0, nby - 1); flat = yb * nbx + xb
    wv = tx.feature_name.map(w).astype(float).fillna(0).to_numpy()
    s = ndi.gaussian_filter(np.bincount(flat, weights=wv, minlength=nby * nbx).reshape(nby, nbx), 1.0 / BIN)
    n = ndi.gaussian_filter(np.bincount(flat, minlength=nby * nbx).reshape(nby, nbx).astype(float), 1.0 / BIN)
    def up(a): return ndi.zoom(a, f, order=1)[:d["H"], :d["W"]]
    sc, dn = up(s / np.maximum(n, 1e-3)), up(n / BIN ** 2)
    pad = ((0, d["H"] - sc.shape[0]), (0, d["W"] - sc.shape[1]))
    return np.pad(sc, pad, mode="edge"), np.pad(dn, pad, mode="edge")

def features(d, w):
    sc, dn = score_map(d, w)
    dp = np.clip((d["dapi"] - PCT_DAPI[0]) / (PCT_DAPI[1] - PCT_DAPI[0]), 0, 1.5)
    return np.stack([dp, sc, np.log1p(dn)], -1)

def detect(prob):
    m = ndi.binary_opening(prob >= 0.5, iterations=2); m = ndi.binary_fill_holes(m)
    dt = ndi.distance_transform_edt(m)
    pk = peak_local_max(dt, min_distance=int(2.0 / UM), labels=ndi.label(m)[0], exclude_border=False)
    mk = np.zeros(m.shape, np.int32); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dt, mk, mask=m)
    a = np.bincount(lab.ravel()) * UM ** 2; bad = (a < 8) | (a > 250); bad[0] = False
    lab[bad[lab]] = 0
    out, _ = ndi.label(lab > 0) if False else (lab, None)
    return out

def match(own, ten):
    m = (own > 0) & (ten > 0)
    pairs = pd.DataFrame({"o": own[m], "t": ten[m]}).value_counts().rename("inter").reset_index()
    ao = np.bincount(own.ravel()); at = np.bincount(ten.ravel())
    pairs["iou"] = pairs.inter / (ao[pairs.o] + at[pairs.t] - pairs.inter)
    best = pairs.sort_values("iou", ascending=False).drop_duplicates("o").drop_duplicates("t")
    hit = best[best.iou >= 0.5]
    no = len(np.unique(own[own > 0])); nt = len(np.unique(ten[ten > 0]))
    own_any = np.bincount(own[ten == 0].ravel(), minlength=len(ao)) / np.maximum(ao, 1)
    new = [i for i in np.unique(own[own > 0]) if own_any[i] > 0.8]
    return dict(own=no, tenx=nt, matched_iou50=len(hit), own_new_no_10x_overlap=len(new),
                tenx_missed=int(nt - len(set(pairs[pairs.iou >= 0.2].t))), median_iou_matched=float(hit.iou.median()) if len(hit) else None), new

if __name__ == "__main__":
    wins = sys.argv[1:]
    d = load(TRAIN); w = gene_weights(d["tx"]); pickle.dump(w, open(OUT / "gene_weights.pkl", "wb"))
    F = features(d, w); y = (d["nuc10x"] > 0)
    tis = F[..., 2] > np.log1p(20)
    idx = rng.choice(np.flatnonzero(tis.ravel()), 400_000, replace=False)
    X = F.reshape(-1, 3)[idx]; Y = y.ravel()[idx]
    clf = LogisticRegression(max_iter=500).fit(X, Y); pickle.dump(clf, open(OUT / "pixel_model.pkl", "wb"))
    print("model coef (dapi, tx score, log density)", clf.coef_.round(3), clf.intercept_.round(3))
    rep = {}
    for win in wins:
        dd = d if win == TRAIN else load(win)
        Fw = F if win == TRAIN else features(dd, w)
        prob = clf.predict_proba(Fw.reshape(-1, 3))[:, 1].reshape(dd["H"], dd["W"]).astype(np.float32)
        yy = dd["nuc10x"] > 0; ti = Fw[..., 2] > np.log1p(20); ii = rng.choice(np.flatnonzero(ti.ravel()), 300_000, replace=False)
        auc = {k: float(roc_auc_score(yy.ravel()[ii], v)) for k, v in (("dapi", Fw.reshape(-1, 3)[ii, 0]), ("tx_score", Fw.reshape(-1, 3)[ii, 1]), ("combined", prob.ravel()[ii]))}
        own = detect(prob)
        mt, new = match(own, dd["nuc10x"])
        np.savez_compressed(OUT / f"{win}_nuclei.npz", own=own, prob=(prob * 255).astype(np.uint8), origin=np.array([dd["X0"], dd["Y0"]]), new=np.array(new))
        rep[win] = dict(auc_vs_10x_nucleus_mask=auc, **mt)
        print(win, json.dumps(rep[win]), flush=True)
    json.dump(rep, open(OUT / "nuclei_report.json", "w"), indent=1)
