"""Purkinje soma cores, shared by pc_repair.py (windows) and full_post.py (full sample).

core candidates = set-A Purkinje density >= thr in tissue, opened by 1 um, >= 60 um2. A candidate is a soma only when
  - it has an 18S body: median normalised 18S inside >= MIN18 (dendrite strips carry Purkinje transcripts but little 18S),
  - it is compact: major / minor axis <= MAX_ELONG,
and a candidate with >= 2 Purkinje density peaks >= 12 um apart (each >= 2 thr) is split along the density valley, one soma
per peak. Cut-offs were set on the spot1 + spot2 windows only (2026-10-02).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

UM = 0.2125; MIN18, MAX_ELONG, PEAK_SEP_UM = 0.2, 3.0, 12.0

def soma_cores(dA, dall, s18, thr):
    """Label image of accepted soma cores (0 = none) and a stats dict."""
    cand = ndi.binary_opening((dA >= thr) & (dall > 20), iterations=int(1 / UM)); cl, _ = ndi.label(cand)
    out = np.zeros(cl.shape, np.int32); nxt = 1; st = dict(candidates=0, low18=0, elongated=0, split=0)
    for p in regionprops(cl):
        if p.area * UM ** 2 < 60: continue
        st["candidates"] += 1; sl = p.slice; m = cl[sl] == p.label
        if float(np.median(s18[sl][m])) < MIN18: st["low18"] += 1; continue
        if p.major_axis_length / max(p.minor_axis_length, 1) > MAX_ELONG: st["elongated"] += 1; continue
        d = np.where(m, dA[sl], 0)
        pk = peak_local_max(ndi.gaussian_filter(d, 2 / UM), min_distance=int(PEAK_SEP_UM / UM), threshold_abs=2 * thr, labels=m.astype(int))
        if len(pk) >= 2:
            mk = np.zeros(m.shape, np.int32); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
            parts = watershed(-d, mk, mask=m); st["split"] += 1
            for q in range(1, len(pk) + 1):
                pm = parts == q
                if pm.sum() * UM ** 2 >= 60: out[sl][pm] = nxt; nxt += 1
        else:
            out[sl][m] = nxt; nxt += 1
    return out, st
