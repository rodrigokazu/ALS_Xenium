"""Phase 1 segmentation (Atera cerebellum): original 18S rules for large neurons + boundary-stain watershed for the rest.

Stage 1: rules.segment_region on the standard 18S signal (unchanged rules and Params). Its 18S cells (keep18) are kept as
they are: large somata, including Purkinje cells without a visible nucleus.
Stage 2: every nucleus that has no 18S cell (orphan) is a seed of a watershed over the boundary stain (ATP1A1 / CD45 /
E-cadherin, whole-sample scaled): the membrane signal forms the ridges, so a cell stops where the stain says it ends instead
of at a fixed radius. Allowed region: free pixels (no 18S cell, no foreign nucleus) with smoothed transcript density above
tlow, within reach_um of a seed (neuron-sized nuclei, >= neuron_um2, get reach_neuron_um). The 3 um-capped Xenium fallback of
rules.add_fallback_cells is not used.
Returns the dict segment_region returns, with final / cells replaced, so run_fullsample's tile code is unchanged.
"""
import json, os
import numpy as np, pandas as pd, zarr
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from splitmerge import rules as R

V2 = json.loads(os.environ.get("ATERA_V2", "{}"))
DEF = dict(b_lo=2.0, b_hi=112.0, sigma_b_um=0.5, lam=0.03, tlow=10.0, sigma_d_um=1.0, reach_um=6.0, reach_neuron_um=12.0, neuron_um2=55.0)
P = {**DEF, **{k: v for k, v in V2.items() if k in DEF}}
_DENS = {}


def density_window(fx0, fy0, shape):
    z = _DENS.setdefault("z", zarr.open_array(V2["density"], mode="r"))
    H, W = shape
    return np.asarray(z[fy0:fy0 + H, fx0:fx0 + W]).astype(np.float32)


def segment_v2(crop, pct, nb, cb, tx, fx0, fy0, p, stats):
    rr = R.segment_region(crop, pct, nb, cb, tx, fx0, fy0, p, stats)
    keep18, nuc, owner_kept = rr["keep18"], rr["nuc"], rr["owner_kept"]
    n_nuc = len(rr["cell_ids"])
    um = R.UM_PER_PX
    orphans = np.array([k for k in range(1, n_nuc + 1) if k not in owner_kept], dtype=int)
    foreign = (nuc > 0) & ~np.isin(nuc, orphans)
    seeds = np.where(np.isin(nuc, orphans) & (keep18 == 0), nuc, 0)
    dens = ndi.gaussian_filter(density_window(fx0, fy0, nuc.shape), R.um_to_px(P["sigma_d_um"]))
    free = (keep18 == 0) & ~foreign & (dens > P["tlow"])
    b = np.clip((crop[1].astype(np.float32) - P["b_lo"]) / (P["b_hi"] - P["b_lo"]), 0, 1)
    b = ndi.gaussian_filter(b, R.um_to_px(P["sigma_b_um"]))
    dist = ndi.distance_transform_edt(seeds == 0) * um                                  # um to the nearest seed
    area_um2 = np.bincount(nuc.ravel(), minlength=n_nuc + 1) * um * um
    big = np.zeros(n_nuc + 1, bool); big[1:] = area_um2[1:] >= P["neuron_um2"]
    mask = (free & (dist <= P["reach_neuron_um"])) | (seeds > 0)
    ws = watershed(b + P["lam"] * dist, markers=seeds, mask=mask) if seeds.any() else seeds
    reach = np.where(big[np.clip(ws, 0, n_nuc)], P["reach_neuron_um"], P["reach_um"])
    ws = np.where((ws > 0) & ((dist <= reach) | (seeds == ws)), ws, 0)
    out = keep18.copy(); nxt = int(keep18.max()); new = {}
    for k, sl in enumerate(ndi.find_objects(ws, max_label=int(ws.max())), start=1):
        if sl is None:
            continue
        m = ws[sl] == k
        lab, _ = ndi.label(m); keep = np.unique(lab[(seeds[sl] == k) & m]); keep = keep[keep > 0]
        m = ndi.binary_fill_holes(np.isin(lab, keep)) & (out[sl] == 0) & ~foreign[sl] | ((seeds[sl] == k) & (out[sl] == 0))
        if not m.any():
            continue
        nxt += 1; out[sl][m] = nxt; new[nxt] = k
    # fix 1: restore the original step. A no-nucleus 18S piece overlapping an orphan's nucleus is that cell's own cytoplasm
    # or process, not a separate object (rules.merge_touching_pieces, as in segment_region), so it joins that cell.
    no_nuc_18s = set(np.unique(keep18[keep18 > 0])) - set(owner_kept.values())
    out, joined = R.merge_touching_pieces(out, no_nuc_18s, new, nuc)
    final_ids = np.unique(out[out > 0])
    a = np.bincount(out.ravel(), minlength=int(out.max()) + 1) * um * um
    cells = pd.DataFrame({"source": ["boundary_ws" if i in new else "18S" for i in final_ids],
                          "has_nucleus": [i in new or i in set(owner_kept.values()) for i in final_ids],
                          "xenium_cell_id": [rr["cell_ids"][new[i] - 1] if i in new else "" for i in final_ids],
                          "area_um2": a[final_ids].round(2), "n_transcripts": 0, "n_18s_pieces_joined": [sum(1 for v in joined.values() if v == i) for i in final_ids]},
                         index=pd.Index(final_ids, name="cell"))
    rr = dict(rr); rr.update(final=out, cells=cells, fallback={}, glia=new)
    return rr
