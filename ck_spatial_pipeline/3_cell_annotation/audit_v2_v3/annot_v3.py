#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/annot_v3.py
#
# Tests three proposals and writes flags only. It marks junk sub-clusters 2, 9 and 10, scans
# a transcript floor of 10 to 25 counts on Other neurons, and asks whether MNX1+
# sub-cluster 7 is a motor-neuron pool, using ventral-horn location and a 50/100/250 um2
# area ladder. cell_type_v3 never reached the canonical object. Sub-cluster 7 was refused as
# an MN pool and the 15-count floor was reported and not applied.
# ========================================================================================

"""Annotation v3: drop the junk sub-clusters, raise the transcript floor on the
"Other neurons" class, and assess sub-cluster 7 (MNX1+) as a motor-neuron candidate
set -- ventral-horn location first, then a cell-area ladder (50/100/250 um2).

Reads the canonical NC v2 object + the audit sub-cluster table + cell_area from the
20 AUG5 per-sample h5ads. Writes annotation_v3/ (json, parquet, csvs). No h5ad is
modified: cells are FLAGGED, never deleted.

Usage: annot_v3.py
"""
import os, sys, json, numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.stats import fisher_exact

B    = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC   = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
AUG5  = f"{B}/SC_MNcorrected_AUG5_h5ads"
PASS2 = f"{B}/Ranger_procd"
AUD  = f"{B}/NicheCompass_SC/cell_audit"
OUT  = f"{AUD}/annotation_v3"
os.makedirs(OUT, exist_ok=True)

JUNK      = [2, 9, 10]          # the discard pile named by the audit
MNX1_SUB  = 7                   # the MNX1+ sub-cluster
THRESH    = [10, 15, 20, 25]    # transcript-count floor sweep ("~15-20")
AREAS     = [50, 100, 250]      # um2 ladder
EXCLUDED  = ["SD01015_BG", "SD01616_BI", "SD02913_BA"]   # no VH v2 mask -> in_VH_v2 is NA

def cat(f, col):
    g = f["obs"][col]
    c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    return np.array(c, dtype=object)[g["codes"][:]].astype(str)

def nbool(f, col):
    """nullable-boolean -> object array of True/False/None (mask True == missing)."""
    g = f["obs"][col]
    v = g["values"][:].astype(bool); m = g["mask"][:].astype(bool)
    o = np.empty(len(v), dtype=object); o[:] = v; o[m] = None
    return o

log = lambda *a: print(*a, flush=True)

# ------------------------------------------------------------------ load object
log("loading the canonical object")
f = h5py.File(NC, "r")
ns    = cat(f, "sample")
ctm   = cat(f, "cell_type_mn")
ctv2  = cat(f, "cell_type_v2")
niche = cat(f, "latent_leiden_0.3")
stat  = cat(f, "status")
vhsrc = cat(f, "vh_v2_source")
in_vh = nbool(f, "in_VH_v2")
is_mn = nbool(f, "is_MN")
obsnm = np.array([x.decode() for x in f["obs"][f["obs"].attrs["_index"]][:]], dtype=object)
spat  = f["obsm"]["spatial"][:]
var   = [x.decode() for x in f["var"][f["var"].attrs["_index"]][:]]
j_mnx1 = var.index("MNX1")

# counts layer, streamed in chunks: row totals, n genes, MNX1 column
g = f["layers"]["counts"]
indptr = g["indptr"][:]
n = len(ns)
tot = np.zeros(n, dtype=np.int64); ngenes = np.zeros(n, dtype=np.int32)
mnx1 = np.zeros(n, dtype=np.int32)
CH = 200_000
for s in range(0, n, CH):
    e = min(s + CH, n)
    a, b = int(indptr[s]), int(indptr[e])
    d  = g["data"][a:b]; ix = g["indices"][a:b]
    ip = indptr[s:e+1] - indptr[s]
    M = sp.csr_matrix((d, ix, ip), shape=(e - s, len(var)))
    tot[s:e]    = np.asarray(M.sum(1)).ravel()
    ngenes[s:e] = np.asarray((M > 0).sum(1)).ravel()
    mnx1[s:e]   = np.asarray(M[:, j_mnx1].todense()).ravel()
    log(f"  counts {e:,}/{n:,}")
f.close()
ON = ctm == "Other neurons"
log(f"  {n:,} cells | Other neurons (cell_type_mn) {int(ON.sum()):,} | residual v2 {int((ctv2=='Other neurons').sum()):,}")

# ------------------------------------------------- sub-cluster join + CONTENT GATE
on = pd.read_csv(f"{AUD}/otherneurons_cells.csv.gz")
assert len(on) == int(ON.sum()), f"row count {len(on)} != {ON.sum()}"
gate = {
    "sample": (on["sample"].values.astype(str) == ns[ON]).all(),
    "niche":  (on["niche"].values.astype(str)  == niche[ON]).all(),
    "counts": (on["counts"].values.astype(np.int64) == tot[ON]).all(),
    "ngenes": (on["ngenes"].values.astype(np.int64) == ngenes[ON]).all(),
}
log("  sub-cluster CONTENT GATE:", gate)
assert all(gate.values()), "sub-cluster join content gate FAILED"
sub = np.full(n, -1, dtype=np.int32)
sub[ON] = on["sub"].values.astype(np.int32)

# ------------------------------------ cell_area from pass2 (parent) + CONTENT GATE
# The NC object's obs_names are '<sample>|<pass2_obs_name>' -- pass2 is the PARENT
# universe. The AUG5 set is a DIFFERENT (smaller) segmentation universe: it is
# missing ~6 cells per section that pass2/NC contain, so it cannot supply cell_area
# for every cell. pass2 is used as the source; AUG5 is used as an INDEPENDENT
# cross-check on the cells the two universes share.
log("joining cell_area from the pass2 parent objects")
area  = np.full(n, np.nan)
areau = np.full(n, np.nan)
mnsig = np.full(n, np.nan)
a250  = np.zeros(n, dtype=bool)
base = np.array([o.split("|", 1)[1] for o in obsnm], dtype=object)
areagate = {}
for s_ in sorted(set(ns)):
    tag = s_.replace("_", "")
    p = f"{PASS2}/ALS_SCXenium_{tag}_pass2.h5ad"
    with h5py.File(p, "r") as a:
        anm = np.array([x.decode() for x in a["obs"][a["obs"].attrs["_index"]][:]], dtype=object)
        ca  = a["obs"]["cell_area"][:]
        cau = a["obs"]["cell_area_um2"][:]
        ax  = a["obs"]["x_centroid"][:]; ay = a["obs"]["y_centroid"][:]
        ms  = a["obs"]["mn_signature_prob"][:]
        g2  = a["obs"]["is_MN_a250"]
        a2  = (g2["values"][:].astype(bool) & ~g2["mask"][:].astype(bool)) if isinstance(g2, h5py.Group) else g2[:].astype(bool)
    pos = pd.Series(np.arange(len(anm)), index=anm)
    m = ns == s_
    idx = pos.reindex(base[m]).values
    assert not np.isnan(idx).any(), f"{s_}: unmatched cells in pass2"
    idx = idx.astype(int)
    area[m]  = ca[idx]; areau[m] = cau[idx]; mnsig[m] = ms[idx]; a250[m] = a2[idx]
    # CONTENT GATE: the NC spatial must be a RIGID per-sample offset of the pass2
    # centroids. A constant delta (sd ~ 0) proves the rows are the same cells;
    # a row-count or index check would not.
    dx = spat[m, 0] - ax[idx]; dy = spat[m, 1] - ay[idx]
    areagate[s_] = {"n": int(m.sum()), "dx_sd": float(dx.std()), "dy_sd": float(dy.std()),
                    "dx_mean": float(dx.mean()), "dy_mean": float(dy.mean())}
    assert dx.std() < 1e-6 and dy.std() < 1e-6, f"{s_}: coordinate gate FAILED {dx.std()} {dy.std()}"
    log(f"  {s_}: n={m.sum():,} offset=({dx.mean():.3f},{dy.mean():.3f}) sd=({dx.std():.2e},{dy.std():.2e})")
assert np.isfinite(area).all(), "cell_area has NaN after join"
log("  cell_area CONTENT GATE: all 20 samples rigid-offset, PASSED")
log(f"  cell_area vs cell_area_um2 max|delta| = {np.nanmax(np.abs(area - areau)):.3e}")

# independent cross-check against the AUG5 files Rodrigo named
log("cross-checking cell_area against the AUG5 universe")
xchk = {}
for s_ in sorted(set(ns)):
    p = f"{AUG5}/{s_}__MNcorrected_AUG5.h5ad"
    with h5py.File(p, "r") as a:
        anm = np.array([x.decode() for x in a["obs"][a["obs"].attrs["_index"]][:]], dtype=object)
        ca  = a["obs"]["cell_area"][:]
    pos = pd.Series(np.arange(len(anm)), index=anm)
    m = ns == s_
    idx = pos.reindex(base[m]).values
    ok = ~np.isnan(idx)
    d = np.abs(area[m][ok] - ca[idx[ok].astype(int)])
    xchk[s_] = {"shared": int(ok.sum()), "missing_from_AUG5": int((~ok).sum()),
                "max_abs_delta": float(d.max()) if ok.any() else None}
    log(f"  {s_}: shared {ok.sum():,} | absent from AUG5 {int((~ok).sum())} | max|delta| {d.max():.3e}")

# ------------------------------------------------------------------- the sweep
# Two readings of "this class". A = the ORIGINAL cell_type_mn class (101,188);
# B = the RESIDUAL class after annotation v2 moved 5 sub-clusters out (53,125).
res_ON = ctv2 == "Other neurons"
junk_m = np.isin(sub, JUNK)
sweep = []
for T in THRESH:
    low = (tot < T) & (sub >= 0)
    for tag, mask_class in (("A_original_class", ON), ("B_residual_v2_class", res_ON)):
        lo = low & mask_class
        dropped = junk_m | lo if tag == "A_original_class" else (junk_m & mask_class) | lo
        # cells the threshold would take out of classes v2 already RESCUED
        collateral = int((low & ~res_ON & ON).sum())
        sweep.append({
            "threshold": T, "reading": tag,
            "class_n": int(mask_class.sum()),
            "low_count_n": int(lo.sum()),
            "junk_n": int((junk_m & mask_class).sum()),
            "dropped_total": int(dropped.sum()),
            "dropped_pct_of_class": round(100 * dropped.sum() / mask_class.sum(), 2),
            "collateral_reassigned_classes": collateral,
            "residual_after": int((mask_class & ~dropped).sum()),
        })
sw = pd.DataFrame(sweep)
log("\n=== THRESHOLD SWEEP ===\n" + sw.to_string(index=False))

# per-sub breakdown at each threshold
persub = []
for T in THRESH:
    for s_ in range(12):
        m = sub == s_
        persub.append({"threshold": T, "sub": s_, "n": int(m.sum()),
                       "n_below": int((m & (tot < T)).sum()),
                       "pct_below": round(100 * (m & (tot < T)).sum() / max(m.sum(), 1), 1)})
ps = pd.DataFrame(persub)

# ------------------------------------------------- v3 label (reading B, T chosen later)
def build_v3(T):
    lab = ctv2.copy().astype(object)
    d_junk = junk_m & res_ON
    d_low  = res_ON & (tot < T) & ~d_junk
    lab[d_junk] = "Discard (junk sub-cluster)"
    lab[d_low]  = "Discard (low counts)"
    return lab, d_junk, d_low

# ------------------------------------------------------ MNX1 sub-cluster 7 tab
log("\n=== MNX1 SUB-CLUSTER 7 ===")
m7 = sub == MNX1_SUB
log(f"  n = {int(m7.sum()):,}")
evaluable = ~np.isin(ns, EXCLUDED)          # in_VH_v2 is NA in the 3 excluded sections
vh_true  = np.array([x is True  for x in in_vh])
vh_false = np.array([x is False for x in in_vh])
vh_na    = np.array([x is None  for x in in_vh])
ismn_true = np.array([x is True for x in is_mn])

# (a) VENTRAL HORN CHECK FIRST
a_in, a_out, a_na = int((m7 & vh_true).sum()), int((m7 & vh_false).sum()), int((m7 & vh_na).sum())
bg_in  = int((evaluable & vh_true).sum()); bg_n = int(evaluable.sum())
s7_ev  = int((m7 & evaluable).sum())
tab = [[a_in, s7_ev - a_in], [bg_in - a_in, (bg_n - s7_ev) - (bg_in - a_in)]]
orv, pv = fisher_exact(tab)
vh = {
    "n_sub7": int(m7.sum()), "in_VH": a_in, "outside_VH": a_out, "NA_excluded_sections": a_na,
    "n_evaluable": s7_ev,
    "pct_in_VH_of_evaluable": round(100 * a_in / max(s7_ev, 1), 2),
    "cohort_pct_in_VH": round(100 * bg_in / bg_n, 2),
    "odds_ratio": round(float(orv), 3), "fisher_p": float(pv),
    "contingency": tab,
}
log("  VH check:", json.dumps(vh, indent=2))

# per sample + per status
by_s = pd.DataFrame({
    "sample": ns[m7], "status": stat[m7], "vh_source": vhsrc[m7],
    "in_VH": [("NA" if x is None else bool(x)) for x in in_vh[m7]],
    "area": area[m7], "counts": tot[m7], "MNX1": mnx1[m7],
    "is_MN": [("NA" if x is None else bool(x)) for x in is_mn[m7]],
    "niche": niche[m7],
})
sam = by_s.groupby(["sample", "status", "vh_source"]).agg(
    n=("area", "size"),
    in_VH=("in_VH", lambda s: int((s == True).sum())),
    med_area=("area", "median"), med_counts=("counts", "median"), med_MNX1=("MNX1", "median"),
).reset_index()
sam["pct_in_VH"] = (100 * sam["in_VH"] / sam["n"]).round(1)
log("\n  per section:\n" + sam.to_string(index=False))

# (b) AREA LADDER -- applied AFTER the VH gate, and unconditionally, for contrast
ladder = []
for A in [0] + AREAS:
    m_area = m7 & (area >= A)
    m_vh   = m_area & vh_true
    ladder.append({
        "area_min_um2": A,
        "sub7_total": int(m_area.sum()),
        "sub7_in_VH": int(m_vh.sum()),
        "pct_of_sub7": round(100 * m_area.sum() / max(m7.sum(), 1), 1),
        "already_is_MN": int((m_vh & ismn_true).sum()),
        "new_vs_is_MN": int((m_vh & ~ismn_true).sum()),
        "also_is_MN_a250": int((m_vh & a250).sum()),
        "med_mn_signature_prob": round(float(np.nanmedian(mnsig[m_vh])), 4) if m_vh.any() else None,
        "med_area": round(float(np.median(area[m_area])), 1) if m_area.any() else None,
        "med_counts": int(np.median(tot[m_area])) if m_area.any() else None,
        "med_MNX1": float(np.median(mnx1[m_area])) if m_area.any() else None,
        "mean_MNX1": round(float(mnx1[m_area].mean()), 3) if m_area.any() else None,
    })
ld = pd.DataFrame(ladder)
log("\n  area ladder (VH-gated column is the candidate set):\n" + ld.to_string(index=False))

# reference: what the existing MN call looks like on the same axes
mn_ref = {
    "is_MN_true": int(ismn_true.sum()),
    "is_MN_med_area": round(float(np.median(area[ismn_true])), 1),
    "is_MN_med_counts": int(np.median(tot[ismn_true])),
    "is_MN_mean_MNX1": round(float(mnx1[ismn_true].mean()), 3),
    "is_MN_in_sub7": int((ismn_true & m7).sum()),
    "motor_neurons_class_n": int((ctv2 == "Motor neurons").sum()),
    "cohort_med_area": round(float(np.median(area)), 1),
    "sub7_med_area": round(float(np.median(area[m7])), 1),
    "sub7_mean_MNX1": round(float(mnx1[m7].mean()), 3),
    "cohort_mean_MNX1": round(float(mnx1.mean()), 3),
    "is_MN_a250_true": int(a250.sum()),
    "sub7_med_mn_signature_prob": round(float(np.nanmedian(mnsig[m7])), 4),
    "is_MN_med_mn_signature_prob": round(float(np.nanmedian(mnsig[ismn_true])), 4),
}
log("\n  reference:", json.dumps(mn_ref, indent=2))

# MNX1 expression by sub, to show sub 7 really is the MNX1 cluster
mx = pd.DataFrame({"sub": sub[ON], "MNX1": mnx1[ON], "area": area[ON], "counts": tot[ON]})
mxs = mx.groupby("sub").agg(n=("MNX1", "size"), mean_MNX1=("MNX1", "mean"),
                            pct_MNX1_pos=("MNX1", lambda s: 100 * (s > 0).mean()),
                            med_area=("area", "median"), med_counts=("counts", "median")).round(3).reset_index()
log("\n  MNX1 by sub:\n" + mxs.to_string(index=False))

# ------------------------------------------------------------------- write out
for T in THRESH:
    lab, dj, dl = build_v3(T)
    pd.DataFrame({"cell": np.arange(n), "sample": ns, "niche": niche,
                  "cell_type_v2": ctv2, "cell_type_v3": lab.astype(str),
                  "sub": sub, "counts": tot, "cell_area": area,
                  "MNX1": mnx1}).to_parquet(f"{OUT}/annotation_v3_T{T}.parquet", index=False)
sw.to_csv(f"{OUT}/threshold_sweep.csv", index=False)
ps.to_csv(f"{OUT}/threshold_by_sub.csv", index=False)
sam.to_csv(f"{OUT}/mnx1_sub7_by_sample.csv", index=False)
ld.to_csv(f"{OUT}/mnx1_sub7_area_ladder.csv", index=False)
mxs.to_csv(f"{OUT}/mnx1_by_sub.csv", index=False)
by_s.to_csv(f"{OUT}/mnx1_sub7_cells.csv.gz", index=False, compression="gzip")

# v3 class totals at each threshold
tot_tab = {}
for T in THRESH:
    lab, dj, dl = build_v3(T)
    tot_tab[T] = pd.Series(lab.astype(str)).value_counts().to_dict()

json.dump({
    "generated": pd.Timestamp.now().isoformat(),
    "object": NC, "n_cells": n,
    "junk_subs": JUNK, "mnx1_sub": MNX1_SUB,
    "content_gates": {"subcluster": {k: bool(v) for k, v in gate.items()},
                      "cell_area_rigid_offset": areagate,
                      "aug5_crosscheck": xchk},
    "class_sizes": {"cell_type_mn_other_neurons": int(ON.sum()),
                    "cell_type_v2_other_neurons": int(res_ON.sum())},
    "threshold_sweep": sweep,
    "threshold_by_sub": ps.to_dict("records"),
    "v3_totals_by_threshold": tot_tab,
    "mnx1_sub7": {"vh_check": vh, "area_ladder": ladder,
                  "by_sample": sam.to_dict("records"),
                  "reference": mn_ref, "mnx1_by_sub": mxs.to_dict("records")},
}, open(f"{OUT}/annotation_v3.json", "w"), indent=2, default=str)
log(f"\nwrote {OUT}")
