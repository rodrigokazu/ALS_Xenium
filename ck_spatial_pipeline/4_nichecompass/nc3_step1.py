#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/nc3_step1.py
#
# Builds one exclusion list from everything the audits condemned: niches 3, 7 and 8 of
# latent_leiden_0.3 and junk sub-clusters 2, 9 and 10 from otherneurons_cells.csv.gz. That
# drops 132,384 cells and keeps 965,285. The join carries a content gate on sample and niche.
# It also stamps the v1 numbering contract into niche_numbering.json. Job 52575211.
# ========================================================================================

"""Step 1 of the niche-3 removal + retrain.

Builds ONE exclusion list covering everything the audits have already condemned,
writes the filtered object, and stamps the v1 numbering contract.

WHAT IS EXCLUDED (and why)
  niche v1_3          110,910  non-cord / nerve root. Bench: 26.6% detached, depth 190 um,
                               0.14% VH, 1.6% astrocyte, 0 MN. Already excluded from the v2
                               DE run (niches 0,1,2,4,5).
  junk subs 2/9/10     23,585  annotation v3 'Discard (junk sub-cluster)'. Only 2,140 of these
                               sit inside niche 3, so dropping niche 3 alone would leave
                               21,445 behind -- 10,779 of them inside niche 0, which is one of
                               the niches being re-clustered. Dropped by IDENTITY, not by a
                               transcript floor.
  niche v1_7 + v1_8        29  single-section specks; Bench verdict 'exclude'.

WHAT IS DELIBERATELY KEPT
  niche v1_6 (4,219)   real tissue identity (meninges/pia) that fails the SAMPLE-SPREAD gate,
                       not a quality failure. Under-sampling is a reason to distrust
                       cross-sample claims, not a reason to delete cells before training.
                       Flagged, not dropped.
  MNX1 sub-7 (2,712)   its own flagged tier by explicit reviewer decision; never a discard.
  the <15-count floor  the v3 verdict was 'object-wide, reported alongside the annotation,
                       NOT baked in' (it would strip up to 66% of the astrocyte rescue).

NUMBERING CONTRACT
  niche_v1      'v1_0' ... 'v1_8'   the ORIGINAL run = the Niche Annotation Bench numbering
  niche_v1_call the Bench's anatomical call
  niche_v2      'v2_0' ... 'v2_k'   the retrain (written by step 4)
  sub-clusters  'v2_2.1', 'v2_2.2'  = retrain niche 2, sub-cluster 1, 2 ...
A bare integer niche label is never written again. Cf. niche_numbering_per_run.
"""
import os, json, numpy as np, pandas as pd, h5py, scanpy as sc

B    = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC   = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
AUD  = f"{B}/NicheCompass_SC/cell_audit"
OUT  = f"{B}/NicheCompass_SC/runs_v3_noN3"
os.makedirs(OUT, exist_ok=True)
DROP_NICHES = ["3", "7", "8"]
JUNK_SUBS   = [2, 9, 10]
log = lambda *a: print(*a, flush=True)

BENCH_CALLS = {
 "0":"Pial surface and vasculature (medium-high)",
 "1":"Grey matter - dorsal and ventral horns (very high)",
 "2":"Deep white matter / mixed glial parenchyma (LOW - under-resolved)",
 "3":"Nerve root / non-cord tissue (high that it is NOT cord)",
 "4":"Myelinated white matter tract (very high)",
 "5":"Central canal / ependyma (very high)",
 "6":"Meninges / pia - ONE section only, sampling fails",
 "7":"not tissue - single-section speck (24 cells)",
 "8":"not tissue - single-section speck (5 cells)",
}

def cat(f,c):
    g=f["obs"][c]; cc=[x.decode() if isinstance(x,bytes) else str(x) for x in g["categories"][:]]
    return np.array(cc,dtype=object)[g["codes"][:]].astype(str)

log("reading obs from the canonical object")
with h5py.File(NC,"r") as f:
    niche=cat(f,"latent_leiden_0.3"); ns=cat(f,"sample"); ctm=cat(f,"cell_type_mn")
    obsnm=np.array([x.decode() for x in f["obs"][f["obs"].attrs["_index"]][:]],dtype=object)
n=len(niche)

# ---- the junk sub-clusters, joined through the audit CSV with a CONTENT GATE
on = pd.read_csv(f"{AUD}/otherneurons_cells.csv.gz")
ON = ctm == "Other neurons"
assert len(on) == int(ON.sum()), f"otherneurons CSV {len(on)} != {ON.sum()}"
assert (on["sample"].values.astype(str) == ns[ON]).all(), "sample content gate FAILED"
assert (on["niche"].values.astype(str) == niche[ON]).all(), "niche content gate FAILED"
sub = np.full(n, -1, dtype=int); sub[ON] = on["sub"].values.astype(int)

m_niche = np.isin(niche, DROP_NICHES)
m_junk  = np.isin(sub, JUNK_SUBS)
drop    = m_niche | m_junk
reason  = np.where(m_niche, "niche v1_" + pd.Series(niche).where(m_niche, "").values,
                   np.where(m_junk, "junk sub-cluster", "retained"))
reason  = np.where(m_niche, np.array(["niche v1_"+x for x in niche], dtype=object),
           np.where(m_junk, "junk sub-cluster", "retained"))

log(f"  {n:,} cells")
log(f"  niche {DROP_NICHES}: {int(m_niche.sum()):,}")
log(f"  junk subs {JUNK_SUBS}: {int(m_junk.sum()):,} "
    f"({int((m_junk & ~m_niche).sum()):,} outside the dropped niches)")
log(f"  UNION dropped: {int(drop.sum()):,} ({100*drop.mean():.2f}%) | "
    f"RETAINED {int((~drop).sum()):,}")
log("\n  junk removed per retained niche:")
for k in sorted(set(niche)):
    m = niche == k
    if m.sum() < 100 or k in DROP_NICHES: continue
    log(f"    v1_{k}: {int((m_junk&m).sum()):>7,} / {int(m.sum()):>9,} "
        f"({100*(m_junk&m).sum()/m.sum():.2f}%)")

# ---- exclusion list keyed by TAG, holding PASS2 obs_names -------------------
base = np.array([o.split("|",1)[1] for o in obsnm], dtype=object)
excl, counts = {}, {}
for s_ in sorted(set(ns)):
    m = (ns == s_) & drop
    tag = s_.replace("_","")
    excl[tag] = sorted(base[m].tolist()); counts[tag] = int(m.sum())
    log(f"  {tag}: exclude {int(m.sum()):,} / {int((ns==s_).sum()):,}")
payload = {"source_object": NC, "source_column": "latent_leiden_0.3",
           "dropped_niches": DROP_NICHES, "junk_subs": JUNK_SUBS,
           "bench_calls": {k: BENCH_CALLS[k] for k in DROP_NICHES},
           "kept_deliberately": {
             "niche v1_6": "real identity, fails sample-spread gate only — flagged not dropped",
             "MNX1 sub-7": "own flagged tier by reviewer decision",
             "<15 count floor": "v3 verdict: object-wide, reported alongside, NOT baked in"},
           "n_dropped_total": int(drop.sum()), "n_retained_total": int((~drop).sum()),
           "per_tag_n": counts, "per_tag_obs_names": excl}
p=f"{OUT}/exclude_niche3_and_junk.json"
with open(p,"w") as fh: json.dump(payload, fh)
log(f"\nwrote {p} ({os.path.getsize(p)/1e6:.1f} MB)")

# ---- filtered object + the v1 numbering contract ---------------------------
log("reading the full object (slow)")
a = sc.read_h5ad(NC)
assert a.n_obs == n
a.obs["niche_v1"] = pd.Categorical(["v1_"+x for x in niche],
                                   categories=[f"v1_{k}" for k in sorted(BENCH_CALLS)])
a.obs["niche_v1_call"] = pd.Categorical([BENCH_CALLS[x] for x in niche])
a.obs["dropped_v3"] = pd.Categorical(np.where(drop, reason, "retained").astype(str))
a.obs["junk_sub"] = pd.Categorical(np.where(m_junk, "junk", "no"))
log("  dropped_v3:\n" + a.obs["dropped_v3"].value_counts().to_string())

full=f"{OUT}/NC_v2_Leiden_nichev1tagged.h5ad"
a.write_h5ad(full, compression="gzip")
log(f"wrote {full} ({os.path.getsize(full)/1e9:.2f} GB)")

b=a[~drop].copy()
log(f"  filtered: {b.n_obs:,} cells")
filt=f"{OUT}/NC_v3_filtered_noNiche3_noJunk.h5ad"
b.write_h5ad(filt, compression="gzip")
log(f"wrote {filt} ({os.path.getsize(filt)/1e9:.2f} GB)")

json.dump({"numbering_contract":{
   "niche_v1":"v1_0..v1_8 — the ORIGINAL 20260902_ncv2 run; the Niche Annotation Bench numbering",
   "niche_v2":"v2_0..v2_k — the 20260914_ncv3 retrain without niche v1_3/7/8 and the junk subs",
   "sub_clusters":"v2_<k>.<j> — e.g. v2_2.1 is sub-cluster 1 of retrain niche 2",
   "rule":"a bare integer niche label is never written again"},
   "bench_calls_v1":BENCH_CALLS, "dropped":payload["n_dropped_total"],
   "dropped_detail":{"niches":DROP_NICHES,"junk_subs":JUNK_SUBS},
   "kept_deliberately":payload["kept_deliberately"],
   "n_retained":payload["n_retained_total"]},
  open(f"{OUT}/niche_numbering.json","w"), indent=2)
log("wrote niche_numbering.json")
