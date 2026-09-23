#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/annot_v2.py
#
# Writes cell_type_v2 by moving five Other neurons sub-clusters into real classes: 1 to
# oligodendrocytes, 3 to microglia, 4 to astrocytes, 8 to endothelial and 5 to a new
# Fibroblast class. 48,063 cells move. Sub-clusters 0, 2, 6, 7, 9, 10 and 11 stay Other
# neurons. Every move rests on the audit's own evidence: profile correlation, the twelve-
# signature reference score and markers.
#
# Run directly with --write-h5ad, so there is no job id. It writes into the 20260902_ncv2
# object. The canonical v3 object picks cell_type_v2 up later through nc3_step2.
# ========================================================================================

"""Annotation v2: move the obvious "Other neurons" sub-clusters into real classes.

Reassignments (all five taken from the audit's own evidence in cell_audit.json --
profile correlation + the independent twelve-signature reference score + markers):

    sub 1  -> Oligodendrocytes   r=0.89, ref Oligodendrocyte 3,542   MYRF ERMN CNDP1
    sub 3  -> Microglia          r=0.91, ref Microglia      2,996    C1QC SYK NCKAP1L
    sub 4  -> Astrocytes         r=0.83, ref Astrocyte      2,849    FGFR3 PTPRZ1 CHI3L1
    sub 8  -> Endothelial        ref Endothelial            6,280    EPAS1 CAV1 COBLL1
    sub 5  -> Fibroblast   NEW   ref Fibroblast_VLMC        9,327    FBLN1 CEMIP SFRP2

Everything else in the class is left as "Other neurons" -- the nerve-root (6) and
ependymal (11) groups still need classes that do not exist, 2/9/10 are the discard
pile and 7 is unresolved, so none of them is an "obvious" move.

Writes obs['cell_type_v2'] to the canonical object; cell_type_mn is untouched.
Usage: annot_v2.py [--write-h5ad]
"""
import os, sys, json, numpy as np, pandas as pd, h5py

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC  = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
AUD = f"{B}/NicheCompass_SC/cell_audit"
OUT = f"{AUD}/annotation_v2"
os.makedirs(OUT, exist_ok=True)
WRITE = "--write-h5ad" in sys.argv

REASSIGN = {1:"Oligodendrocytes", 3:"Microglia", 4:"Astrocytes", 8:"Endothelial", 5:"Fibroblast"}
NEW_CLASSES = {"Fibroblast"}

def cat(f, col):
    g = f["obs"][col]
    c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    return np.array(c, dtype=object)[g["codes"][:]].astype(str)

print("loading the canonical object", flush=True)
f = h5py.File(NC, "r")
ns    = cat(f, "sample")
ctm   = cat(f, "cell_type_mn")
niche = cat(f, "latent_leiden_0.3")
stat  = cat(f, "status")
grp = f["layers"]["counts"]
import scipy.sparse as sp
Xc = sp.csr_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]),
                   shape=(len(ns), len(f["var"]["_index"])))
tot = np.asarray(Xc.sum(1)).ravel()
ngenes = np.asarray((Xc > 0).sum(1)).ravel()
f.close()
n = len(ns)
ON = ctm == "Other neurons"
print(f"  {n:,} cells | Other neurons {int(ON.sum()):,}", flush=True)

on = pd.read_csv(f"{AUD}/otherneurons_cells.csv.gz")
print(f"  sub-cluster table {on.shape}", flush=True)

# ---------------------------------------------------------------- CONTENT GATE
# the sub-cluster rows must be the ON rows of the object, in order. Row count alone
# proves nothing, so check the content of four independent columns.
assert len(on) == int(ON.sum()), "row count mismatch"
gate = {
  "sample":  bool((on["sample"].values == ns[ON]).all()),
  "niche":   bool((on["niche"].astype(str).values == niche[ON]).all()),
  "counts":  bool((on["counts"].values == tot[ON]).all()),
  "ngenes":  bool((on["ngenes"].values == ngenes[ON]).all()),
}
print("\n== CONTENT GATE (sub-cluster rows vs the object's Other-neuron rows) ==")
for k, v in gate.items():
    print(f"   {k:<8} {'MATCH' if v else 'MISMATCH'}")
assert all(gate.values()), "content gate failed - do not trust this join"

# ---------------------------------------------------------------- relabel
sub = on["sub"].astype(int).values
v2 = ctm.copy()
idx = np.where(ON)[0]
moved = {}
for s, lab in REASSIGN.items():
    m = sub == s
    v2[idx[m]] = lab
    moved[lab] = moved.get(lab, 0) + int(m.sum())
print("\n== MOVED ==")
for s in sorted(REASSIGN):
    print(f"   sub {s:<2} -> {REASSIGN[s]:<18} {int((sub==s).sum()):>7,}")
print(f"   total moved {sum(moved.values()):,} of {int(ON.sum()):,} "
      f"({100*sum(moved.values())/ON.sum():.1f}% of the class)")
print(f"   still 'Other neurons': {int((v2=='Other neurons').sum()):,}")

# ---------------------------------------------------------------- tables
CLASSES = ["Oligodendrocytes","Astrocytes","Other neurons","Endothelial","Fibroblast","OPCs",
           "Macrophages","Inhibitory neurons","Microglia","Excitatory neurons","Motor neurons"]
before = pd.Series(ctm).value_counts().reindex(CLASSES).fillna(0).astype(int)
after  = pd.Series(v2).value_counts().reindex(CLASSES).fillna(0).astype(int)
tab = pd.DataFrame({"v1": before, "v2": after})
tab["delta"] = tab.v2 - tab.v1
tab["pc_v1"] = 100*tab.v1/tab.v1.sum(); tab["pc_v2"] = 100*tab.v2/tab.v2.sum()
print("\n== CLASS TOTALS ==")
print(tab.to_string())

by_sample = pd.crosstab(ns, v2).reindex(columns=CLASSES).fillna(0).astype(int)
by_sample_pc = by_sample.div(by_sample.sum(1), axis=0).mul(100)
by_niche  = pd.crosstab(niche, v2).reindex(columns=CLASSES).fillna(0).astype(int)
by_niche_pc = by_niche.div(by_niche.sum(1), axis=0).mul(100)
# what the five moved groups did to each niche
on_by_niche_v1 = pd.Series(niche[ON]).value_counts().sort_index()

print("\n== PER-NICHE, v2 (%) ==")
print(by_niche_pc.round(1).to_string())

status_by_sample = pd.Series(stat, index=ns).groupby(level=0).first()

pd.DataFrame({"cell": np.arange(n), "sample": ns, "niche": niche,
              "cell_type_mn": ctm, "cell_type_v2": v2}).to_parquet(f"{OUT}/annotation_v2.parquet", index=False)
by_sample.to_csv(f"{OUT}/by_sample_counts.csv"); by_niche.to_csv(f"{OUT}/by_niche_counts.csv")
json.dump({"reassign": {str(k): v for k, v in REASSIGN.items()},
           "new_classes": sorted(NEW_CLASSES),
           "gate": gate,
           "n_cells": int(n), "n_other_v1": int(ON.sum()),
           "n_moved": int(sum(moved.values())),
           "n_other_v2": int((v2=="Other neurons").sum()),
           "moved_per_class": moved,
           "classes": CLASSES,
           "totals": tab.to_dict(),
           "by_sample_counts": by_sample.to_dict(),
           "by_sample_pc": by_sample_pc.round(3).to_dict(),
           "by_niche_counts": by_niche.to_dict(),
           "by_niche_pc": by_niche_pc.round(3).to_dict(),
           "sample_status": status_by_sample.to_dict(),
           "other_by_niche_v1": {str(k): int(v) for k, v in on_by_niche_v1.items()},
           }, open(f"{OUT}/annotation_v2.json","w"), indent=1)
print(f"\nwrote {OUT}/annotation_v2.json + .parquet + the two CSVs")

if WRITE:
    print("\nwriting obs['cell_type_v2'] into the canonical object", flush=True)
    with h5py.File(NC, "r+") as fh:
        o = fh["obs"]
        if "cell_type_v2" in o: del o["cell_type_v2"]
        cats = sorted(set(v2))
        codes = np.array([cats.index(x) for x in v2], np.int16)
        g = o.create_group("cell_type_v2")
        g.attrs["encoding-type"]="categorical"; g.attrs["encoding-version"]="0.2.0"; g.attrs["ordered"]=False
        g.create_dataset("codes", data=codes)
        ds = g.create_dataset("categories", data=np.array(cats, dtype=object),
                              dtype=h5py.special_dtype(vlen=str))
        ds.attrs["encoding-type"]="string-array"; ds.attrs["encoding-version"]="0.2.0"
        order = list(o.attrs["column-order"])
        if "cell_type_v2" not in order: order.append("cell_type_v2")
        o.attrs["column-order"] = np.array(order, dtype=object)
    import anndata as adm
    a = adm.read_h5ad(NC, backed="r")
    print("  readback:", a.obs["cell_type_v2"].value_counts().to_dict())
    a.file.close()
else:
    print("\n(dry run - pass --write-h5ad to add obs['cell_type_v2'])")
