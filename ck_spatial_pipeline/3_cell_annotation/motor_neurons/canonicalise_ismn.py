#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/motor_neurons/canonicalise_ismn.py
#
# Makes is_MN_v2 the canonical is_MN (decision of 2026-09-02). The old 774-cell call moves
# to is_MN_v1 verbatim and is_MN becomes the 844-cell v2 call. Backups carry the suffix
# .bak-preCanon-20260902. Job 52471756, submitted with --wrap.
# ========================================================================================

"""
Make is_MN_v2 the canonical is_MN (Rodrigo's decision, 2026-09-02).

For each of the 20 pass2 h5ads and the NicheCompass latent:
  is_MN_v1 := current is_MN         (the superseded 774-cell v6 call, preserved verbatim)
  is_MN    := is_MN_v2              (844 cells; <NA> in SD01015BG / SD01616BI / SD02913BA)
  is_MN_v2 stays in place unchanged.
  uns['is_mn_canonical'] records the switch.

Idempotent: a file that already carries is_MN_v1 is skipped.
Safety: hardlink backup (.bak-preCanon-20260902) -> write tmp -> gate -> atomic swap.
Gates: n_obs, is_MN_v1 == old is_MN bit-for-bit (774 cohort), is_MN == is_MN_v2 bit-for-bit
(844 cohort), every other pre-existing column bit-identical, layers/obsm intact.
"""
import glob, os, re, sys, time
import numpy as np, pandas as pd, anndata as ad

BASE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
PASS2 = sorted(glob.glob(os.path.join(BASE, "Ranger_procd", "ALS_SCXenium_*_pass2.h5ad")))
LATENT = os.path.join(BASE, "NicheCompass_SC/artifacts/sample_integration/latest/model/"
                            "ALS_SC_cohort_NC_Leiden.h5ad")
BAKSUF = ".bak-preCanon-20260902"
EXPECT_OLD, EXPECT_NEW = 774, 844


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a); sys.stdout.flush()


def nbool(s):
    return pd.array(s, dtype="boolean")


def process(path, label):
    log("=" * 70); log(f"{label}: {os.path.basename(path)}")
    A = ad.read_h5ad(path)
    if "is_MN_v1" in A.obs.columns:
        log("  already canonicalised (is_MN_v1 present) -- skip")
        return {"label": label, "skipped": True}
    for c in ("is_MN", "is_MN_v2"):
        if c not in A.obs.columns:
            sys.exit(f"ABORT {label}: missing {c}")
    old = nbool(A.obs["is_MN"]); new = nbool(A.obs["is_MN_v2"])
    n_old, n_new = int(old.fillna(False).sum()), int(new.fillna(False).sum())
    na_old, na_new = int(old.isna().sum()), int(new.isna().sum())
    log(f"  is_MN(old) True={n_old} NA={na_old} | is_MN_v2 True={n_new} NA={na_new}")

    pre = {c: A.obs[c].copy() for c in A.obs.columns if c != "is_MN"}
    A.obs["is_MN_v1"] = pd.Series(old, index=A.obs_names, dtype="boolean")
    A.obs["is_MN"] = pd.Series(new, index=A.obs_names, dtype="boolean")
    A.uns["is_mn_canonical"] = {
        "marker": "IS_MN CANONICAL = is_MN_v2 (2026-09-02)",
        "decision": "Rodrigo 2026-09-02: adopt Marcel's is_MN_v2 (is_neuron_v3 & ~is_GAD & in_VH_v2) as is_MN",
        "is_MN_v1": "the superseded v6 delivered call (774 cells cohort-wide post-QC)",
        "na_policy": "<NA> where vh_v2_source=='excluded'",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    n = A.n_obs
    tmp = path + ".tmp-canon"
    if os.path.exists(tmp):
        os.remove(tmp)
    log("  writing tmp"); A.write_h5ad(tmp, compression="gzip"); del A

    log("  gating tmp")
    B = ad.read_h5ad(tmp)
    ok, why = True, ""
    v1 = nbool(B.obs["is_MN_v1"]); cur = nbool(B.obs["is_MN"]); v2 = nbool(B.obs["is_MN_v2"])
    if B.n_obs != n:
        ok, why = False, "n_obs drift"
    elif not (pd.Series(v1).equals(pd.Series(old))):
        ok, why = False, "is_MN_v1 != old is_MN"
    elif not (pd.Series(cur).equals(pd.Series(v2))):
        ok, why = False, "is_MN != is_MN_v2"
    elif int(cur.fillna(False).sum()) != n_new or int(cur.isna().sum()) != na_new:
        ok, why = False, "is_MN counts drift"
    else:
        for c, a0 in pre.items():
            b0 = B.obs[c]
            if str(a0.dtype) != str(b0.dtype) or not a0.reset_index(drop=True).equals(b0.reset_index(drop=True)):
                ok, why = False, f"pre-existing column drift: {c}"; break
        if ok and "spatial" not in B.obsm:
            ok, why = False, "obsm spatial lost"
        if ok and "counts" not in B.layers:
            ok, why = False, "layers counts lost"
    del B
    if not ok:
        os.remove(tmp); sys.exit(f"GATE FAILED {label}: {why} -- original untouched")
    bak = path + BAKSUF
    if not os.path.exists(bak):
        os.link(path, bak)
    os.replace(tmp, path)
    log(f"  SWAPPED (backup {os.path.basename(bak)})")
    return {"label": label, "skipped": False, "old": n_old, "old_na": na_old, "new": n_new, "new_na": na_new}


rows = [process(f, re.search(r"ALS_SCXenium_(.+)_pass2", f).group(1)) for f in PASS2]
rows.append(process(LATENT, "LATENT"))
t = pd.DataFrame(rows)
print("\n" + t.to_string(index=False))
p2 = t[(t.label != "LATENT") & (~t.skipped)]
if len(p2):
    print(f"\npass2 cohort: is_MN_v1 = {int(p2.old.sum())} (expect {EXPECT_OLD}) | is_MN = {int(p2.new.sum())} (expect {EXPECT_NEW})")
    assert int(p2.old.sum()) == EXPECT_OLD and int(p2.new.sum()) == EXPECT_NEW
lat = t[t.label == "LATENT"]
if len(lat) and not bool(lat.skipped.iloc[0]):
    assert int(lat.old.iloc[0]) == EXPECT_OLD and int(lat.new.iloc[0]) == EXPECT_NEW
print("\nCANONICALISATION COMPLETE")
