#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/motor_neurons/attach_latent_vhv2.py
#
# The same attach for the v1 NicheCompass latent object, so the niche by MN crosses could
# re-run on is_MN_v2. Latent obs_names are '<sample>|<pass2 obs_name>'. Submitted with
# --wrap as job 52471705.
# ========================================================================================

"""
Attach the VH-v2 flags to the NicheCompass latent object so niche_ismn_cross.py and
niche_wdr49_ismn_coloc.py can be re-run against is_MN_v2.

Latent obs_names are '<sample>|<pass2_obs_name>'; pass2 obs_names are bare '<sample>:<cell_id>'.
The mapping is a verified 1:1 bijection (1,097,669 rows), but we join on the KEY, never on order,
and then gate on content (per-sample True counts + is_MN still 774).

Safety: hardlink backup -> write tmp -> gate -> atomic swap.
"""
import glob, os, re, sys, time
import numpy as np
import pandas as pd
import anndata as ad

BASE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
PASS2 = os.path.join(BASE, "Ranger_procd")
LATENT = os.path.join(BASE, "NicheCompass_SC/artifacts/sample_integration/latest/model/"
                            "ALS_SC_cohort_NC_Leiden.h5ad")
BAKSUF = ".bak-preVHv2-20260902"
COLS = ["in_VH_v2", "is_MN_v2", "vh_v2_source", "has_vh_v2"]

# expected is_MN_v2 True per pass2 tag (from the attach report) -- the content gate
EXPECT = {
    "SD00614BG": 77, "SD01015BG": 0, "SD01115BG": 40, "SD01320BG": 65, "SD01413BA": 64,
    "SD01616BI": 0, "SD01620BI": 109, "SD01623BI": 53, "SD01915BG": 85, "SD01922BI": 15,
    "SD01923BI": 16, "SD02022BI": 73, "SD02622BI": 5, "SD02818BG": 41, "SD02913BA": 0,
    "SD03522BG": 47, "SD03614BG": 31, "SD03914BG": 65, "SD04219BI": 44, "SD05413BG": 14,
}
EXCLUDED_TAGS = {"SD01015BG", "SD01616BI", "SD02913BA"}
EXPECT_NA = 122902
EXPECT_TRUE = 844


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a)
    sys.stdout.flush()


# ------------------------------------------------- gather the flags from pass2
log("collecting VH-v2 flags from the 20 pass2 files")
frames = []
for f in sorted(glob.glob(os.path.join(PASS2, "ALS_SCXenium_*_pass2.h5ad"))):
    tag = re.search(r"ALS_SCXenium_(.+)_pass2\.h5ad", os.path.basename(f)).group(1)
    A = ad.read_h5ad(f, backed="r")
    o = A.obs
    for c in COLS:
        if c not in o.columns:
            sys.exit(f"ABORT: {tag} lacks {c} -- run attach_vhv2.py first")
    d = pd.DataFrame({
        "key": o.index.astype(str),
        "in_VH_v2": pd.array(o["in_VH_v2"], dtype="boolean"),
        "is_MN_v2": pd.array(o["is_MN_v2"], dtype="boolean"),
        "vh_v2_source": o["vh_v2_source"].astype(str).to_numpy(),
        "has_vh_v2": o["has_vh_v2"].astype(bool).to_numpy(),
    })
    got = int(d["is_MN_v2"].fillna(False).sum())
    if got != EXPECT[tag]:
        sys.exit(f"ABORT: {tag} is_MN_v2={got}, expected {EXPECT[tag]}")
    if tag in EXCLUDED_TAGS and int(d["is_MN_v2"].isna().sum()) != len(d):
        sys.exit(f"ABORT: {tag} is excluded but not fully <NA>")
    frames.append(d)
    A.file.close()
    log(f"  {tag}: n={len(d)} is_MN_v2={got}")

side = pd.concat(frames, ignore_index=True)
assert side["key"].is_unique, "duplicate pass2 obs_names across samples"
log(f"pass2 union: {len(side)} rows, is_MN_v2 True={int(side['is_MN_v2'].fillna(False).sum())}, "
    f"NA={int(side['is_MN_v2'].isna().sum())}")
if int(side["is_MN_v2"].fillna(False).sum()) != EXPECT_TRUE:
    sys.exit("ABORT: union True count wrong")
side = side.set_index("key")

# ----------------------------------------------------------- load the latent
log(f"reading latent {LATENT}")
L = ad.read_h5ad(LATENT)
n = L.n_obs
log(f"latent n_obs={n}")
if n != len(side):
    sys.exit(f"ABORT: latent n_obs {n} != pass2 union {len(side)}")

# key = everything after the first '|'
lk = pd.Index([s.split("|", 1)[1] if "|" in s else s for s in L.obs_names.astype(str)])
if not lk.is_unique:
    sys.exit("ABORT: derived latent keys are not unique")
miss = int((~lk.isin(side.index)).sum())
log(f"latent keys not found in pass2 union: {miss} (must be 0)")
if miss:
    sys.exit("ABORT: key join incomplete")

# JOIN BY KEY -- not by order
al = side.loc[lk]
pre_ismn = int(pd.array(L.obs["is_MN"], dtype="boolean").fillna(False).sum())
log(f"pre-existing latent is_MN True = {pre_ismn} (expect 774)")
if pre_ismn != 774:
    sys.exit("ABORT: latent is_MN is not the expected v1 set")

L.obs["in_VH_v2"] = pd.Series(al["in_VH_v2"].to_numpy(), index=L.obs_names, dtype="boolean")
L.obs["is_MN_v2"] = pd.Series(al["is_MN_v2"].to_numpy(), index=L.obs_names, dtype="boolean")
L.obs["vh_v2_source"] = pd.Categorical(al["vh_v2_source"].to_numpy(),
                                       categories=["manual", "robin", "excluded"])
L.obs["has_vh_v2"] = al["has_vh_v2"].to_numpy().astype(bool)
L.uns["vh_v2_attach"] = {
    "marker": "VH_V2 ATTACH 2026-09-02 (latent)",
    "recipe": "is_MN_v2 = is_neuron_v3 & ~is_GAD & in_VH_v2",
    "na_policy": "<NA> where vh_v2_source=='excluded' (SD01015BG, SD01616BI, SD02913BA)",
    "join": "latent obs_name '<sample>|<pass2_obs_name>' -> pass2 obs_name, joined by key",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

t = int(pd.array(L.obs["is_MN_v2"], dtype="boolean").fillna(False).sum())
na = int(pd.array(L.obs["is_MN_v2"], dtype="boolean").isna().sum())
log(f"attached: is_MN_v2 True={t} NA={na}")
if t != EXPECT_TRUE or na != EXPECT_NA:
    sys.exit(f"ABORT: expected {EXPECT_TRUE}/{EXPECT_NA}, got {t}/{na}")

# per-sample content gate on the latent itself
ps = (pd.DataFrame({"s": L.obs["sample"].astype(str).to_numpy(),
                    "v": pd.array(L.obs["is_MN_v2"], dtype="boolean").fillna(False).to_numpy()})
      .groupby("s")["v"].sum())
bad = []
for s, v in ps.items():
    tag = s.replace("_", "")
    if EXPECT.get(tag) != int(v):
        bad.append(f"{s}({tag}): {int(v)} != {EXPECT.get(tag)}")
if bad:
    sys.exit("ABORT per-sample gate: " + "; ".join(bad))
log("per-sample content gate PASSED on all 20")

# ------------------------------------------------- tmp -> gate -> atomic swap
tmp = LATENT + ".tmp-vhv2"
if os.path.exists(tmp):
    os.remove(tmp)
log(f"writing {os.path.basename(tmp)}")
L.write_h5ad(tmp, compression="gzip")
del L

log("gating tmp ...")
B = ad.read_h5ad(tmp, backed="r")
ok, why = True, ""
if B.n_obs != n:
    ok, why = False, f"n_obs {B.n_obs} != {n}"
elif not all(c in B.obs.columns for c in COLS):
    ok, why = False, "new columns missing"
elif int(pd.array(B.obs["is_MN_v2"], dtype="boolean").fillna(False).sum()) != EXPECT_TRUE:
    ok, why = False, "is_MN_v2 drifted"
elif int(pd.array(B.obs["is_MN"], dtype="boolean").fillna(False).sum()) != 774:
    ok, why = False, "is_MN (v1) drifted"
elif "latent_leiden_0.3" not in B.obs.columns:
    ok, why = False, "latent_leiden_0.3 lost"
elif B.obs["latent_leiden_0.3"].nunique() != 8:
    ok, why = False, "niche count != 8"
elif "nichecompass_latent" not in B.obsm:
    ok, why = False, "obsm['nichecompass_latent'] lost"
elif "counts" not in B.layers:
    ok, why = False, "layers['counts'] lost"
B.file.close()
if not ok:
    os.remove(tmp)
    sys.exit(f"GATE FAILED: {why} -- original untouched")

bak = LATENT + BAKSUF
if not os.path.exists(bak):
    os.link(LATENT, bak)
os.replace(tmp, LATENT)
log(f"SWAPPED ok (backup {os.path.basename(bak)})")
log("LATENT ATTACH COMPLETE")
