#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/motor_neurons/attach_vhv2.py
#
# Attaches Marcel's ventral-horn v2 drop (2026-09-02) to the 20 pass2 h5ads: in_VH_v2,
# is_MN_v2, vh_v2_source and has_vh_v2. The recipe is is_MN_v2 = is_neuron_v3 & ~is_GAD &
# in_VH_v2, reproduced with 0 mismatches over 2,063,633 rows. There is no area gate, grey
# matter is not required and interneurons are not excluded. Three sections carry NA:
# SD01015_BG, SD01616_BI and SD02913_BA.
#
# SAMPLE_MAP holds five naming traps between Marcel's table and ours. The coordinate gate
# and the decoy gate both have to pass before anything writes.
# ========================================================================================

"""
Attach Marcel's vh_correction_v2 flags (2026-09-02) to the 20 QCed pass2 h5ads.

New obs columns
  in_VH_v2      nullable boolean  -- NA where vh_v2_source == 'excluded'
  is_MN_v2      nullable boolean  -- NA where vh_v2_source == 'excluded'
  vh_v2_source  categorical       -- manual / robin / excluded
  has_vh_v2     bool              -- source != 'excluded'

Recipe (reverse-engineered, 0/2,063,633 mismatches):
  is_MN_v2 == is_neuron_v3 & ~is_GAD & in_VH_v2
  (the v4 >=250um2 area gate is GONE; in_GM is NOT required; interneurons NOT excluded)

Safety: hardlink backup (zero cost, survives the rename) -> write tmp -> GATE tmp -> atomic swap.
A file whose gates fail is left untouched and reported.

Usage: attach_vhv2.py [--dry-run] [--samples TAG,TAG]
"""
import argparse, json, os, re, sys, glob, shutil, time
import numpy as np
import pandas as pd
import anndata as ad

BASE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
PASS2 = os.path.join(BASE, "Ranger_procd")
VHV2 = os.path.join(BASE, "Ranger_procd_mw_final/vh_correction_v2")
PARQ = os.path.join(VHV2, "neuron_classification_tdp_v6_WITH_VH_V2.parquet")
SIDECAR = os.path.join(BASE, "astro_WDR49/vh_correction_v2_sidecars")
BAKSUF = ".bak-preVHv2-20260902"
MARKER = "VH_V2 ATTACH 2026-09-02"

# obs['sample'] (our pass2)  ->  parquet 'sample'.  The 5 non-identity entries are
# the documented name traps; every one is re-verified by a coordinate gate below.
SAMPLE_MAP = {
    "SD00614_BG": "SD00614_BG_",
    "SD01015_BG": "SD01015_BG",
    "SD01115_BG": "SD01115_BG",
    "SD01320_BG": "SD01320_BG",
    "SD01413_BA": "SD01413_BA",
    "SD01616_BI": "SD01616_BI",
    "SD01620_BI": "SD016_20_BI",   # trap: SD01620_BI in parquet is the superseded original
    "SD01623_BI": "SD01623_BI",
    "SD01915_BG": "SD01915_BG",
    "SD01922_BI": "SD01922_BI",
    "SD01923_BI": "SD01923_BI",
    "SD02022_BI": "SD020_22_BI",   # trap
    "SD02622_BI": "SD02622_BI",
    "SD02818_BG": "SD02818_BG_",
    "SD02913_BA": "SD02913_BA",
    "SD03522_BG": "SD03522_BG",
    "SD03614_BG": "SD03614_BG",
    "SD03914_BG": "SD03914_BG",
    "SD04219_BI": "SD04219_BI",
    "SD05413_BG": "Region_2",      # trap
}
COORD_TOL_UM = 1e-6
NEWCOLS = ["in_VH_v2", "is_MN_v2", "vh_v2_source", "has_vh_v2"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a)
    sys.stdout.flush()


def nb(s):
    """-> pandas nullable boolean array."""
    return pd.array(s, dtype="boolean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--samples", default=None)
    args = ap.parse_args()

    os.makedirs(SIDECAR, exist_ok=True)
    log("reading parquet", PARQ)
    big = pd.read_parquet(PARQ, columns=[
        "sample", "cell_index", "x_centroid", "y_centroid",
        "in_VH_v2", "is_MN_v2", "vh_v2_source", "is_neuron_v3", "is_GAD"])
    log(f"parquet {big.shape}")

    # global recipe re-check on the full table
    rec = (big["is_neuron_v3"].to_numpy(bool)
           & ~big["is_GAD"].to_numpy(bool)
           & big["in_VH_v2"].to_numpy(bool))
    mm = int((rec != big["is_MN_v2"].to_numpy(bool)).sum())
    log(f"GATE recipe on full parquet: mismatches={mm} (must be 0)")
    if mm:
        sys.exit("ABORT: is_MN_v2 recipe does not hold")

    files = sorted(glob.glob(os.path.join(PASS2, "ALS_SCXenium_*_pass2.h5ad")))
    if args.samples:
        keep = set(args.samples.split(","))
        files = [f for f in files if re.search(r"ALS_SCXenium_(.+)_pass2", f).group(1) in keep]
    log(f"{len(files)} pass2 files targeted")

    by_samp = {s: g for s, g in big.groupby("sample", sort=False)}
    report, failures = [], []

    for f in files:
        tag = re.search(r"ALS_SCXenium_(.+)_pass2\.h5ad", os.path.basename(f)).group(1)
        log("=" * 70)
        log(f"{tag}: reading {os.path.basename(f)}")
        A = ad.read_h5ad(f)
        n = A.n_obs
        our_samp = str(A.obs["sample"].iloc[0])
        assert A.obs["sample"].nunique() == 1, f"{tag}: multiple samples in one file"
        psamp = SAMPLE_MAP[our_samp]
        g = by_samp[psamp]

        # ---- join key: obs_names '<sample>:<cell_id>', cell_index = cell_id - 1
        cid = np.array([int(x.rsplit(":", 1)[1]) for x in A.obs_names], dtype=np.int64)
        ci = cid - 1
        gi = g.set_index("cell_index")
        missing = int((~pd.Index(ci).isin(gi.index)).sum())
        log(f"  our_sample={our_samp} -> parquet={psamp} | n_obs={n} | unmatched cell_index={missing}")
        if missing:
            failures.append((tag, f"{missing} cell_index not in parquet"))
            continue
        sub = gi.loc[ci]

        # ---- GATE 1: coordinate content gate (true match ~0; decoys ~1e4 um)
        sp = np.asarray(A.obsm["spatial"], dtype=float)
        d = np.abs(sp - sub[["x_centroid", "y_centroid"]].to_numpy(float)).max()
        log(f"  GATE1 coord max|delta| = {d:.3e} um (tol {COORD_TOL_UM})")
        if not np.isfinite(d) or d > COORD_TOL_UM:
            failures.append((tag, f"coordinate gate failed: {d:.4g} um"))
            continue

        # ---- GATE 2: decoy check -- no other parquet sample may match this well
        best_decoy, best_name = np.inf, None
        rs = np.random.default_rng(0).choice(n, size=min(3000, n), replace=False)
        for dn, dg in by_samp.items():
            if dn == psamp:
                continue
            dgi = dg.set_index("cell_index")
            common = pd.Index(ci[rs]).intersection(dgi.index)
            if len(common) < 100:
                continue
            pos = pd.Index(ci[rs]).get_indexer(common)
            dd = np.abs(sp[rs][pos] - dgi.loc[common, ["x_centroid", "y_centroid"]].to_numpy(float)).max()
            if dd < best_decoy:
                best_decoy, best_name = dd, dn
        log(f"  GATE2 best decoy = {best_name} at {best_decoy:.1f} um (must be >> tol)")
        if best_decoy < 1.0:
            failures.append((tag, f"decoy {best_name} matches at {best_decoy:.4g} um"))
            continue

        # ---- build the columns
        src = sub["vh_v2_source"].astype(str).to_numpy()
        excluded = src == "excluded"
        vh2 = nb(sub["in_VH_v2"].to_numpy(bool))
        mn2 = nb(sub["is_MN_v2"].to_numpy(bool))
        # THE NA LANDMINE: Marcel ships plain bool with zero NA, so an excluded
        # sample reads as "definitively 0 MNs". Restore NA = "not annotated".
        vh2[excluded] = pd.NA
        mn2[excluded] = pd.NA

        # ---- GATE 3: recipe holds on this file's own obs columns
        isn = pd.Series(A.obs["is_neuron_v3"]).astype("boolean").fillna(False).to_numpy()
        gad = pd.Series(A.obs["is_GAD"]).astype("boolean").fillna(False).to_numpy()
        rec_l = isn & ~gad & pd.Series(vh2).astype("boolean").fillna(False).to_numpy()
        mn_l = pd.Series(mn2).astype("boolean").fillna(False).to_numpy()
        mm_l = int((rec_l != mn_l).sum())
        log(f"  GATE3 local recipe mismatches = {mm_l} (must be 0)")
        if mm_l:
            failures.append((tag, f"local recipe mismatch {mm_l}"))
            continue

        # ---- GATE 4: is_MN_v2 subset of in_VH_v2
        bad = int((mn_l & ~pd.Series(vh2).astype("boolean").fillna(False).to_numpy()).sum())
        if bad:
            failures.append((tag, f"{bad} is_MN_v2 outside in_VH_v2"))
            continue

        pre_cols = list(A.obs.columns)
        pre_snapshot = {c: A.obs[c].copy() for c in pre_cols}

        A.obs["in_VH_v2"] = pd.Series(vh2, index=A.obs_names, dtype="boolean")
        A.obs["is_MN_v2"] = pd.Series(mn2, index=A.obs_names, dtype="boolean")
        A.obs["vh_v2_source"] = pd.Categorical(
            src, categories=["manual", "robin", "excluded"])
        A.obs["has_vh_v2"] = ~excluded
        A.uns["vh_v2_attach"] = {
            "marker": MARKER,
            "source_parquet": PARQ,
            "parquet_sample": psamp,
            "recipe": "is_MN_v2 = is_neuron_v3 & ~is_GAD & in_VH_v2",
            "na_policy": "in_VH_v2/is_MN_v2 = <NA> where vh_v2_source=='excluded'",
            "coord_gate_max_delta_um": float(d),
            "decoy_best_um": float(best_decoy),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        stat = {
            "tag": tag, "our_sample": our_samp, "parquet_sample": psamp, "n_obs": n,
            "vh_v2_source": src[0] if len(set(src)) == 1 else "MIXED",
            "in_VH_v2": int(pd.Series(vh2).astype("boolean").fillna(False).sum()),
            "in_VH_v2_NA": int(pd.Series(vh2).isna().sum()),
            "is_MN_v2": int(mn_l.sum()),
            "is_MN_v2_NA": int(pd.Series(mn2).isna().sum()),
            "is_MN_v1": int(pd.Series(A.obs["is_MN"]).astype("boolean").fillna(False).sum()),
            "in_VH_v1": int(pd.Series(A.obs["in_VH"]).astype("boolean").fillna(False).sum()),
            "coord_gate_um": float(d), "decoy_um": float(best_decoy),
        }
        stat["delta_MN"] = stat["is_MN_v2"] - stat["is_MN_v1"]
        report.append(stat)
        log(f"  in_VH_v2={stat['in_VH_v2']} (NA {stat['in_VH_v2_NA']})  "
            f"is_MN_v2={stat['is_MN_v2']} (v1 {stat['is_MN_v1']}, delta {stat['delta_MN']:+d})")

        # sidecar (portable record for downstream, written even on dry-run)
        side = pd.DataFrame({
            "cell_id": A.obs_names,
            "in_VH_v2": pd.Series(vh2).astype("boolean").to_numpy(dtype=object),
            "is_MN_v2": pd.Series(mn2).astype("boolean").to_numpy(dtype=object),
            "vh_v2_source": src,
        })
        side.to_parquet(os.path.join(SIDECAR, f"{tag}_vh_v2.parquet"), index=False)

        if args.dry_run:
            log("  DRY-RUN: no h5ad written")
            del A
            continue

        # ---- write tmp -> gate -> atomic swap (hardlink backup costs no space)
        tmp = f + ".tmp-vhv2"
        if os.path.exists(tmp):
            os.remove(tmp)
        log(f"  writing {os.path.basename(tmp)}")
        A.write_h5ad(tmp, compression="gzip")
        del A

        log("  gating tmp ...")
        B = ad.read_h5ad(tmp)
        ok, why = True, ""
        if B.n_obs != n:
            ok, why = False, f"n_obs {B.n_obs} != {n}"
        elif not all(c in B.obs.columns for c in NEWCOLS):
            ok, why = False, "new columns missing"
        elif int(pd.Series(B.obs["is_MN_v2"]).astype("boolean").fillna(False).sum()) != stat["is_MN_v2"]:
            ok, why = False, "is_MN_v2 sum drifted"
        elif int(pd.Series(B.obs["in_VH_v2"]).astype("boolean").fillna(False).sum()) != stat["in_VH_v2"]:
            ok, why = False, "in_VH_v2 sum drifted"
        else:
            # every pre-existing column must be bit-identical (NA-aware)
            for c in pre_cols:
                a0, b0 = pre_snapshot[c], B.obs[c]
                if str(a0.dtype) != str(b0.dtype):
                    ok, why = False, f"dtype drift on {c}: {a0.dtype} -> {b0.dtype}"
                    break
                if not a0.reset_index(drop=True).equals(b0.reset_index(drop=True)):
                    ok, why = False, f"value drift on pre-existing column {c}"
                    break
            else:
                for k in ("counts", "log1p_norm"):
                    if k in B.layers and B.layers[k].shape != (n, B.n_vars):
                        ok, why = False, f"layer {k} shape drift"
                        break
                if "spatial" not in B.obsm:
                    ok, why = False, "obsm['spatial'] lost"
        del B
        if not ok:
            log(f"  GATE FAILED: {why} -- leaving original untouched")
            os.remove(tmp)
            failures.append((tag, f"write gate: {why}"))
            continue

        bak = f + BAKSUF
        if not os.path.exists(bak):
            os.link(f, bak)          # zero-cost backup, survives the rename
        os.replace(tmp, f)
        log(f"  SWAPPED ok (backup {os.path.basename(bak)})")

    # ---------------------------------------------------------------- summary
    log("=" * 70)
    t = pd.DataFrame(report)
    if len(t):
        pd.set_option("display.width", 250)
        print("\n" + t.to_string(index=False))
        print(f"\nCOHORT is_MN_v2 = {t['is_MN_v2'].sum()}  (v1 {t['is_MN_v1'].sum()}, "
              f"delta {t['is_MN_v2'].sum() - t['is_MN_v1'].sum():+d})")
        print(f"COHORT in_VH_v2 = {t['in_VH_v2'].sum()}  (v1 {t['in_VH_v1'].sum()})")
        print(f"samples with NA (excluded) = "
              f"{sorted(t.loc[t['is_MN_v2_NA'] > 0, 'tag'].tolist())}")
        print(f"total n_obs = {t['n_obs'].sum()}")
        out = "/home/rodrigok/vhv2_attach_report.csv"
        t.to_csv(out, index=False)
        print(f"\nreport -> {out}")
    if failures:
        print("\n*** FAILURES ***")
        for tg, why in failures:
            print(f"  {tg}: {why}")
        sys.exit(1)
    print("\nALL FILES OK" + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
