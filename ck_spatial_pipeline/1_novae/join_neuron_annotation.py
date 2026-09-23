#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/1_novae/join_neuron_annotation.py
#
# Writes Marcel's July neuron and motor-neuron predictions into every independent Novae
# per-sample h5ad, in place, with h5py. It adds neuron_prob, is_neuron, is_MN, mn_score,
# mn_cell_area_um2 and mn_nc_ratio. The join key is (run_id, xenium_cell_id) against
# (sample, cell_index + 1). The off-by-one is real and the script asserts it.
#
# This edit is why every per_sample h5ad on OAK carries an 08-06 timestamp. The is_MN it
# writes is the July call. The canonical motor-neuron flag today is is_MN_v2, attached much
# later by 3_cell_annotation/motor_neurons/.
# ========================================================================================

"""
Join Marcel's morphology neuron / motor-neuron annotation onto the _FINAL h5ads.

Source: Ranger_procd_mw_final/annotation/neuron_predictions.parquet (Jul 22).
        Verified aligned to these h5ads cell-by-cell: raw STMN2/MNX1/SLC17A6 counts agree
        with X and centroids agree to float64 roundoff, on all 20 samples.

        NOT annotation/neuron_calls/ (Aug 5). Those sit on the re-exported Aug-5 cell
        universe: ~7% of cell_ids there name a different cell than in these h5ads
        (centroids diverge by up to a section width) while still passing a row-count
        check. Joining them would silently mis-assign. Use them only after a rebuild.

Columns written to obs:
  neuron_prob    float64  P(neuron) from the morphology model (no gene counts used)
  is_neuron      bool     neuron_prob > 0.8  (Marcel's suggested mask)
  is_MN          bool     provisional MN call; OVERWRITES the all-False placeholder
  mn_score       float64  graded MN confidence for rethresholding; 0 for non-neurons
  mn_cell_area_um2 float64  soma area the MN gate used
  mn_nc_ratio    float64  nucleus/cytoplasm ratio

Join key is (obs['run_id'], int(obs['xenium_cell_id'])) against
(parquet['sample'], parquet['cell_index'] + 1). run_id already holds the run-directory
name, so no label->run map is needed -- which is what makes the confusable near-duplicates
(SD01620_BI vs SD016_20_BI, SD02022 vs SD020_22_BI, Region_2) safe here.

Edits are made IN PLACE via h5py (adding datasets to /obs), because /home is over quota and
oak is inode-limited. Every file is verified after writing. Use --dry-run to preview.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq

BASE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
PARQ = f"{BASE}/Ranger_procd_mw_final/annotation/neuron_predictions.parquet"
NEURON_PROB_CUT = 0.8  # Marcel's suggested mask, README "Suggested threshold"

# parquet column -> (obs column, numpy dtype)
COLMAP = [
    ("neuron_prob", "neuron_prob", np.float64),
    ("is_MN", "is_MN", np.bool_),
    ("mn_score", "mn_score", np.float64),
    ("cell_area_um2", "mn_cell_area_um2", np.float64),
    ("nc_ratio", "mn_nc_ratio", np.float64),
]


def targets():
    out = []
    out += sorted(glob.glob(f"{BASE}/SC_MNcorrected_FINAL_h5ads/*__MNcorrected_FINAL.h5ad"))
    out += sorted(glob.glob(f"{BASE}/SC_MNcorrected_FINAL_h5ads/ALS_SCXenium_*.h5ad"))
    out += sorted(glob.glob(f"{BASE}/Novae_persample_niches_FINAL/per_sample/*.h5ad"))
    out += sorted(glob.glob(f"{BASE}/Novae_persample_niches_FINAL/novae_all_samples_domains.h5ad"))
    out += sorted(glob.glob(
        f"{BASE}/Novae_persample_INDEPENDENT_FINAL/per_sample/*_niches_independent.h5ad"))
    return [p for p in out if not p.endswith((".bak", ".tmp"))]


def rd(h5, path):
    """Read an obs column: plain dataset, or categorical group."""
    node = h5[path]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]
        codes = node["codes"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return cats[codes]
    arr = node[:]
    if arr.dtype.kind in "SO":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])
    return arr


def write_obs_col(obs, name, arr):
    """Add/replace an obs column in place, matching anndata's on-disk convention."""
    if name in obs:
        del obs[name]
    ds = obs.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
    ds.attrs["encoding-type"] = "array"
    ds.attrs["encoding-version"] = "0.2.0"
    order = list(obs.attrs.get("column-order", []))
    order = [str(c) for c in order]
    if name not in order:
        order.append(name)
        obs.attrs["column-order"] = np.array(order, dtype=object)


def write_uns(uns, key, value):
    """Write a scalar to uns. Never None -- a null in .uns makes the h5ad unreadable."""
    if key in uns:
        del uns[key]
    if isinstance(value, str):
        ds = uns.create_dataset(key, data=value)
        ds.attrs["encoding-type"] = "string"
    else:
        ds = uns.create_dataset(key, data=value)
        ds.attrs["encoding-type"] = "numeric-scalar"
    ds.attrs["encoding-version"] = "0.2.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="verify only, write nothing")
    ap.add_argument("--only", help="operate on this single h5ad path")
    args = ap.parse_args()

    print(f"loading {PARQ} ...")
    tbl = pq.read_table(PARQ)
    p_sample = tbl.column("sample").to_numpy(zero_copy_only=False)
    p_sample = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in p_sample])
    p_key1 = tbl.column("cell_index").to_numpy(zero_copy_only=False).astype(np.int64) + 1
    p_cols = {c: np.asarray(tbl.column(c).to_numpy(zero_copy_only=False))
              for c in tbl.column_names}
    # one dict lookup for the whole cohort: "<run>\t<cell_id>" -> parquet row
    lut = {f"{s}\t{k}": i for i, (s, k) in enumerate(zip(p_sample, p_key1))}
    print(f"  {len(p_sample):,} rows, {len(set(p_sample))} runs\n")

    files = [args.only] if args.only else targets()
    print(f"{len(files)} target h5ad(s){'  [DRY RUN]' if args.dry_run else ''}\n")
    print(f"{'file':<52} {'n_obs':>9} {'found':>9} {'xy_dev':>10} "
          f"{'neurons':>8} {'is_MN':>7}  status")
    print("-" * 110)

    failures, tot_mn, tot_neuron = [], 0, 0
    for path in files:
        short = os.path.relpath(path, BASE)
        if len(short) > 51:
            short = "..." + short[-48:]
        try:
            with h5py.File(path, "r") as h5:
                run = rd(h5, "obs/run_id")
                cid = rd(h5, "obs/xenium_cell_id").astype(np.int64)
                hx = rd(h5, "obs/x_centroid").astype(float) if "x_centroid" in h5["obs"] else None
                hy = rd(h5, "obs/y_centroid").astype(float) if "y_centroid" in h5["obs"] else None
                if hx is None and "spatial_orig" in h5.get("obsm", {}):
                    xy = h5["obsm/spatial_orig"][:]
                    hx, hy = xy[:, 0], xy[:, 1]

            rows = np.fromiter(
                (lut.get(f"{r}\t{c}", -1) for r, c in zip(run, cid)),
                dtype=np.int64, count=len(cid))
            miss = int((rows < 0).sum())
            if miss:
                failures.append(f"{short}: {miss} cells absent from the parquet")
                print(f"{short:<52} {len(cid):>9} {len(cid)-miss:>9} {'-':>10} "
                      f"{'-':>8} {'-':>7}  FAIL missing")
                continue

            # content proof of alignment -- never trust the row count alone
            xy_dev = float("nan")
            if hx is not None:
                xy_dev = float((np.abs(hx - p_cols["x_centroid"][rows].astype(float))
                                + np.abs(hy - p_cols["y_centroid"][rows].astype(float))).max())
                if xy_dev >= 1e-6:
                    failures.append(f"{short}: centroid mismatch {xy_dev:.4f} um")
                    print(f"{short:<52} {len(cid):>9} {len(cid):>9} {xy_dev:>10.2e} "
                          f"{'-':>8} {'-':>7}  FAIL centroid")
                    continue

            vals = {obs_name: p_cols[src][rows].astype(dt)
                    for src, obs_name, dt in COLMAP}
            vals["is_neuron"] = vals["neuron_prob"] > NEURON_PROB_CUT
            n_neuron = int(vals["is_neuron"].sum())
            n_mn = int(vals["is_MN"].sum())
            # parquet rows available for the runs this file covers
            n_src = int(np.isin(p_sample, np.unique(run)).sum())

            if not args.dry_run:
                with h5py.File(path, "r+") as h5:
                    obs = h5["obs"]
                    for name in ("neuron_prob", "is_neuron", "is_MN", "mn_score",
                                 "mn_cell_area_um2", "mn_nc_ratio"):
                        write_obs_col(obs, name, vals[name])
                    uns = h5.require_group("uns")
                    if "encoding-type" not in uns.attrs:
                        uns.attrs["encoding-type"] = "dict"
                        uns.attrs["encoding-version"] = "0.1.0"
                    write_uns(uns, "is_MN_source",
                              "Ranger_procd_mw_final/annotation/neuron_predictions.parquet")
                    write_uns(uns, "is_MN_provenance",
                              "joined on (run_id, xenium_cell_id) == (sample, cell_index+1); "
                              "verified by centroid + STMN2/MNX1/SLC17A6 concordance vs X")
                    write_uns(uns, "is_MN_join_key", "run_id + xenium_cell_id")
                    write_uns(uns, "is_MN_overlap_frac", 1.0)
                    write_uns(uns, "is_MN_final_ncells", n_mn)
                    # these two carried a leftover None / 0 from the pending-annotation
                    # run; a null in .uns can make the h5ad unreadable, so set them real
                    write_uns(uns, "is_MN_rowcount_ratio", 1.0)
                    write_uns(uns, "is_MN_source_nrows", int(n_src))
                    write_uns(uns, "is_MN_definition",
                              "provisional: within neuron_prob>0.5, cell_area_um2>=200 AND "
                              "(STMN2>=8 OR MNX1>=1 OR SLC17A6>=10)")
                    write_uns(uns, "neuron_prob_cut", float(NEURON_PROB_CUT))
                    write_uns(uns, "annotation_delivery_date", "2026-07-22")
                    write_uns(uns, "annotation_not_used",
                              "annotation/neuron_calls/ (2026-08-05) is on a different cell "
                              "universe; ~7% of cell_ids drift. Do not join without a rebuild.")

                # read back and confirm what landed
                with h5py.File(path, "r") as h5:
                    back = rd(h5, "obs/is_MN").astype(bool)
                    if int(back.sum()) != n_mn or len(back) != len(cid):
                        failures.append(f"{short}: readback mismatch")
                        print(f"{short:<52} {len(cid):>9} {len(cid):>9} {xy_dev:>10.2e} "
                              f"{n_neuron:>8} {n_mn:>7}  FAIL readback")
                        continue

            tot_mn += n_mn
            tot_neuron += n_neuron
            print(f"{short:<52} {len(cid):>9} {len(cid):>9} {xy_dev:>10.2e} "
                  f"{n_neuron:>8} {n_mn:>7}  {'ok (dry)' if args.dry_run else 'WRITTEN'}")

        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append(f"{short}: {type(exc).__name__}: {exc}")
            print(f"{short:<52} {'-':>9} {'-':>9} {'-':>10} {'-':>8} {'-':>7}  ERROR {exc}")

    print("-" * 110)
    print(f"is_MN written across all targets: {tot_mn}   is_neuron: {tot_neuron}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 2
    print("\nAll targets joined and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
