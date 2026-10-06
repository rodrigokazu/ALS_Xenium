"""Write a Xenium-Explorer-readable bundle from a final cell label image plus nucleus polygons.

File formats, zarr layout and the polygon helpers are taken from scripts/merge_with_original.py (steps 4i-12),
which produced Ranger_procd_mw_final and is known to open in Xenium Explorer. Differences: the cell mask in
cells.zarr.zip and the transcript -> cell assignment come directly from the label image (not from
re-rasterizing 25-vertex polygons), nucleus_count is 0 for cells without a nucleus, and cluster labels are
inherited from the Xenium cell whose nucleus a cell owns.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.io
import scipy.sparse
import zarr
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points
from skimage.draw import polygon as draw_polygon
from zarr.storage import LocalStore, ZipStore

IMPORTED_METHOD = 6
ZARRAY_FIELD_ORDER = ["chunks", "compressor", "dtype", "fill_value", "filters", "order", "shape", "zarr_format"]


def eliminate_holes_keyhole(poly, channel_width=0.1, overshoot=0.4):
    """Xenium stores one exterior ring per cell; cut a thin channel from every hole to the outside so no area
    is silently lost (see merge_with_original.py for the full rationale)."""
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if not poly.interiors:
        return poly
    result = Polygon(poly.exterior)
    for interior in poly.interiors:
        hole = Polygon(interior)
        a, b = nearest_points(hole.exterior, poly.exterior)
        a, b = np.array(a.coords[0]), np.array(b.coords[0])
        d = b - a
        n = np.linalg.norm(d)
        if n == 0:
            continue
        channel = LineString([tuple(a), tuple(b + d / n * overshoot)]).buffer(channel_width, cap_style=2)
        try:
            result = result.difference(hole.union(channel))
        except Exception:
            continue
        if result.geom_type == "MultiPolygon":
            result = max(result.geoms, key=lambda g: g.area)
    if not result.is_valid:
        result = result.buffer(0)
        if result.geom_type == "MultiPolygon":
            result = max(result.geoms, key=lambda g: g.area)
    return result


def prepare_poly_coords(poly, max_verts=25):
    poly = eliminate_holes_keyhole(poly)
    coords = np.array(poly.exterior.coords[:-1])
    if len(coords) > max_verts:
        coords = coords[np.round(np.linspace(0, len(coords) - 1, max_verts)).astype(int)]
    return coords


def _flat(coords):
    out = np.zeros(50, dtype=np.float32)
    f = coords.flatten().astype(np.float32)
    out[: len(f)] = f
    return out


def atomic_to_parquet(df, path, **kw):
    tmp = Path(f"{path}.tmp{os.getpid()}")
    df.to_parquet(tmp, **kw)
    os.replace(tmp, path)


def atomic_to_csv(df, path, **kw):
    tmp = Path(f"{path}.tmp{os.getpid()}")
    df.to_csv(tmp, **kw)
    os.replace(tmp, path)


def _clean_zarray(raw):
    d = json.loads(raw)
    d.pop("dimension_separator", None)
    ordered = {k: d[k] for k in ZARRAY_FIELD_ORDER if k in d}
    ordered.update({k: v for k, v in d.items() if k not in ordered})
    return json.dumps(ordered, indent=4).encode()


def zip_dir(tmpdir, out_zip, skip_empty_zattrs=True):
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for dirpath, _, files in os.walk(tmpdir):
            for fn in files:
                fp = os.path.join(dirpath, fn)
                arc = os.path.relpath(fp, tmpdir)
                if skip_empty_zattrs and fn == ".zattrs" and arc != ".zattrs" and open(fp).read().strip() in ("{}", ""):
                    continue
                if fn == ".zarray":
                    zf.writestr(arc, _clean_zarray(open(fp, "rb").read()))
                else:
                    zf.write(fp, arc)


def _poly_rows(label, poly):
    if poly is None or poly.is_empty:
        return []
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.interiors:
        poly = eliminate_holes_keyhole(poly)
    xs, ys = poly.exterior.coords.xy
    return [(label, float(x), float(y)) for x, y in zip(xs, ys)]


def assign_transcripts(path: Path, labels: np.ndarray, um_per_px: float, N: int, g_idx: dict,
                       batch_size: int = 20_000_000):
    """Rewrite transcripts.parquet with cell_id = the label under each transcript, in row batches.

    Returns the per-cell gene transcript counts (is_gene, cell_id != 0; no qv filter) and the features x cells
    count matrix as COO ordered by cell then feature. Streaming keeps the peak at one batch plus the per-batch
    (feature, cell) counts, which a whole-transcriptome sample (3.4 B transcripts) needs; the result is the same
    as assigning the whole table at once."""
    H, W = labels.shape
    pf = pq.ParquetFile(path)
    tmp = Path(f"{path}.tmp{os.getpid()}")
    writer = None
    gene_counts = np.zeros(N + 1, dtype=np.int64)
    keys, counts = [], []
    try:
        for b in pf.iter_batches(batch_size=batch_size):
            col = np.floor(b.column("x_location").to_numpy() / um_per_px).astype(np.int64)
            row = np.floor(b.column("y_location").to_numpy() / um_per_px).astype(np.int64)
            ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            cid = np.zeros(len(b), dtype=np.int64)
            order = np.argsort(row[ok], kind="stable")  # row-sorted reads keep the memmap access sequential
            idx = np.flatnonzero(ok)[order]
            cid[idx] = labels[row[idx], col[idx]]
            names = b.schema.names
            arr = pa.array(cid, type=pa.int64())
            if "cell_id" in names:
                b = b.set_column(names.index("cell_id"), "cell_id", arr)
            else:
                b = b.append_column("cell_id", arr)
            if writer is None:
                writer = pq.ParquetWriter(tmp, b.schema)
            writer.write_batch(b)
            gm = (cid != 0) & pc.cast(b.column("is_gene"), pa.bool_()).to_numpy(zero_copy_only=False)
            c = cid[gm]
            gene_counts += np.bincount(c, minlength=N + 1)
            fn = pc.dictionary_encode(pc.cast(pc.filter(b.column("feature_name"), pa.array(gm)), pa.string()))
            fmap = np.array([g_idx.get(n, -1) for n in fn.dictionary.to_pylist()], dtype=np.int64)
            f = fmap[fn.indices.to_numpy(zero_copy_only=False)] if len(fmap) else np.empty(0, np.int64)
            keep = f >= 0
            k, n = np.unique(c[keep] * len(g_idx) + f[keep], return_counts=True)  # key = cell x feature
            keys.append(k)
            counts.append(n.astype(np.int32))
        if writer is not None:
            writer.close()
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    k, n = (np.concatenate(keys), np.concatenate(counts)) if keys else (np.empty(0, np.int64), np.empty(0, np.int32))
    del keys, counts
    F = len(g_idx)
    mat = scipy.sparse.csc_matrix((n, (k % F, k // F - 1)), shape=(F, N))  # duplicates across batches summed
    mat.sort_indices()
    return gene_counts[1:], mat.tocoo()


def write_bundle(base: Path, backup: Path, labels: np.ndarray, um_per_px: float, cell_polys: dict,
                 nuc_polys: dict, backup_index: np.ndarray) -> None:
    """base: staged work dir holding experiment.xenium, transcripts.parquet and
    cell_feature_matrix/features.tsv.gz; backup: the Xenium bundle the run started from (cells.zarr.zip,
    analysis.zarr.zip); labels: final cell label image (level-0, values 1..N); cell_polys / nuc_polys: label ->
    shapely Polygon in um; backup_index: (N,) row index of the Xenium cell each new cell inherits its cluster
    from, -1 for none."""
    N = int(len(backup_index))
    H, W = labels.shape
    labels_all = np.arange(1, N + 1)
    cfm = base / "cell_feature_matrix"

    # boundaries
    cb = pd.DataFrame([r for l in labels_all for r in _poly_rows(int(l), cell_polys.get(int(l)))],
                      columns=["cell_id", "vertex_x", "vertex_y"])
    nb = pd.DataFrame([r for l in labels_all for r in _poly_rows(int(l), nuc_polys.get(int(l)))],
                      columns=["cell_id", "vertex_x", "vertex_y"])
    for df, name in ((cb, "cell_boundaries"), (nb, "nucleus_boundaries")):
        df["label_id"] = df["cell_id"]
        atomic_to_parquet(df, base / f"{name}.parquet", index=False)
        atomic_to_csv(df, base / f"{name}.csv.gz", index=False, compression="gzip")

    # per-cell summary
    cs = np.zeros((N, 8))
    for i, l in enumerate(labels_all):
        c, n = cell_polys.get(int(l)), nuc_polys.get(int(l))
        if c is not None:
            cs[i, :3] = c.centroid.x, c.centroid.y, c.area
        if n is not None:
            cs[i, 3:6] = n.centroid.x, n.centroid.y, n.area
            cs[i, 7] = 1
        else:
            cs[i, 3:5] = cs[i, :2]

    # cell_feature_matrix features (the transcript pass below counts into them)
    with gzip.open(cfm / "features.tsv.gz", "rt") as f:
        feat = [l.rstrip("\n").split("\t") for l in f]
    feat_ids = [x[0] for x in feat]
    feat_names = [x[1] if len(x) > 1 else x[0] for x in feat]
    feat_types = [x[2] if len(x) > 2 else "Gene Expression" for x in feat]
    g_idx = {n: i for i, n in enumerate(feat_names)}

    # transcripts: the cell under each transcript in the label image
    gene_counts, mat = assign_transcripts(base / "transcripts.parquet", labels, um_per_px, N, g_idx)

    # cells.parquet
    cells = pd.DataFrame({
        "cell_id": [str(l) for l in labels_all], "x_centroid": cs[:, 0], "y_centroid": cs[:, 1],
        "transcript_counts": gene_counts, "control_probe_counts": 0, "genomic_control_counts": 0,
        "control_codeword_counts": 0, "unassigned_codeword_counts": 0, "deprecated_codeword_counts": 0,
        "total_counts": gene_counts, "cell_area": cs[:, 2], "nucleus_area": cs[:, 5],
        "nucleus_count": cs[:, 7].astype(int), "segmentation_method": "Imported Cell Segmentation"})
    atomic_to_parquet(cells, base / "cells.parquet", index=False)
    atomic_to_csv(cells, base / "cells.csv.gz", index=False, compression="gzip")

    # cell_feature_matrix (mtx + zarr)
    with gzip.open(cfm / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(str(l) for l in labels_all) + "\n")
    scipy.io.mmwrite(str(cfm / "_tmp.mtx"), mat)
    with open(cfm / "_tmp.mtx", "rb") as fi, gzip.open(cfm / "matrix.mtx.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(cfm / "_tmp.mtx")

    # cells.zarr.zip
    bk = zarr.open(ZipStore(str(backup / "cells.zarr.zip"), mode="r"), mode="r")
    bk_attrs = dict(bk.attrs)
    transform = bk["masks"]["homogeneous_transform"][:]
    mshape, mchunks = bk["masks"]["0"].shape, bk["masks"]["0"].chunks
    bk.store.close()
    px_per_um = float(transform[0, 0])  # the transform maps um -> mask pixels (4.7059 = 1 / 0.2125 at level 0)
    import numcodecs
    comp = numcodecs.Blosc(cname="zstd", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
    tmp = tempfile.mkdtemp(suffix="_xen")
    root = zarr.open_group(LocalStore(tmp), mode="w", zarr_format=2)
    root.attrs.update({**bk_attrs, "number_cells": N})
    u32 = labels_all.astype(np.uint32)
    a = root.create_array("cell_id", shape=(N, 2), dtype=np.uint32, chunks=(min(N, 57686), 1))
    a[:] = np.stack([u32, np.ones(N, dtype=np.uint32)], axis=1)
    a = root.create_array("cell_summary", shape=(N, 8), dtype=np.float64, chunks=(N, 1), fill_value=np.nan)
    a[:] = cs
    a.attrs.update({"column_descriptions": ["Cell centroid in X", "Cell centroid in Y", "Cell area",
                                            "Nucleus centroid in X", "Nucleus centroid in Y", "Nucleus area",
                                            "z_level", "Nucleus count"],
                    "column_names": ["cell_centroid_x", "cell_centroid_y", "cell_area", "nucleus_centroid_x",
                                     "nucleus_centroid_y", "nucleus_area", "z_level", "nucleus_count"]})
    ps = root.require_group("polygon_sets")
    for gname, polys in (("0", nuc_polys), ("1", cell_polys)):
        entries = [(i, int(l)) for i, l in enumerate(labels_all) if int(l) in polys]
        if not entries:
            continue
        prepared = [prepare_poly_coords(polys[l]) for _, l in entries]
        M = len(entries)
        chk = min(M, 55483)
        g = ps.require_group(gname)
        for name, arr, c in (("cell_index", np.array([i for i, _ in entries], dtype=np.uint32), (chk,)),
                             ("num_vertices", np.array([len(x) for x in prepared], dtype=np.int32), (chk,)),
                             ("method", np.full(M, IMPORTED_METHOD, dtype=np.uint32), (chk,)),
                             ("vertices", np.array([_flat(x) for x in prepared], dtype=np.float32), (min(M, 14422), 13))):
            arr_z = g.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=c)
            arr_z[:] = arr
    masks = root.require_group("masks")
    a = masks.create_array("homogeneous_transform", shape=transform.shape, dtype=np.float32, chunks=(4, 4))
    a[:] = transform
    CH, CW = mchunks
    step = (1.0 / px_per_um) / um_per_px  # label pixels per mask pixel (1.0 when both are level-0)
    nmask = masks.create_array("0", shape=mshape, dtype=np.uint32, chunks=mchunks, compressor=comp, fill_value=0)
    cmask = masks.create_array("1", shape=mshape, dtype=np.uint32, chunks=mchunks, compressor=comp, fill_value=0)
    for r0 in range(0, mshape[0], CH):
        r1 = min(mshape[0], r0 + CH)
        rows_src = np.minimum((np.arange(r0, r1) * step).astype(np.int64), H - 1)
        band = np.asarray(labels[rows_src[0]:rows_src[-1] + 1])
        for c0 in range(0, mshape[1], CW):
            c1 = min(mshape[1], c0 + CW)
            cols_src = np.minimum((np.arange(c0, c1) * step).astype(np.int64), W - 1)
            tile = band[np.ix_(rows_src - rows_src[0], cols_src)].astype(np.uint32)
            if tile.any():
                cmask[r0:r1, c0:c1] = tile
    ntile = {}
    for l, poly in nuc_polys.items():
        poly = eliminate_holes_keyhole(poly)
        minx, miny, maxx, maxy = poly.bounds
        for cr in range(int(miny * px_per_um) // CH, int(maxy * px_per_um) // CH + 1):
            for cc in range(int(minx * px_per_um) // CW, int(maxx * px_per_um) // CW + 1):
                ntile.setdefault((cr, cc), []).append((l, poly))
    for (cr, cc), items in ntile.items():
        r0, c0 = cr * CH, cc * CW
        r1, c1 = min(mshape[0], r0 + CH), min(mshape[1], c0 + CW)
        if r1 <= r0 or c1 <= c0:
            continue
        t = np.zeros((r1 - r0, c1 - c0), dtype=np.uint32)
        for l, poly in items:
            rr, cc2 = draw_polygon(np.array(poly.exterior.xy[1]) * px_per_um - r0,
                                   np.array(poly.exterior.xy[0]) * px_per_um - c0, shape=t.shape)
            t[rr, cc2] = l
        nmask[r0:r1, c0:c1] = t
    zip_dir(tmp, str(base / "cells.zarr.zip"))
    shutil.rmtree(tmp)

    csr, csc = mat.tocsr(), mat.tocsc()
    nnz = max(1, mat.nnz)
    tmp = tempfile.mkdtemp(suffix="_cfm")
    grp = zarr.open_group(LocalStore(tmp), mode="w", zarr_format=2).require_group("cell_features")
    grp.attrs.update({"feature_ids": feat_ids, "feature_keys": feat_names, "feature_types": feat_types,
                      "number_cells": N, "number_features": len(feat_ids)})

    def put(g, name, arr, chunks):
        z = g.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=chunks)
        z[:] = arr

    put(grp, "cell_id", np.stack([u32, np.ones(N, dtype=np.uint32)], axis=1), (N, 2))
    put(grp, "data", csr.data.astype(np.uint32), (nnz,))
    put(grp, "indices", csr.indices.astype(np.uint32), (nnz,))
    put(grp, "indptr", csr.indptr.astype(np.uint32), (len(feat_ids) + 1,))
    cg = grp.require_group("csc")
    put(cg, "data", csc.data.astype(np.uint32), (nnz,))
    put(cg, "indices", csc.indices.astype(np.uint16), (nnz,))
    put(cg, "indptr", csc.indptr.astype(np.uint32), (N + 1,))
    zip_dir(tmp, str(base / "cell_feature_matrix.zarr.zip"), skip_empty_zattrs=False)
    shutil.rmtree(tmp)

    # analysis.zarr.zip: each cell inherits the cluster of the Xenium cell whose nucleus it owns
    bka = zarr.open(ZipStore(str(backup / "analysis.zarr.zip"), mode="r"), mode="r")
    grouping_names = list(bka["cell_groups"].attrs.get("grouping_names", []))
    group_names = list(bka["cell_groups"].attrs.get("group_names", []))
    tmp = tempfile.mkdtemp(suffix="_ana")
    cgr = zarr.open_group(LocalStore(tmp), mode="w", zarr_format=2).require_group("cell_groups")
    cgr.attrs.update({"group_names": group_names, "grouping_names": grouping_names, "major_version": 1,
                      "minor_version": 1, "number_groupings": len(grouping_names)})
    for gi in range(len(grouping_names)):
        g = bka["cell_groups"][str(gi)]
        ind, ptr = g["indices"][:], g["indptr"][:]
        n_bk = int(ind.max()) + 1 if len(ind) else 0
        orig = np.zeros(max(n_bk, int(backup_index.max()) + 1, 1), dtype=np.int32)
        for k in range(len(ptr) - 1):
            orig[ind[ptr[k]:ptr[k + 1]]] = k
        assign = np.where(backup_index >= 0, orig[np.clip(backup_index, 0, None)], 0).astype(np.int32)
        n_cl = len(group_names[gi])
        indptr = np.zeros(n_cl + 1, dtype=np.uint32)
        for k in range(n_cl):
            indptr[k + 1] = indptr[k] + int((assign == k).sum())
        sub = cgr.require_group(str(gi))
        put(sub, "indices", np.argsort(assign, kind="stable").astype(np.uint32), (min(N, 55483),))
        put(sub, "indptr", indptr, (n_cl + 1,))
    bka.store.close()
    zip_dir(tmp, str(base / "analysis.zarr.zip"))
    shutil.rmtree(tmp)

    exp = json.load(open(base / "experiment.xenium"))
    exp["num_cells"] = N
    exp["analysis_uuid"] = str(uuid.uuid4())
    json.dump(exp, open(base / "experiment.xenium", "w"), indent=4)
