#!/usr/bin/env python3
"""Run splitmerge.rules on a whole Xenium sample, in overlapping tiles, with whole-sample statistics.

Pass 1 estimates, from a systematic subsample of every tile, the statistics the method must take over the
whole sample rather than per tile: the tissue DAPI threshold (Otsu on log DAPI), each channel's tissue
median (background of the enrichment signal), quantile tables of the smoothed and unsmoothed 18S - Vimentin
enrichment signal (main segmentation and sense check), the median DAPI inside Xenium's nuclei and the
quantiles of their DAPI contrast (missed-nucleus rescue). Pass 2 runs rules.segment_region on
each tile plus a margin with those statistics and keeps the cells whose centroid lies in the tile's core,
so every cell is decided once, by the tile that sees all of it.

Outputs in --out:
  final_labels.npy      uint32 (H, W) level-0 label image of the final cells (memory-mapped)
  cells.parquet         one row per label: source (18S / xenium_fallback), nucleus_id (Xenium cell_id or
                        dapi_rescue_*), has_nucleus, area_um2, centroid_um
  rescued_nuclei.parquet  rescued nucleus polygons (cell_id, vertex_x, vertex_y in um)
  stats.json            the whole-sample statistics and run parameters

Usage:
  python scg/run_fullsample.py --bundle <xenium dir> --sample SD03914_BG --out <dir> \
      [--pct 0 811 0 1871] [--workers 8] [--window x0 y0 x1 y1] [--tile 4096] [--margin 800]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tifffile
import zarr
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import threshold_otsu

# splitmerge/rules.py is used unchanged from the paper pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge import rules as R  # noqa: E402

TILE_PX = 4096
MARGIN_PX = 800          # 170 um: larger than any soma plus proximal process in this tissue
STATS_STRIDE = 4          # 1-in-4 pixels per axis, as for the channel normalization
QGRID = np.round(np.arange(50.0, 99.951, 0.05), 2)

_G: dict = {}


# --density: segment on transcript density instead of 18S (option B). Passed through the environment so worker
# processes see it under both fork (Linux) and spawn (macOS, where module globals are not inherited).
DENSITY_ENV = "ATERA_DENSITY_ZARR"


TISSUE_ENV = "ATERA_DENSITY_TISSUE_THR"   # density mode: tissue = transcript density above this, not DAPI
_LAST_DENSITY: dict = {}


def _density_tissue_mask(dapi, thr=None, _orig=R.tissue_mask):
    """rules.tissue_mask on the density window read just before (same shape), same fill + 5 px dilation, but
    tissue is defined by transcripts: DAPI only lights nuclei, so the DAPI mask left out the cytoplasm and
    neuropil between them (45% of a granule-layer window, and 79% of its unlabelled pixels above 28 tx/um^2)."""
    d = _LAST_DENSITY.get("d")
    if d is None or d.shape != dapi.shape:
        return _orig(dapi, thr)
    m = ndi.binary_dilation(ndi.binary_fill_holes(d > float(os.environ[TISSUE_ENV])), iterations=5)
    return m if m.any() else np.ones_like(m)


class DensityStack:
    """(4, H, W) view: DAPI and boundary stain from the morphology, transcript density (make_density_image.py) in
    the 18S slot and zeros in the Vimentin slot, so rules.py's 18S - Vimentin signal becomes the whole-sample-
    normalized transcript density and nothing else in the method changes."""

    def __init__(self, morph, dens):
        self.morph, self.dens = morph, dens
        self.shape = (4,) + tuple(morph.shape[-2:])

    def __getitem__(self, key):
        _, ys, xs = key
        m = np.asarray(self.morph[0:2, ys, xs]).astype(np.float32)
        d = np.asarray(self.dens[ys, xs]).astype(np.float32)
        _LAST_DENSITY["d"] = d
        return np.stack([m[0], m[1], d, np.zeros_like(d)])


COMBINED_ENV = "ATERA_COMBINED"   # "bg18,hi18,cut18,bgd,hid,cutd": 18S and density window anchors and main cuts


class CombinedStack(DensityStack):
    """As DensityStack, but the 18S slot holds both stains: s = 0.5 * max(n18 / cut18, nd / cutd) clipped to [0, 1],
    with n18 / nd the whole-sample-normalized 18S and density (background to p99). A pixel is at 0.5 exactly where
    either stain reaches its own main cut, so one threshold (0.5) serves both: 18S-bright somata (Purkinje, large
    neurons) and density-defined cells pass the same mask, and every later rule reads this one channel."""

    def __getitem__(self, key):
        _, ys, xs = key
        m = np.asarray(self.morph[0:3, ys, xs]).astype(np.float32)
        d = np.asarray(self.dens[ys, xs]).astype(np.float32)
        _LAST_DENSITY["d"] = d
        bg18, hi18, cut18, bgd, hid, cutd = (float(v) for v in os.environ[COMBINED_ENV].split(","))
        n18 = np.clip((m[2] - bg18) / (hi18 - bg18), 0, 1) / cut18
        nd = np.clip((d - bgd) / (hid - bgd), 0, 1) / cutd
        comb = np.clip(0.5 * np.maximum(n18, nd), 0, 1)
        return np.stack([m[0], m[1], comb, np.zeros_like(comb)])


def open_morphology(bundle: Path):
    tf = tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif")
    morph = zarr.open(tf.aszarr(level=0, series=0), mode="r")
    path = os.environ.get(DENSITY_ENV)
    if not path:
        return morph
    dens = zarr.open_array(path, mode="r")
    assert tuple(dens.shape) == tuple(morph.shape[-2:]), "density image must be on the morphology grid"
    if os.environ.get(TISSUE_ENV):
        R.tissue_mask = _density_tissue_mask
    return (CombinedStack if os.environ.get(COMBINED_ENV) else DensityStack)(morph, dens)


def centroids(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("cell_id", sort=False)[["vertex_x", "vertex_y"]].mean()


def read_qc_transcripts(path: Path, bbox_um=None) -> pd.DataFrame:
    """QC transcripts (is_gene, qv >= 20): x / y as stored, categorical feature_name, sorted by y.

    Read in batches and filtered as they come, so the peak is the kept columns rather than the whole table
    with strings (a whole-transcriptome Atera sample has 3.4 B rows); sorted by y so a tile takes its band
    with a binary search instead of a pass over every transcript (tile_transcripts). bbox_um (x0, y0, x1, y1):
    keep only transcripts inside it (inclusive), for a --window run whose tiles never look further out."""
    pf = pq.ParquetFile(path)
    xs, ys, codes, cats = [], [], [], {}
    for b in pf.iter_batches(batch_size=50_000_000, columns=["x_location", "y_location", "feature_name", "qv", "is_gene"]):
        keep = pc.and_(pc.cast(b.column("is_gene"), pa.bool_()), pc.greater_equal(b.column("qv"), 20))
        if bbox_um is not None:
            bx0, by0, bx1, by1 = bbox_um
            keep = pc.and_(keep, pc.and_(pc.and_(pc.greater_equal(b.column("x_location"), bx0), pc.less_equal(b.column("x_location"), bx1)),
                                         pc.and_(pc.greater_equal(b.column("y_location"), by0), pc.less_equal(b.column("y_location"), by1))))
        b = b.filter(keep)
        xs.append(b.column("x_location").to_numpy())
        ys.append(b.column("y_location").to_numpy())
        fn = pc.dictionary_encode(pc.cast(b.column("feature_name"), pa.string()))
        local = [cats.setdefault(s, len(cats)) for s in fn.dictionary.to_pylist()]
        codes.append(np.asarray(local, dtype=np.int32)[fn.indices.to_numpy(zero_copy_only=False)] if local
                     else np.empty(0, dtype=np.int32))
    x, y, code = (np.concatenate(v) if v else np.empty(0, np.int32) for v in (xs, ys, codes))
    del xs, ys, codes
    order = np.argsort(y, kind="stable")
    names = np.array(sorted(cats, key=cats.get), dtype=object)
    tx = pd.DataFrame({"x_location": x[order], "y_location": y[order],
                       "feature_name": pd.Categorical.from_codes(code[order], categories=names)})
    tx.attrs["y_sorted"] = True
    return tx


def tile_transcripts(tx: pd.DataFrame, fx0, fy0, fx1, fy1) -> pd.DataFrame:
    """Transcripts inside the window (level-0 px, inclusive bounds as Series.between)."""
    um = R.UM_PER_PX
    if tx.attrs.get("y_sorted"):
        yv = tx["y_location"].to_numpy()
        i0, i1 = np.searchsorted(yv, fy0 * um, "left"), np.searchsorted(yv, fy1 * um, "right")
        band = tx.iloc[i0:i1]
        return band[band.x_location.between(fx0 * um, fx1 * um)]
    return tx[tx.x_location.between(fx0 * um, fx1 * um) & tx.y_location.between(fy0 * um, fy1 * um)]


def tiles(x0, y0, x1, y1):
    for ty in range(y0, y1, TILE_PX):
        for tx in range(x0, x1, TILE_PX):
            yield tx, ty, min(tx + TILE_PX, x1), min(ty + TILE_PX, y1)


def _init(bundle, pct, stats, params, nb, cb, tx, nuc_cent, cell_cent, tile_px, margin_px):
    global TILE_PX, MARGIN_PX
    TILE_PX, MARGIN_PX = tile_px, margin_px
    _G.update(z=open_morphology(Path(bundle)), pct=pct, stats=stats, p=params, nb=nb, cb=cb, tx=tx,
              nuc_cent=nuc_cent, cell_cent=cell_cent)


def _window(core, shape):
    cx0, cy0, cx1, cy1 = core
    H, W = shape
    return max(0, cx0 - MARGIN_PX), max(0, cy0 - MARGIN_PX), min(W, cx1 + MARGIN_PX), min(H, cy1 + MARGIN_PX)


def _subset(df: pd.DataFrame, cent: pd.DataFrame, fx0, fy0, fx1, fy1):
    um = R.UM_PER_PX
    ids = cent.index[cent.vertex_x.between(fx0 * um, fx1 * um) & cent.vertex_y.between(fy0 * um, fy1 * um)]
    return df[df.cell_id.isin(ids)]


def stats_tile(core):
    """Subsampled values for the whole-sample statistics (pass 1)."""
    z, (lo18, hi18, lov, hiv) = _G["z"], _G["pct"]
    p = _G["p"]
    pad = p.median_px
    cx0, cy0, cx1, cy1 = core
    H, W = z.shape[-2:]
    fx0, fy0, fx1, fy1 = max(0, cx0 - pad), max(0, cy0 - pad), min(W, cx1 + pad), min(H, cy1 + pad)
    crop = np.asarray(z[:, fy0:fy1, fx0:fx1]).astype(np.float32)
    dapi = crop[0]
    sub = (slice(cy0 - fy0, cy1 - fy0, STATS_STRIDE), slice(cx0 - fx0, cx1 - fx0, STATS_STRIDE))
    nz = dapi[sub][dapi[sub] > 0]
    return {"dapi_nonzero": nz, "crop": (fx0, fy0), "shape": crop.shape}


def stats_tile_channels(core, dapi_tissue_thr):
    """Subsampled raw 18S and Vimentin over tissue: whole-sample medians (signal background) and p1 / p99."""
    z = _G["z"]
    cx0, cy0, cx1, cy1 = core
    crop = np.asarray(z[:, cy0:cy1, cx0:cx1])
    if not (crop[0] > dapi_tissue_thr).any():   # glass only (tissue_mask would fall back to all-ones)
        return np.empty(0), np.empty(0)
    valid = R.tissue_mask(crop[0].astype(np.float32), dapi_tissue_thr)[::STATS_STRIDE, ::STATS_STRIDE]
    return crop[2, ::STATS_STRIDE, ::STATS_STRIDE][valid], crop[3, ::STATS_STRIDE, ::STATS_STRIDE][valid]


def stats_tile_signal(core, dapi_tissue_thr, bg18, bgvim):
    """Subsampled smoothed / raw signal over positive tissue pixels, DAPI inside Xenium's nuclei, and the DAPI
    contrast of every Xenium nucleus whose centroid lies in the core."""
    z = _G["z"]
    p = _G["p"]
    pad = max(p.median_px, int(np.ceil(R.um_to_px(p.contrast_ring_um))) + 2)
    cx0, cy0, cx1, cy1 = core
    H, W = z.shape[-2:]
    fx0, fy0, fx1, fy1 = max(0, cx0 - pad), max(0, cy0 - pad), min(W, cx1 + pad), min(H, cy1 + pad)
    crop = np.asarray(z[:, fy0:fy1, fx0:fx1]).astype(np.float32)
    if not (crop[0] > dapi_tissue_thr).any():
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    valid = R.tissue_mask(crop[0], dapi_tissue_thr)
    diff = R.sample_signal(crop, _G["pct"], bg18, bgvim)
    filt, positive = R.smoothed_signal(diff, valid, p.median_px)
    sub = (slice(cy0 - fy0, cy1 - fy0, STATS_STRIDE), slice(cx0 - fx0, cx1 - fx0, STATS_STRIDE))
    pos = positive[sub]
    nb = _subset(_G["nb"], _G["nuc_cent"], fx0, fy0, fx1, fy1)   # every nucleus of the window, for the rings
    nuc_dapi, contrast = np.empty(0), np.empty(0)
    if len(nb):
        nuc, ids = R.rasterize_nuclei(nb, fx0, fy0, crop.shape[-2:])
        d = ndi.gaussian_filter(crop[0].astype(float), R.um_to_px(0.5))
        nuc_dapi = d[sub][nuc[sub] > 0]
        core_ids = set(_subset(_G["nb"], _G["nuc_cent"], cx0, cy0, cx1, cy1).cell_id)
        present = set(np.unique(nuc[nuc > 0]).tolist())
        lab = np.array([k for k, c in enumerate(ids, start=1) if c in core_ids and k in present], dtype=int)
        contrast = R.nucleus_contrast(d, nuc, lab, p.contrast_ring_um)
    return filt[sub][pos], diff[sub][pos], nuc_dapi, contrast


def run_tile(core):
    """Pass 2: the method on core + margin; returns the cells whose centroid lies in the core."""
    z, pct, st, p = _G["z"], _G["pct"], _G["stats"], _G["p"]
    H, W = z.shape[-2:]
    fx0, fy0, fx1, fy1 = _window(core, (H, W))
    crop = np.asarray(z[:, fy0:fy1, fx0:fx1]).astype(np.float32)
    # glass only (tissue_mask falls back to all-ones when nothing passes, so test the threshold itself)
    if not (crop[0] > st["dapi_tissue_thr"]).any():
        return core, []
    nb = _subset(_G["nb"], _G["nuc_cent"], fx0, fy0, fx1, fy1)
    cb = _subset(_G["cb"], _G["cell_cent"], fx0, fy0, fx1, fy1)
    um = R.UM_PER_PX
    tx = tile_transcripts(_G["tx"], fx0, fy0, fx1, fy1)
    if os.environ.get("ATERA_V2"):
        import seg_v2
        rr = seg_v2.segment_v2(crop, pct, nb, cb, tx, fx0, fy0, p, st)
    else:
        rr = R.segment_region(crop, pct, nb, cb, tx, fx0, fy0, p, st)
    final, nuc, cells, cell_ids = rr["final"], rr["nuc"], rr["cells"], rr["cell_ids"]
    owner, _ = R.nucleus_owners(final, nuc, p)
    nucleus_of = {}
    for k, lab in owner.items():
        nucleus_of.setdefault(lab, k)
    cx0, cy0, cx1, cy1 = core
    out = []
    for lab, sl in enumerate(ndi.find_objects(final), start=1):
        if sl is None or lab not in cells.index:
            continue
        m = final[sl] == lab
        rows, cols = np.nonzero(m)
        r0, c0 = sl[0].start + fy0, sl[1].start + fx0
        cy, cx = rows.mean() + r0, cols.mean() + c0
        if not (cx0 <= cx < cx1 and cy0 <= cy < cy1):
            continue
        k = nucleus_of.get(lab)
        nuc_poly = None
        if k is not None and k > rr["n_xenium_nuclei"]:
            cs = measure.find_contours(np.pad(nuc[sl] == k, 1).astype(float), 0.5)
            if cs:
                c = max(cs, key=len)
                nuc_poly = np.column_stack([(c[:, 1] - 1 + c0 + 0.5) * um, (c[:, 0] - 1 + r0 + 0.5) * um])
        out.append({"r0": r0, "c0": c0, "mask": np.packbits(m, axis=None), "mshape": m.shape,
                    "source": cells.at[lab, "source"],
                    "nucleus_id": cell_ids[k - 1] if k is not None else "",
                    "rescued": bool(k is not None and k > rr["n_xenium_nuclei"]), "nuc_poly": nuc_poly,
                    "area_um2": float(m.sum() * um * um), "cx_um": float(cx * um), "cy_um": float(cy * um)})
    return core, out


def pass1(a, p, nb, cb, tx, nuc_cent, cell_cent, cores, t0):
    """Whole-sample statistics from a systematic subsample of every tile."""
    pct0 = tuple(a.pct) if a.pct is not None else (0.0, 1.0, 0.0, 1.0)  # the DAPI pass does not use them
    init = (str(a.bundle), pct0, {}, p, nb, cb, tx, nuc_cent, cell_cent, TILE_PX, MARGIN_PX)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=init) as ex:
        nz = np.concatenate([r["dapi_nonzero"] for r in ex.map(stats_tile, cores)])
    # tissue vs glass: Otsu split of log DAPI over the whole sample. The "5th percentile of non-zero DAPI" rule
    # used inside a crop lands in the empty-glass background on a whole slide (3-6 vs ~31 in tissue crops), which
    # lets glass into every whole-sample percentile, differently per slide.
    dapi_tissue_thr = float(np.expm1(threshold_otsu(np.log1p(nz.astype(float))))) if nz.size else 0.0
    # channel backgrounds (tissue medians) for the enrichment signal; p1 / p99 kept for the record
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=init) as ex:
        ch = list(ex.map(stats_tile_channels, cores, [dapi_tissue_thr] * len(cores)))
    r18 = np.concatenate([c[0] for c in ch])
    rvim = np.concatenate([c[1] for c in ch])
    bg18, bgvim = float(np.median(r18)), float(np.median(rvim))
    if a.pct is None:
        a.pct = [float(np.percentile(r18, 1)), float(np.percentile(r18, 99)),
                 float(np.percentile(rvim, 1)), float(np.percentile(rvim, 99))]
        print(f"channel percentiles (18S p1/p99, Vim p1/p99) from the sample's tissue: {a.pct}", flush=True)
        init = (str(a.bundle), tuple(a.pct), {}, p, nb, cb, tx, nuc_cent, cell_cent, TILE_PX, MARGIN_PX)
    n = len(cores)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=init) as ex:
        parts = list(ex.map(stats_tile_signal, cores, [dapi_tissue_thr] * n, [bg18] * n, [bgvim] * n))
    filt = np.concatenate([q[0] for q in parts])
    raw = np.concatenate([q[1] for q in parts])
    nd = np.concatenate([q[2] for q in parts])
    con = np.concatenate([q[3] for q in parts])
    stats = {"dapi_tissue_thr": dapi_tissue_thr,
             "bg_18s": bg18, "bg_vim": bgvim,
             # QC only: DAPI contrast of Xenium's own nuclei (inside / 3 um ring)
             "nuc_contrast_quantiles": {f"{q:g}": float(v) for q, v in
                                        zip(np.arange(1, 51), np.percentile(con, np.arange(1, 51)))},
             "n_nuclei_contrast": int(con.size),
             "thr": float(np.percentile(filt, p.thr_percentile)),
             "thr_alt": float(np.percentile(raw, p.alt_thr_percentile)),
             "dapi_ref": float(np.median(nd)),
             # the full whole-sample distributions, so a percentile can be recalibrated without re-reading the image
             "filt_quantiles": {f"{q:g}": float(v) for q, v in zip(QGRID, np.percentile(filt, QGRID))},
             "raw_quantiles": {f"{q:g}": float(v) for q, v in zip(QGRID, np.percentile(raw, QGRID))},
             "n_positive_subsampled": int(filt.size)}
    print(f"pass 1 ({time.time() - t0:.0f}s): {stats}", flush=True)
    return stats, a.pct


def main():
    global TILE_PX, MARGIN_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--pct", type=float, nargs=4, metavar=("S18_P1", "S18_P99", "VIM_P1", "VIM_P99"),
                    help="whole-sample channel percentiles; estimated from the sample's tissue if omitted")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--tile", type=int, default=TILE_PX)
    ap.add_argument("--margin", type=int, default=MARGIN_PX)
    ap.add_argument("--stats-only", action="store_true", help="compute the whole-sample statistics, write stats.json, stop")
    ap.add_argument("--stats", type=Path, help="stats.json from an earlier --stats-only run: reuse instead of recomputing")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="override one rules.Params field (tuning runs; repeatable, value parsed as JSON)")
    ap.add_argument("--density-bg", type=float, help="background anchor of the density signal (transcripts / um^2), "
                    "replacing the whole-sample tissue median; must be > 0 (0 makes rules.py fall back to a crop median)")
    ap.add_argument("--combined", help="18S + density: 'bg18,hi18,cut18,bgd,hid,cutd' (see CombinedStack); needs --density; sets "
                    "the signal to [0,1] units (pct 0,1; background 0.001) and main cut 0.5 unless overridden with --param")
    ap.add_argument("--stat", action="append", default=[], metavar="KEY=VALUE", help="override one whole-sample stat (e.g. thr_alt=0.6)")
    ap.add_argument("--v2", help="phase 1 segmentation (seg_v2.py): JSON of its settings, must include {\"density\": <density zarr>}")
    ap.add_argument("--density-tissue-thr", type=float, help="tissue = density above this (transcripts / um^2), replacing the DAPI tissue mask")
    ap.add_argument("--density", help="transcript-density zarr (make_density_image.py): used in place of 18S, Vimentin zeroed")
    a = ap.parse_args()
    if a.density_tissue_thr is not None:
        os.environ[TISSUE_ENV] = str(a.density_tissue_thr)
    if a.v2:
        os.environ["ATERA_V2"] = a.v2
    if a.combined:
        os.environ[COMBINED_ENV] = a.combined
    if a.density:
        os.environ[DENSITY_ENV] = a.density  # before any worker pool starts, so every worker opens the same stack
    else:
        os.environ.pop(DENSITY_ENV, None)
    TILE_PX, MARGIN_PX = a.tile, a.margin
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    overrides = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v)
    p = R.Params(**overrides)
    if overrides:
        print(f"Params overrides: {overrides}", flush=True)

    z = open_morphology(a.bundle)
    H, W = z.shape[-2:]
    x0, y0, x1, y1 = a.window or (0, 0, W, H)
    nb = pd.read_parquet(a.bundle / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    cb = pd.read_parquet(a.bundle / "cell_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    for df in (nb, cb):
        df["cell_id"] = df["cell_id"].astype(str)
    if a.stats_only:
        tx = pd.DataFrame(columns=["x_location", "y_location", "feature_name"])
    else:
        # a --window run only ever reads transcripts within the window plus the tile margin
        bbox = None if a.window is None else tuple(v * R.UM_PER_PX for v in
                                                   (max(0, x0 - MARGIN_PX), max(0, y0 - MARGIN_PX), min(W, x1 + MARGIN_PX), min(H, y1 + MARGIN_PX)))
        tx = read_qc_transcripts(a.bundle / "transcripts.parquet", bbox)
    nuc_cent, cell_cent = centroids(nb), centroids(cb)
    # Some Xenium copies carry circle placeholders instead of a real nucleus segmentation (every polygon 26
    # vertices, near-constant radius); the method needs real nuclei, so refuse to run on those.
    nv = nb.groupby("cell_id").size()
    if (nv == 26).mean() > 0.9:
        sys.exit(f"{a.sample}: nucleus_boundaries look like circle placeholders ({(nv == 26).mean():.0%} of "
                 f"nuclei have exactly 26 vertices); use a bundle with the real nucleus segmentation")
    cores = list(tiles(x0, y0, x1, y1))
    print(f"{a.sample}: {W}x{H} px, window {x0},{y0}-{x1},{y1}, {len(cores)} tiles, "
          f"{len(nb.cell_id.unique())} nuclei, {len(tx):,} QC transcripts", flush=True)

    if a.stats is not None:
        prev = json.load(open(a.stats))
        stats, a.pct = prev["stats"], prev["pct"]
        print(f"whole-sample statistics from {a.stats}: {stats}, pct {a.pct}", flush=True)
    else:
        stats, a.pct = pass1(a, p, nb, cb, tx, nuc_cent, cell_cent, cores, t0)
    if a.combined:
        a.pct = [0.0, 1.0, 0.0, 1.0]        # the combined channel is already in whole-sample-normalized units
        stats["bg_18s"], stats["bg_vim"] = 1e-3, 0.0
        print("combined 18S + density signal: pct (0,1), background 0.001", flush=True)
    for kv in a.stat:
        k, v = kv.split("=", 1); stats[k] = float(v); print(f"stat override {k} = {v}", flush=True)
    if a.density_bg is not None:
        assert a.density_bg > 0, "--density-bg must be > 0"
        print(f"density background anchor {stats.get('bg_18s')} -> {a.density_bg}", flush=True)
        stats["bg_18s"] = a.density_bg
    # record the thresholds at the current Params percentiles (segment_region reads them from the same tables)
    for key, table, q in (("thr", "filt_quantiles", p.thr_percentile), ("thr_alt", "raw_quantiles", p.alt_thr_percentile)):
        if f"{q:g}" in stats.get(table, {}):
            stats[key] = stats[table][f"{q:g}"]
    if a.stats_only:
        json.dump({"sample": a.sample, "stats": stats, "pct": a.pct, "params": asdict(p), "window": [x0, y0, x1, y1],
                   "runtime_s": round(time.time() - t0)}, open(a.out / "stats.json", "w"), indent=1)
        print(f"stats only: written {a.out / 'stats.json'}", flush=True)
        return


    labels = np.lib.format.open_memmap(a.out / "final_labels.npy", mode="w+", dtype=np.uint32, shape=(H, W))
    rows, rescued_rows, nxt, n_resc = [], [], 0, 0
    init = (str(a.bundle), tuple(a.pct), stats, p, nb, cb, tx, nuc_cent, cell_cent, TILE_PX, MARGIN_PX)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=init) as ex:
        for i, (core, cells) in enumerate(ex.map(run_tile, cores), start=1):
            for c in cells:
                m = np.unpackbits(c["mask"])[: c["mshape"][0] * c["mshape"][1]].reshape(c["mshape"]).astype(bool)
                sl = (slice(c["r0"], c["r0"] + m.shape[0]), slice(c["c0"], c["c0"] + m.shape[1]))
                m &= labels[sl] == 0  # a pixel already taken by a cell from a neighbouring tile stays there
                if not m.any():
                    continue
                nxt += 1
                labels[sl][m] = nxt
                nid = c["nucleus_id"]
                if c["rescued"]:
                    n_resc += 1
                    nid = f"dapi_rescue_{n_resc}"
                    if c["nuc_poly"] is not None:
                        rescued_rows += [(nid, float(x), float(y)) for x, y in c["nuc_poly"]]
                rows.append({"label": nxt, "source": c["source"], "nucleus_id": nid, "has_nucleus": bool(nid),
                             "area_um2": round(c["area_um2"], 2), "x_centroid": c["cx_um"], "y_centroid": c["cy_um"]})
            if i % 10 == 0 or i == len(cores):
                print(f"  tile {i}/{len(cores)}: {nxt} cells ({time.time() - t0:.0f}s)", flush=True)
    labels.flush()
    cells_df = pd.DataFrame(rows)
    # seam guard: if two tiles gave the same Xenium nucleus to different cells, the larger cell keeps it
    dup = cells_df[cells_df.has_nucleus].sort_values("area_um2", ascending=False).duplicated("nucleus_id")
    lose = dup[dup].index
    cells_df.loc[lose, ["nucleus_id", "has_nucleus"]] = ["", False]
    print(f"  nucleus claimed by two cells at tile seams: {len(lose)}", flush=True)
    cells_df.to_parquet(a.out / "cells.parquet", index=False)
    pd.DataFrame(rescued_rows, columns=["cell_id", "vertex_x", "vertex_y"]).to_parquet(
        a.out / "rescued_nuclei.parquet", index=False)
    json.dump({"sample": a.sample, "stats": stats, "pct": a.pct, "params": asdict(p), "tile_px": TILE_PX,
               "margin_px": MARGIN_PX, "window": [x0, y0, x1, y1], "n_cells": nxt, "n_rescued": n_resc,
               "runtime_s": round(time.time() - t0)}, open(a.out / "stats.json", "w"), indent=1)
    print(f"done: {nxt} cells, {n_resc} rescued nuclei, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
