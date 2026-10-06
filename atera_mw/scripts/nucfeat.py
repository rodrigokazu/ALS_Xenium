#!/usr/bin/env python3
"""Image features of 10x nuclei (DAPI, 18S, membrane), for testing whether the image alone tells cell types apart.

A random 120,000 of the section's nuclei (seed 0), processed one 4096 px tile at a time (the image is read once per tile).
Per nucleus: shape (area, perimeter, eccentricity, solidity, axes), DAPI (mean, sd, CV, p10/p50/p90, bright-spot fraction =
pixels > 1.5 x nucleus median, edge / centre ratio, local texture = mean |Laplacian|), 18S and membrane in the nucleus and in
rings 0-2 um and 2-4 um outside it, crowding (nuclei within 10 um, distance to the nearest nucleus), and the final-segmentation
cell label under the nucleus centroid (to join the transcript annotation later). Image values are raw counts; every model
compares nuclei across the whole section, so no per-tile normalisation is applied.
python nucfeat.py [--workers 4]
"""
import argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, tifffile, zarr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_poly
from skimage.measure import regionprops

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); B = SSD / "Cerebellum_sample"; O = SSD / "segmentation_full/nucleus_model"; T = 4096; PAD = 64

def tile_job(args):
    (tx, ty), sub = args
    X0, Y0 = max(tx * T - PAD, 0), max(ty * T - PAD, 0); X1, Y1 = (tx + 1) * T + PAD, (ty + 1) * T + PAD
    z = zarr.open(tifffile.TiffFile(B / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
    img = np.asarray(z[:3, Y0:Y1, X0:X1]).astype(np.float32); dapi, mem, r18 = img[0], img[1], img[2]
    lap = np.abs(ndi.laplace(ndi.gaussian_filter(dapi, 1.0)))
    final = np.asarray(zarr.open_array(str(SSD / "segmentation_full/final_labels.zarr"), mode="r")[Y0:Y1, X0:X1])
    H, W = dapi.shape; rows = []
    for cid, g in sub.groupby("cell_id", sort=False):
        yy, xx = g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0
        r0, r1, c0, c1 = int(yy.min()) - 25, int(yy.max()) + 26, int(xx.min()) - 25, int(xx.max()) + 26
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W: continue
        m = np.zeros((r1 - r0, c1 - c0), bool); rr, cc = draw_poly(yy - r0, xx - c0, shape=m.shape); m[rr, cc] = True
        if m.sum() < 20: continue
        sl = (slice(r0, r1), slice(c0, c1)); d = dapi[sl][m]
        dt_out = ndi.distance_transform_edt(~m) * UM; dt_in = ndi.distance_transform_edt(m) * UM
        ring1, ring2 = (dt_out > 0) & (dt_out <= 2), (dt_out > 2) & (dt_out <= 4)
        core, edge = m & (dt_in > dt_in.max() * 0.5), m & (dt_in <= 1.0)
        p = regionprops(m.astype(np.uint8))[0]; med = float(np.median(d))
        cy, cx = p.centroid
        rows.append(dict(nucleus_id=cid, x_um=(c0 + cx + X0) * UM, y_um=(r0 + cy + Y0) * UM,
            area_um2=p.area * UM ** 2, perimeter_um=p.perimeter * UM, eccentricity=p.eccentricity, solidity=p.solidity,
            major_um=p.major_axis_length * UM, minor_um=p.minor_axis_length * UM,
            dapi_mean=float(d.mean()), dapi_sd=float(d.std()), dapi_cv=float(d.std() / max(d.mean(), 1)), dapi_p10=float(np.percentile(d, 10)),
            dapi_p50=med, dapi_p90=float(np.percentile(d, 90)), dapi_bright_frac=float((d > 1.5 * med).mean()),
            dapi_edge_centre=float(dapi[sl][edge].mean() / max(dapi[sl][core].mean(), 1)) if core.any() and edge.any() else np.nan,
            dapi_texture=float(lap[sl][m].mean() / max(d.mean(), 1)),
            s18_nuc=float(r18[sl][m].mean()), s18_ring1=float(r18[sl][ring1].mean()), s18_ring2=float(r18[sl][ring2].mean()),
            mem_nuc=float(mem[sl][m].mean()), mem_ring1=float(mem[sl][ring1].mean()), mem_ring2=float(mem[sl][ring2].mean()),
            dapi_ring1=float(dapi[sl][ring1].mean()),
            final_label=int(final[int(r0 + cy), int(c0 + cx)])))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(); O.mkdir(parents=True, exist_ok=True); t0 = time.time()
    nb = pd.read_parquet(B / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    cent = nb.groupby("cell_id").agg(x=("vertex_x", "mean"), y=("vertex_y", "mean"))
    tree = cKDTree(cent.to_numpy()); nn_d, _ = tree.query(cent.to_numpy(), k=2); crowd = np.array([len(v) - 1 for v in tree.query_ball_point(cent.to_numpy(), 10.0)])
    cent["nn_dist_um"], cent["n_within_10um"] = nn_d[:, 1], crowd
    pick = cent.sample(min(120_000, len(cent)), random_state=0)
    pick["tx"], pick["ty"] = (pick.x / UM // T).astype(int), (pick.y / UM // T).astype(int)
    nbp = nb[nb.cell_id.isin(pick.index)].merge(pick[["tx", "ty"]], left_on="cell_id", right_index=True)
    jobs = [((tx, ty), g.drop(columns=["tx", "ty"])) for (tx, ty), g in nbp.groupby(["tx", "ty"])]
    print(len(pick), "nuclei in", len(jobs), "tiles", flush=True); out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(tile_job, jobs), start=1):
            out += r
            if i % 20 == 0: print(f"tile {i}/{len(jobs)} nuclei {len(out)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out).merge(cent[["nn_dist_um", "n_within_10um"]], left_on="nucleus_id", right_index=True)
    df.to_parquet(O / "nucleus_features.parquet", index=False); print("done", len(df), f"{time.time() - t0:.0f}s")
