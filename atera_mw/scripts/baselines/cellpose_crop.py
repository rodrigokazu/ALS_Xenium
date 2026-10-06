#!/usr/bin/env python3
"""Cellpose baseline on one Atera crop: the per-tile step of paper_reseg_pipeline/scg/baselines/cellpose_fullsample.py on
the crop's tile(s), with the whole-sample normalization.

As in the whole-sample run: cellpose 4.2.1.1, CellposeModel(gpu=True) default weights (cpsam_v2), input
np.stack([boundary stain ch0001, DAPI ch0000], -1) normalized with normalize99's arithmetic (cellpose_fullsample.norm99,
imported) at the WHOLE-SAMPLE p1 / p99 of each channel (stats_cellpose_norm.py), eval(channels=[1, 2],
diameter=None, normalize=False), every other argument at its default; tile = core + 800 px margin; a tile keeps the
cells whose pixel centroid lies in its core (common.cells_from_labels: labels_to_polygons + pixel-centre
rasterization, sequential codes, later label wins).
Crop changes only: the cores are the crop window +- 25 um (one 4096-px tile), and the label image is kept for the
tile window only, then cut to the crop window.

Three stages, because the local cellpose_env holds cellpose only (no zarr / matplotlib / skimage / pyarrow, which
the benchmark's common.py needs); the SCG cellpose_env has them, so --stage all runs everything in one process:
  prep     (spatial env)    read + normalize the tile image(s)           -> <out>/_tiles.npz
  segment  (cellpose_env)   model.eval on each tile                      -> <out>/_masks.npz
  finish   (spatial env)    loader, paste, crop outputs
Usage: python cellpose_crop.py --crop spot1_purkinje_left --stage prep|segment|finish|all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def stage_prep(a, out):
    import atera_common as A
    A.paper_path()
    import common as C
    import cellpose_fullsample as CP  # unchanged: norm99, channel indices
    win = A.crop_window(a.crop)
    nr = json.load(open(a.norm_json or A.WORK / "_stats" / "cellpose_norm.json"))
    z = C.open_morphology(A.BUNDLE)
    H, W = z.shape[-2:]
    cores = C.tiles(*A.padded_core(win, H, W))
    arrs, meta = {}, []
    for i, core in enumerate(cores):
        wx0, wy0, wx1, wy1 = C.window(core, (H, W))
        b = np.asarray(z[CP.CH_BOUNDARY, wy0:wy1, wx0:wx1])
        d = np.asarray(z[CP.CH_DAPI, wy0:wy1, wx0:wx1])
        if not (b.any() or d.any()):
            continue
        arrs[f"img{i}"] = np.stack([CP.norm99(b, nr["b_lo"], nr["b_hi"]), CP.norm99(d, nr["d_lo"], nr["d_hi"])], axis=-1)
        meta.append({"i": i, "core": list(core), "win": [wx0, wy0, wx1, wy1]})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "_tiles.npz", **arrs)
    json.dump({"crop": a.crop, "window_px": list(win), "H": int(H), "W": int(W), "tiles": meta, "norm": nr,
               "t0": time.time()}, open(out / "_tiles.json", "w"), indent=1)
    print(f"prep: {len(meta)} tile(s) {meta}", flush=True)


def stage_segment(a, out):
    import torch
    from cellpose import models, version as cp_version
    meta = json.load(open(out / "_tiles.json"))
    tiles = np.load(out / "_tiles.npz")
    model = models.CellposeModel(gpu=True)
    print(f"cellpose {cp_version}, torch {torch.__version__}, model {model.pretrained_model} on {model.device}",
          flush=True)
    masks, t_net = {}, 0.0
    for t in meta["tiles"]:
        t1 = time.time()
        m, _, _ = model.eval(tiles[f"img{t['i']}"], channels=[1, 2], diameter=None, normalize=False)
        t_net += time.time() - t1
        masks[f"m{t['i']}"] = m.astype(np.int32)
        print(f"  tile {t['core']}: {int(m.max())} cellpose cells in window (net {t_net:.0f}s)", flush=True)
    np.savez(out / "_masks.npz", **masks)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    json.dump({"version": f"cellpose {cp_version}, torch {torch.__version__}", "model": str(model.pretrained_model),
               "device": str(model.device), "gpu_name": gpu, "net_seconds": round(t_net),
               "cellpose_models_path": os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")},
              open(out / "_segment.json", "w"), indent=1)


def stage_finish(a, out):
    import atera_common as A
    A.paper_path()
    import common as C
    meta = json.load(open(out / "_tiles.json"))
    seg = json.load(open(out / "_segment.json"))
    masks = np.load(out / "_masks.npz")
    win = tuple(meta["window_px"])
    x0, y0, x1, y1 = win
    H, W = meta["H"], meta["W"]
    fx0 = min(t["win"][0] for t in meta["tiles"])
    fy0 = min(t["win"][1] for t in meta["tiles"])
    fx1 = max(t["win"][2] for t in meta["tiles"])
    fy1 = max(t["win"][3] for t in meta["tiles"])
    local = np.zeros((fy1 - fy0, fx1 - fx0), dtype=np.int64)
    n = n_native = 0
    for t in meta["tiles"]:
        m = masks[f"m{t['i']}"]
        n_native += int(m.max())
        for c in C.cells_from_labels(m, t["win"][0], t["win"][1], tuple(t["core"]), H, W):
            n += 1
            A.paste_local(local, fy0, fx0, c, n)
    labels_win = local[y0 - fy0:y1 - fy0, x0 - fx0:x1 - fx0].astype(np.uint32)
    ids = np.array([f"cp-{k}" for k in range(1, n + 1)])
    nr = meta["norm"]
    info = {"tool": "cellpose", "crop": a.crop, "bundle": str(A.BUNDLE),
            "reference_driver": "paper_reseg_pipeline/scg/baselines/cellpose_fullsample.py (norm99 imported)",
            "version": seg["version"],
            "parameters": {"model": seg["model"], "input": "[ch0001 boundary stain, ch0000 DAPI]",
                           "eval": {"channels": [1, 2], "diameter": None, "other": "defaults"},
                           "normalization": {"boundary_p1_p99": [nr["b_lo"], nr["b_hi"]],
                                             "dapi_p1_p99": [nr["d_lo"], nr["d_hi"]], "source": nr.get("source")},
                           "tile_px": C.TILE_PX, "margin_px": C.MARGIN_PX, "cores": [t["core"] for t in meta["tiles"]],
                           "loader": "labels_to_polygons(min_px=15, n_verts=40) + pixel-centre rasterization, "
                                     "later label wins"},
            "device": seg["device"], "gpu_name": seg["gpu_name"], "net_seconds": seg["net_seconds"],
            "n_native_cellpose_cells": n_native, "n_kept": n, "cellpose_models_path": seg["cellpose_models_path"]}
    A.write_crop_outputs(out, win, labels_win, ids, info, meta["t0"])
    for f in ("_tiles.npz", "_masks.npz"):
        (out / f).unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--stage", choices=["prep", "segment", "finish", "all"], default="all")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--norm-json", type=Path)
    a = ap.parse_args()
    import atera_common as A   # numpy / pandas only at import time
    out = a.out or A.WORK / "cellpose" / a.crop
    for s in (["prep", "segment", "finish"] if a.stage == "all" else [a.stage]):
        {"prep": stage_prep, "segment": stage_segment, "finish": stage_finish}[s](a, out)


if __name__ == "__main__":
    main()
