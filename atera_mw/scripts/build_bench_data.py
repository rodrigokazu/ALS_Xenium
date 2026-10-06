#!/usr/bin/env python3
"""Per window and method: everything the benchmark statistics need, computed once.
For each method: per-50um-block transcript counts (total QC, assigned), and per cell: block, size, marker presence (all transcripts; and
after downsampling the cell to 1000 transcripts, as splitmerge.metrics.mecr_downsampled does).
python build_bench_data.py <registry.json> <out.pkl>
registry: {"windows": {name: {"box": [x0,y0,x1,y1] px, "bundle": path}}, "methods": {window: {method: ["10x"] | ["npz", path] | ["npy", path]}}}"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import MARKERS

UM = 0.2125; BLOCK_UM = 50.0; N_DS = 1000; MIN_TX = 5
MG = [g for t, gs in MARKERS.items() for g in gs]; MT = [t for t, gs in MARKERS.items() for g in gs]
reg = json.load(open(sys.argv[1])); out = {}
for wn, w in reg["windows"].items():
    x0, y0, x1, y1 = w["box"]; H, W = y1 - y0, x1 - x0
    t = ds.dataset(Path(w["bundle"]) / "transcripts.parquet").to_table(
        columns=["x_location", "y_location", "feature_name", "qv", "cell_id"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20))
    xs, ys = t["x_location"].to_numpy(), t["y_location"].to_numpy()
    col = np.clip((xs / UM).astype(np.int64) - x0, 0, W - 1); row = np.clip((ys / UM).astype(np.int64) - y0, 0, H - 1)
    gm = pc.index_in(t["feature_name"].cast(pa.string()), value_set=pa.array(MG)).fill_null(-1).to_numpy().astype(np.int16)     # marker index or -1
    de = pc.dictionary_encode(t["cell_id"].cast(pa.string())).combine_chunks(); dv = de.dictionary.to_pylist()
    un = [i for i, v in enumerate(dv) if v in ("UNASSIGNED", "-1", "")]
    idx10 = de.indices.to_numpy(zero_copy_only=False).astype(np.int64) + 1
    if un: idx10[np.isin(idx10 - 1, un)] = 0
    nbx = int(np.ceil(W * UM / BLOCK_UM)); nby = int(np.ceil(H * UM / BLOCK_UM))
    blk = np.minimum((ys / BLOCK_UM).astype(np.int64) - int(y0 * UM // BLOCK_UM), nby), np.minimum((xs / BLOCK_UM).astype(np.int64) - int(x0 * UM // BLOCK_UM), nbx)
    blk = np.clip(blk[0], 0, nby) * (nbx + 1) + np.clip(blk[1], 0, nbx); nblk = (nby + 1) * (nbx + 1)
    rng = np.random.default_rng(0); u = rng.random(len(xs)).astype(np.float32)
    N = len(xs); print(f"{wn}: {N:,} QC transcripts", flush=True); out[wn] = {"n_blocks": nblk, "total_by_block": np.bincount(blk, minlength=nblk), "methods": {}}
    for mn, src in reg["methods"][wn].items():
        if src[0] == "10x": lab = idx10
        else:
            L = (np.load(src[1])["labels"] if src[0] == "npz" else np.asarray(np.load(src[1] + "/final_labels.npy", mmap_mode="r")[y0:y1, x0:x1]) if src[0] == "run" else np.load(src[1]))
            lab = L[row, col].astype(np.int64)
        a = lab > 0
        size = np.bincount(lab, minlength=int(lab.max()) + 1); size[0] = 0
        # per-cell marker presence from all transcripts
        sel = a & (gm >= 0)
        pres = np.zeros((len(size), len(MG)), bool); pres[lab[sel], gm[sel]] = True
        # per-cell block = block of its mean transcript position
        cx = np.bincount(lab[a], weights=xs[a], minlength=len(size)); cy = np.bincount(lab[a], weights=ys[a], minlength=len(size))
        with np.errstate(invalid="ignore", divide="ignore"):
            cb = (np.clip((cy / np.maximum(size, 1) / BLOCK_UM).astype(np.int64) - int(y0 * UM // BLOCK_UM), 0, nby) * (nbx + 1) + np.clip((cx / np.maximum(size, 1) / BLOCK_UM).astype(np.int64) - int(x0 * UM // BLOCK_UM), 0, nbx))
        # downsample every cell with >= N_DS transcripts to exactly N_DS (same random key for every method)
        big = size >= N_DS
        ai = np.flatnonzero(a & big[lab]); order = np.lexsort((u[ai], lab[ai])); ai = ai[order]
        ls = lab[ai]; start = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]; rank = np.arange(len(ls)) - np.repeat(start, np.diff(np.r_[start, len(ls)]))
        keep = ai[rank < N_DS]; ks = keep[gm[keep] >= 0]
        pds = np.zeros((len(size), len(MG)), bool); pds[lab[ks], gm[ks]] = True
        ids = np.flatnonzero(size >= MIN_TX)
        out[wn]["methods"][mn] = {"assigned_by_block": np.bincount(blk[a], minlength=nblk), "cell_block": cb[ids], "cell_size": size[ids],
                                  "pres": pres[ids], "pres_ds": pds[ids], "big": big[ids], "n_assigned": int(a.sum())}
        print(f"   {mn:22s} coverage {100 * a.mean():5.1f}%  cells>=5tx {len(ids):5d}  cells>=1000tx {int(big.sum()):5d}", flush=True)
pickle.dump({"markers": MG, "types": MT, "data": out}, open(sys.argv[2], "wb")); print("saved", sys.argv[2])
