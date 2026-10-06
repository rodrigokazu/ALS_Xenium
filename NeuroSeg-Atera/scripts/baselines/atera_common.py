"""Shared pieces of the Atera cerebellum crop runs of the paper's six baselines (see README.md).

The benchmark's own modules are imported unchanged by path (paper_reseg_pipeline/scg/baselines/common.py and the
method drivers); only the input/output around them is crop-sized here:
  * a crop = one line of crops_marcel.tsv (level-0 px window, 400 x 400 um)
  * every method sees the crop +- PAD_UM of context (its "core"); image methods additionally get the benchmark's
    own 800 px tile margin around that core, exactly as a tile of the whole-sample run
  * transcripts are read from the 47 GB bundle with pyarrow filters (x / y window), never whole
  * outputs are cut to the crop window: labels_window.npy (uint32, level-0 px of the window) and
    assignment.parquet (transcript_id, cell_id for every QC transcript of the scoring window)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

UM = 0.2125
QV_MIN = 20
PAD_UM = 25.0
PAD_PX = int(np.ceil(PAD_UM / UM))            # 118 px
# SCG defaults; ATERA_LOCAL=1 (or the ATERA_* variables) switches to the local Mac layout
SPATIAL_ROOT = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial")
_LOCAL = os.environ.get("ATERA_LOCAL") == "1"
_SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
_REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("ATERA_BUNDLE", _SSD / "work" / "local_bundle" if _LOCAL
                             else SPATIAL_ROOT / "Atera_CB" / "Cerebellum_sample"))
WORK = Path(os.environ.get("ATERA_WORK", "/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/"
                           "49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local/baselines" if _LOCAL
                           else SPATIAL_ROOT / "_work_reseg_atera" / "baselines"))
CROPS_TSV = Path(os.environ.get("ATERA_CROPS", _REPO / "atera_reseg" / "crops_marcel.tsv" if _LOCAL
                                else "/home/mw28/spatial_dapi_pipeline/atera_reseg/crops_marcel.tsv"))
PAPER = Path(os.environ.get("ATERA_PAPER", _REPO / "paper_reseg_pipeline" if _LOCAL
                            else "/home/mw28/spatial_dapi_pipeline/paper_reseg_pipeline"))
WORKERS = int(os.environ.get("ATERA_WORKERS", "6"))   # cap on cores (local: the option B run needs the rest)
PAPER_BASELINES = PAPER / "scg" / "baselines"


def paper_path():
    """Put the benchmark's baselines dir and paper root on sys.path (read-only imports)."""
    for p in (str(PAPER_BASELINES), str(PAPER)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # ATERA_MARGIN_PX (unset = benchmark's 800): tile margin override for windows whose local transcripts cover less
    # context (500 um runs: 282 px margin + 118 px core pad = 400 px). common.window() binds its default at def time,
    # so patch both the constant and the default.
    if os.environ.get("ATERA_MARGIN_PX"):
        import common as _C
        m = int(os.environ["ATERA_MARGIN_PX"])
        _C.MARGIN_PX = m
        _C.window.__defaults__ = (m,)


def crop_window(name: str, tsv: Path = CROPS_TSV):
    for line in open(tsv):
        f = line.split()
        if f and f[0] == name:
            return tuple(int(v) for v in f[1:5])
    raise KeyError(f"{name} not in {tsv}")


def crop_names(tsv: Path = CROPS_TSV):
    return [line.split()[0] for line in open(tsv) if line.strip()]


def padded_core(win, H, W, pad=PAD_PX):
    x0, y0, x1, y1 = win
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def image_shape(bundle: Path = BUNDLE):
    import tifffile
    return tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif").series[0].shape[-2:]


def read_tx(x0_um, y0_um, x1_um, y1_um, columns, qc=True, bundle: Path = BUNDLE, right_closed=False) -> pd.DataFrame:
    """Transcripts in [x0, x1) x [y0, y1) um (closed on the right with right_closed) from the bundle, pyarrow
    filter pushdown; qc = is_gene & qv >= 20 (the benchmark's QC)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pds
    lt = (lambda f, v: f <= v) if right_closed else (lambda f, v: f < v)
    f = ((pc.field("x_location") >= x0_um) & lt(pc.field("x_location"), x1_um)
         & (pc.field("y_location") >= y0_um) & lt(pc.field("y_location"), y1_um))
    if qc:
        f = f & (pc.field("qv") >= QV_MIN) & (pc.field("is_gene") == True)  # noqa: E712
    cols = list(dict.fromkeys(columns))
    return pds.dataset(bundle / "transcripts.parquet").to_table(columns=cols, filter=f).to_pandas()


def scoring_tx(win, columns=("transcript_id", "x_location", "y_location")) -> pd.DataFrame:
    """The QC transcripts a crop is scored on: [x0 * UM, x1 * UM) x [y0 * UM, y1 * UM), as atera_reseg/compare_crop.py."""
    x0, y0, x1, y1 = win
    return read_tx(x0 * UM, y0 * UM, x1 * UM, y1 * UM, columns, qc=True)


def window_lookup(labels_win: np.ndarray, win, x_um, y_um):
    """Label under each transcript: level-0 pixel floor(um / 0.2125) (the benchmark's lookup), clipped to the
    window for the float32 rounding at its edge (as compare_crop.py)."""
    x0, y0, x1, y1 = win
    col = np.clip(np.floor(np.asarray(x_um) / UM).astype(np.int64) - x0, 0, labels_win.shape[1] - 1)
    row = np.clip(np.floor(np.asarray(y_um) / UM).astype(np.int64) - y0, 0, labels_win.shape[0] - 1)
    return labels_win[row, col]


def paste_local(local: np.ndarray, R0: int, C0: int, c: dict, code: int):
    """common.paste (later / higher code wins) into a local array whose global top-left is (R0, C0)."""
    paper_path()
    import common as C
    m = C.unpack(c)
    r0, c0 = c["r0"] - R0, c["c0"] - C0
    ra, ca = max(0, r0), max(0, c0)
    rb, cb = min(local.shape[0], r0 + m.shape[0]), min(local.shape[1], c0 + m.shape[1])
    if rb <= ra or cb <= ca:
        return
    mm = m[ra - r0:rb - r0, ca - c0:cb - c0]
    sub = local[ra:rb, ca:cb]
    sub[mm] = np.maximum(sub[mm], code)


def write_crop_outputs(out: Path, win, labels_win: np.ndarray | None, code_to_id: np.ndarray, info: dict, t0: float,
                       tx_codes: np.ndarray | None = None, tx: pd.DataFrame | None = None,
                       areas_um2: dict | None = None):
    """labels_window.npy + assignment.parquet (+ cells.parquet, run_info.json).
    code_to_id[k - 1] = cell id of label code k. Either a label image (transcripts looked up in it) or per-transcript
    codes for `tx` (scoring transcripts)."""
    out.mkdir(parents=True, exist_ok=True)
    if tx is None:
        tx = scoring_tx(win)
    if tx_codes is None:
        tx_codes = window_lookup(labels_win, win, tx.x_location.to_numpy(), tx.y_location.to_numpy()).astype(np.int64)
    names = np.concatenate([["UNASSIGNED"], np.asarray(code_to_id, dtype=object).astype(str)])
    pd.DataFrame({"transcript_id": tx.transcript_id.to_numpy(), "cell_id": names[tx_codes]}).to_parquet(
        out / "assignment.parquet", index=False)
    if labels_win is not None:
        np.save(out / "labels_window.npy", labels_win.astype(np.uint32))
        cnt = np.bincount(labels_win.ravel().astype(np.int64), minlength=len(code_to_id) + 1)
        vis = np.flatnonzero(cnt[1:] > 0) + 1
        cells = pd.DataFrame({"cell_id": names[vis], "visible_area_um2_in_window": cnt[vis] * UM * UM})
    else:
        cells = pd.DataFrame({"cell_id": list(areas_um2.keys()) if areas_um2 else [],
                              "cell_area": list(areas_um2.values()) if areas_um2 else []})
    cells.to_parquet(out / "cells.parquet", index=False)
    info = {**info, "window_px": list(win), "pad_um": PAD_UM, "n_scoring_transcripts": int(len(tx)),
            "pct_scoring_transcripts_assigned": round(100 * float((tx_codes > 0).mean()), 2) if len(tx) else None,
            "n_cells_in_window": int(len(np.unique(tx_codes[tx_codes > 0]))),
            "runtime_s": round(time.time() - t0, 1), "host": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
    json.dump(info, open(out / "run_info.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k not in ("parameters",)}, default=str), flush=True)
    return info
