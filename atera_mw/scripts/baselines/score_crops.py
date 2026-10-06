#!/usr/bin/env python3
"""Score every method x crop with the paper benchmark's metrics, as scg/benchmark_fullsample.py does.

Transcripts = the crop window's QC transcripts (is_gene & qv >= 20, [x0 * UM, x1 * UM) x [y0 * UM, y1 * UM), the
scoring set of atera_reseg/compare_crop.py); nucleus reference = the bundle's overlaps_nucleus; each method's
transcript -> cell assignment (<method>/<crop>/assignment.parquet) joined on transcript_id and turned into integer
codes as benchmark_fullsample.cell_codes (UNASSIGNED / 0 / -1 / "" / nan -> 0). Metrics: splitmerge.metrics.score
unchanged, plus mecr_downsampled(labels, genes, n=1000) (mecr_n1000) and n_cells_ge1000_tx, as compare_crop.py.
10x's own assignment (bundle cell_id) is scored alongside as "xenium_original".

Output: <work>/scores/<method>_<crop>.json and <work>/scores/all.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
from splitmerge.metrics import score, mecr_downsampled  # noqa: E402  (unchanged)

METHODS = ["voronoi", "watershed", "cellpose", "baysor", "bidcell", "segger"]
NONE = ["UNASSIGNED", "0", "-1", "", "nan", "None"]


def codes_from(cell: pd.Series) -> np.ndarray:
    s = cell.astype(str)
    none = cell.isna() | s.isin(NONE)
    c, _ = pd.factorize(s.where(~none))
    return np.where(none.to_numpy(), 0, c + 1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    a = ap.parse_args()
    outdir = A.WORK / "scores"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for crop in a.crops or A.crop_names():
        win = A.crop_window(crop)
        tx = A.scoring_tx(win, ["transcript_id", "cell_id", "feature_name", "x_location", "y_location",
                                "overlaps_nucleus"])
        ids = tx.transcript_id.to_numpy()
        genes = tx.feature_name.astype(str).to_numpy()
        xy = tx[["x_location", "y_location"]].to_numpy()
        on_nuc = tx.overlaps_nucleus.astype(bool).to_numpy()
        versions = {"xenium_original": codes_from(tx.cell_id)}
        for m in a.methods:
            f = A.WORK / m / crop / "assignment.parquet"
            if not f.exists():
                print(f"{crop} {m}: no assignment.parquet, skipped", flush=True)
                continue
            t = pd.read_parquet(f).set_index("transcript_id")["cell_id"]
            versions[m] = codes_from(t.reindex(ids))
        for name, lab in versions.items():
            r = {"crop": crop, "method": name, "n_qc_transcripts": int(len(ids)), **score(name, lab, genes, on_nuc, xy)}
            r["mecr_n1000"] = round(mecr_downsampled(lab, genes, n=1000), 4)
            r["n_cells_ge1000_tx"] = int((pd.Series(lab[lab > 0]).value_counts() >= 1000).sum())
            ri = A.WORK / name / crop / "run_info.json"
            if ri.exists():
                info = json.load(open(ri))
                rt = info.get("runtime_s")
                r["runtime_s"] = rt.get("total_s") if isinstance(rt, dict) else rt
            json.dump(r, open(outdir / f"{name}_{crop}.json", "w"), indent=1, default=float)
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "all.csv", index=False)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
