#!/usr/bin/env python3
"""Whole-window benchmark scores (splitmerge.metrics.score unchanged + MECR at 1,000 tx) for several runs on the 500 um windows.
python win_score.py <out.csv> <run dir name>... (10x always included)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import score, mecr_downsampled
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WINS = ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]
rows = []
for w in WINS:
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]
    bundle = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas(strings_to_categorical=True)
    gc, gr = (tx.x_location.to_numpy() / UM).astype(int), (tx.y_location.to_numpy() / UM).astype(int)
    genes = tx.feature_name.astype(str).to_numpy(); on = tx.overlaps_nucleus.astype(bool).to_numpy(); xy = tx[["x_location", "y_location"]].to_numpy()
    cid = tx.cell_id.astype(str); labs = {"10x": (pd.factorize(cid.where(~cid.isin(["", "UNASSIGNED"]), None))[0] + 1).astype(np.int64)}
    for d in sys.argv[2:]:
        f = RUNS / d / w / "final_labels.npy"
        if f.exists(): labs[d.replace("crops_", "").replace("_500", "")] = np.asarray(np.load(f, mmap_mode="r")[gr, gc]).astype(np.int64)
    for k, lb in labs.items():
        s = score(k, lb, genes, on, xy); s["mecr_n1000"] = round(mecr_downsampled(lb, genes, n=1000), 4)
        rows.append({"window": w, "method": k, "n_tx": len(lb), **{c: s[c] for c in ("pct_assigned", "n_cells", "mecr", "mecr_n1000", "shell_coherence_median")}})
    print(w, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1], index=False)
df["assigned"] = df.pct_assigned * df.n_tx / 100
g = df.groupby("method"); out = pd.DataFrame({"pct_assigned_pooled": 100 * g.assigned.sum() / g.n_tx.sum(), "mecr_mean": g.mecr.mean(), "mecr1000_mean": g.mecr_n1000.mean(), "windows": g.size()})
for part, sel in (("tuning (spot1+2)", df.window.isin(WINS[:2])), ("held-out (spot3 + held1-6)", ~df.window.isin(WINS[:2]))):
    gg = df[sel].groupby("method"); print(part); print(pd.DataFrame({"pct": 100 * gg.assigned.sum() / gg.n_tx.sum(), "mecr": gg.mecr.mean(), "mecr1000": gg.mecr_n1000.mean()}).round(4).to_string())
