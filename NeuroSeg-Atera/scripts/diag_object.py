"""Follow one object through segment_region (combined signal): which stage shrinks / splits / rejects it.
python diag_object.py <cx_um> <cy_um> [half_um]    absolute slide um of the object's centre"""
import os, sys, json
os.environ.update({"ATERA_DENSITY_ZARR": "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/work/density/tx_density.zarr",
                   "ATERA_DENSITY_TISSUE_THR": "10", "ATERA_COMBINED": "48,941,0.459,5.815,177.18,0.793"})
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1] / "paper_reseg_pipeline"))
import run_fullsample as RF
from splitmerge import rules as R
cx, cy = float(sys.argv[1]), float(sys.argv[2]); half = float(sys.argv[3]) if len(sys.argv) > 3 else 70
B = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/work/local_bundle")
W = "/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local"
st = json.load(open(f"{W}/stats_density/stats.json"))["stats"]
st["bg_18s"], st["bg_vim"], st["thr_alt"] = 1e-3, 0.0, 0.6
p = R.Params(thr_value=0.5, glia_soma_thr_value=0.2, glia_soma_max_um=8)
z = RF.open_morphology(B)
fx0, fy0 = int((cx - half) / R.UM_PER_PX), int((cy - half) / R.UM_PER_PX); fx1, fy1 = int((cx + half) / R.UM_PER_PX), int((cy + half) / R.UM_PER_PX)
crop = np.asarray(z[:, fy0:fy1, fx0:fx1]).astype(np.float32)
nb = pd.read_parquet(B / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"]); cb = pd.read_parquet(B / "cell_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
for d in (nb, cb): d["cell_id"] = d["cell_id"].astype(str)
nb = RF._subset(nb, RF.centroids(nb), fx0, fy0, fx1, fy1); cb = RF._subset(cb, RF.centroids(cb), fx0, fy0, fx1, fy1)
tx = RF.read_qc_transcripts(B / "transcripts.parquet", (fx0 * R.UM_PER_PX, fy0 * R.UM_PER_PX, fx1 * R.UM_PER_PX, fy1 * R.UM_PER_PX))
rr = R.segment_region(crop, (0.0, 1.0, 0.0, 1.0), nb, cb, tx, fx0, fy0, p, dict(st))
cy0, cx0 = int((cy - fy0 * R.UM_PER_PX) / R.UM_PER_PX), int((cx - fx0 * R.UM_PER_PX) / R.UM_PER_PX)
r = int(14 / R.UM_PER_PX); sl = (slice(cy0 - r, cy0 + r), slice(cx0 - r, cx0 + r))
um2 = R.UM_PER_PX ** 2
def top(a, name, n=4):
    v = a[sl].ravel(); v = v[v > 0]
    u, c = np.unique(v, return_counts=True); o = np.argsort(c)[::-1][:n]
    print(f"{name:10s}", [(int(u[i]), round(c[i] * um2, 0)) for i in o], "(label, um2 inside the 28 um box)")
sig = crop[2][sl]; print(f"combined signal in box: p50 {np.percentile(sig,50):.2f} p90 {np.percentile(sig,90):.2f} max {sig.max():.2f}; above main cut 0.5: {100*(sig>0.5).mean():.0f}% of box")
print("raw 18S in box: p50/p90/max", np.percentile(crop[2][sl]*0+0,50))
for k in ("comp", "alt", "pieces", "merged", "keep18", "final"): top(rr[k], k)
nu = rr["nuc"][sl]; v = nu[nu > 0]
print("nuclei in box (label, area um2, is_neuron):", [(int(k), round((nu == k).sum() * um2, 0), bool(rr["is_neuron"][k])) for k in np.unique(v)])
t = rr["table"]; fin = np.unique(rr["pieces"][sl][rr["pieces"][sl] > 0])
print("piece table rows in box:"); print(t.loc[t.index.intersection(fin)].to_string(max_cols=14))
f = rr["flags"]; print("flags type:", type(f).__name__, "| comp labels flag:", {int(k): (f.get(k) if hasattr(f, 'get') else None) for k in np.unique(rr['comp'][sl][rr['comp'][sl] > 0])[:4]} if hasattr(f, 'get') else "")
cells = rr["cells"]; fl = np.unique(rr["final"][sl][rr["final"][sl] > 0]); print("final cells in box:"); print(cells.loc[cells.index.intersection(fl)].to_string())

# ---- what is the big cell? genes of its transcripts
g = pd.Series(rr["genes"]); tp = rr["tx_piece"]
big = int(max(rr["cells"].loc[rr["cells"].index.intersection(fl)].area_um2.idxmax(), 0))
sub = g[tp == big]; vc = sub.value_counts()
print(f"cell {big}: {len(sub):,} transcripts; Purkinje markers:", {m: int(vc.get(m, 0)) for m in ("CALB1", "PCP2", "PCP4", "CAR8", "ITPR1", "GRID2")},
      "| granule GABRA6:", int(vc.get("GABRA6", 0)), "| MBP:", int(vc.get("MBP", 0)), "| GFAP:", int(vc.get("GFAP", 0)))
print("top 10 genes:", vc.head(10).to_dict())
