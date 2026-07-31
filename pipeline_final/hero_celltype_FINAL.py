#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/hero_celltype_FINAL.py
#
# Close-up crops coloured by coarse cell type instead of by Novae domain. For each sample it
# finds the densest cluster of a showcase cell type and renders that field with the type
# highlighted and its canonical marker transcripts overlaid.
#
# Currently degraded, and honestly so. The _final cell and nucleus boundary files carry
# polygons for only about 10 to 13 percent of cells and their ids are still in the old gm
# namespace, so a filled-polygon Xenium-Explorer style figure is not renderable. Until Marcel
# re-exports full-coverage integer-id boundaries, this falls back to a cell-type-coloured
# centroid map, which is a valid figure rather than a broken one. The guard re-enables
# polygons automatically once the boundaries are fixed, with no code change.
#
# Region_2 has already been re-exported with integer ids at 98.6 percent coverage, so that one
# sample will produce real polygons.
# ========================================================================================

"""hero_celltype_FINAL.py -- Xenium hero close-ups coloured by COARSE CELL TYPE (not Novae
domain) for the MN-corrected _FINAL cohort. For each sample, finds the densest cluster of each
showcase cell type and renders a crop: cells coloured by cell type, the showcase type
highlighted, and that type's canonical marker transcript overlaid.

=====================================================================================
*** HOLD -- BLOCKED by F1 (do NOT expect filled-polygon "Xenium-Explorer" heroes yet). ***
The _final cell_boundaries.parquet / nucleus_boundaries.parquet are STALE / PARTIAL:
only ~10% of cells carry a polygon and the cell_id lives in the OLD gm namespace, so a
filled-polygon cell-type map is NOT scientifically renderable until Marcel re-exports
full-coverage integer-id boundaries on the imported segmentation. Until then this script
DEGRADES CLEANLY to a valid cell-type-coloured CENTROID map (a squidpy-style spatial plot)
over the real DAPI+18S image -- no crash, no fabricated polygons. The boundary guard
(xenium_overlay_FINAL.resolve_boundary_join) auto-RE-ENABLES the polygon path with ZERO
code change the moment the boundaries pass the alignment/coverage test.
=====================================================================================

F-review deltas vs CellTyping_coarse/code/hero_celltype.py:
  F5: GBASE = cfg.RANGER_FINAL; CT_CSV = cfg.COARSE_TYPING_DIR/obs_celltype.csv (no _gausss).
  F2/F1: boundaries are bridged via xo.resolve_boundary_join (which tries label_id(int) AND
      core_id(cell_id) and picks the best-aligned key), NOT the reference's blind
      core_id(cell_id) join. polys_ok gates polygon vs centroid rendering.
  Import the ALREADY-STAGED xenium_overlay_FINAL + nature_style from final_rerun_code.
  Bundle resolution uses cfg.discover_samples() (authoritative dedup label->dir), NOT a
      private resolve_bundle.
Env: xenium_vistools (pyarrow+tifffile+zarr). Read-only on bundles."""
import os, sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as cfg
import xenium_overlay_FINAL as xo   # ALREADY-STAGED (F2/F1): boundary guard + DAPI/18S/tx helpers
import nature_style as ns          # ALREADY-STAGED

GBASE = str(cfg.RANGER_FINAL)                                   # F5: Ranger_procd_mw_final
CT_CSV = f"{cfg.COARSE_TYPING_DIR}/obs_celltype.csv"           # F5: written by celltype_pipeline_FINAL

CT_COLOR = {
    "Oligodendrocyte":"#3B6FB6","OPC":"#55A868","Astrocyte":"#C44E52","Microglia":"#8172B3",
    "Macrophage_perivasc":"#8C6D5C","Neuron_excit":"#DA8BC3","Neuron_inhib":"#B79F3B",
    "MotorNeuron":"#E8000B","Endothelial":"#5FBFD6","Mural":"#7F7F7F","Fibroblast_VLMC":"#FF9F1C",
    "Lymphoid":"#2CA02C","LowQuality":"#D9D9D9",
}
MARKER = {  # main transcript per cell type
    "Oligodendrocyte":"MOG","OPC":"PDGFRA","Astrocyte":"AQP4","Microglia":"P2RY12",
    "Macrophage_perivasc":"CD163","Neuron_excit":"SLC17A6","Neuron_inhib":"GAD1",
    "MotorNeuron":"MNX1","Endothelial":"PECAM1","Fibroblast_VLMC":"DCN","Lymphoid":"PTPRC",
}
SHOWCASE = ["Oligodendrocyte","Astrocyte","Microglia","MotorNeuron","Neuron_excit","OPC","Endothelial"]
TXCOL = "#FFF200"


def build_bundle_map():
    """label -> bundle dir, from the authoritative discover_samples() dedup (F5). This
    already excludes the 2 superseded duplicate-donor originals and maps Region_2 ->
    SD05413_BG, so no private resolve_bundle / norm-matching is needed."""
    return {label: str(p) for label, p in cfg.discover_samples(verbose=False)}


def densest_center(xy, k=8):
    if len(xy) <= k: return xy.mean(0) if len(xy) else None
    tree=cKDTree(xy); d,_=tree.query(xy, k=k); i=np.argmin(d[:,k-1]); return xy[i]


def render(sample, bundle, ct_map, showtype, out, half=120.0):
    ns.apply_style()
    # cells.parquet -> integer core-id join to obs cell type (delta D2). Read only what we need.
    cells = pd.read_parquet(f"{bundle}/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    cells["core"] = cells["cell_id"].map(xo.core_id)
    cells["ct"] = cells["core"].map(ct_map)
    cells = cells[cells["ct"].notna()].copy()
    sub = cells[cells["ct"] == showtype]
    if len(sub) < 8:
        print(f"[{sample}] {showtype}: only {len(sub)} cells, skip"); return
    c = densest_center(sub[["x_centroid", "y_centroid"]].to_numpy())
    if c is None: return
    cx, cy = float(c[0]), float(c[1]); xa, xb, ya, yb = cx-half, cx+half, cy-half, cy+half

    # background: DAPI + 18S composite over the crop (helpers reused from xenium_overlay_FINAL)
    arrs = xo.load_dapi_levels(bundle)
    if not arrs:
        print(f"[{sample}] {showtype}: no DAPI image in bundle, skip"); return
    full = arrs[0]
    s18a = xo.load_18s_levels(bundle); s18 = s18a[0] if s18a else None
    Hpx, Wpx = full.shape[-2], full.shape[-1]
    X0, X1 = max(0, int(xa/xo.PX)), min(Wpx, int(xb/xo.PX))
    Y0, Y1 = max(0, int(ya/xo.PX)), min(Hpx, int(yb/xo.PX))
    rgb = xo.composite_dapi_18s(np.asarray(full[Y0:Y1, X0:X1]),
                                np.asarray(s18[Y0:Y1, X0:X1]) if s18 is not None else None)
    ext = [X0*xo.PX, X1*xo.PX, Y1*xo.PX, Y0*xo.PX]

    # F2/F1: only draw filled polygons if the boundary guard trusts the bundle's boundaries.
    cb, key, polys_ok, bdiag = xo.resolve_boundary_join(bundle, cells)

    fig, ax = plt.subplots(figsize=(7.6, 7.2)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.imshow(rgb, origin="upper", extent=ext, zorder=0)
    ctmap = cells.set_index("core")["ct"]

    if polys_ok and cb is not None:
        # ---- polygon path (auto-enabled once Marcel re-exports full-coverage boundaries) ----
        cb = cb.copy(); cb["ct"] = cb["k"].map(ctmap)
        incore = set(cells.loc[cells["x_centroid"].between(xa, xb) &
                               cells["y_centroid"].between(ya, yb), "core"])
        cbc = cb[cb["k"].isin(incore) & cb["ct"].notna()]
        polys, fcs, types = [], [], []
        for kk, s in cbc.groupby("k", sort=False):
            t = s["ct"].iloc[0]; v = s[["vertex_x", "vertex_y"]].to_numpy()
            if t is None or len(v) < 3: continue
            polys.append(v); fcs.append(CT_COLOR.get(t, "#888")); types.append(t)
        ax.add_collection(PolyCollection(polys, facecolors=fcs, edgecolors="white",
                                         linewidths=0.35, alpha=0.42, zorder=2))
        showp = [p for p, t in zip(polys, types) if t == showtype]
        if showp:
            ax.add_collection(PolyCollection(showp, facecolors="none",
                                             edgecolors=CT_COLOR[showtype], linewidths=2.0, zorder=5))
        present = [t for t in CT_COLOR if t in set(types)]
        n_shown = len(polys); mode = "polygons"
    else:
        # ---- CENTROID FALLBACK (expected on _final today; boundaries stale/partial) ----
        print(f"[{sample}] {showtype}: CENTROID FALLBACK -- boundaries unusable "
              f"({bdiag.get('key')}: median {bdiag.get('median_um', float('nan')):.0f}um, "
              f"cov {bdiag.get('coverage', 0.0):.2f}). See HOLD/F1 header.", flush=True)
        inc = cells[cells["x_centroid"].between(xa, xb) & cells["y_centroid"].between(ya, yb)]
        cvals = inc["ct"].astype(str).map(CT_COLOR).fillna("#888").tolist()
        ax.scatter(inc["x_centroid"].to_numpy(), inc["y_centroid"].to_numpy(), s=44, c=cvals,
                   marker="o", edgecolors="white", linewidths=0.25, alpha=0.78, zorder=2, rasterized=True)
        showc = inc[inc["ct"] == showtype]
        if len(showc):
            ax.scatter(showc["x_centroid"].to_numpy(), showc["y_centroid"].to_numpy(), s=170,
                       facecolors="none", edgecolors=CT_COLOR.get(showtype, "#888"),
                       linewidths=2.0, zorder=5)
        present = [t for t in CT_COLOR if t in set(inc["ct"].astype(str))]
        n_shown = len(inc); mode = "centroids (boundaries stale)"

    # marker transcript overlay (namespace-agnostic; unaffected by the boundary issue)
    gene = MARKER.get(showtype)
    if gene:
        xy = xo.load_transcripts(bundle, [gene]).get(gene)
        if xy is not None and len(xy):
            m = (xy[:, 0] >= xa) & (xy[:, 0] <= xb) & (xy[:, 1] >= ya) & (xy[:, 1] <= yb)
            xyc = xy[m]
            ax.scatter(xyc[:, 0], xyc[:, 1], s=22, c=TXCOL, alpha=0.95,
                       edgecolors="black", linewidths=0.4, zorder=7)
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([xa+half*0.12, xa+half*0.12+50], [yb-half*0.12, yb-half*0.12], color="white", lw=3, zorder=20)
    ax.text(xa+half*0.12+25, yb-half*0.12-half*0.06, "50 um", color="white", ha="center",
            va="bottom", fontsize=9, zorder=20)
    h = [Line2D([0],[0], marker='s', linestyle='none', markersize=8, markerfacecolor=CT_COLOR[t],
                markeredgecolor='none', label=t) for t in present]
    if gene:
        h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=8, markerfacecolor=TXCOL,
                        markeredgecolor='black', markeredgewidth=0.5, label=gene))
    lg = ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.005, 0.5), frameon=False,
                   labelcolor='white', fontsize=8, title='coarse cell type'); lg.get_title().set_color('white')
    ax.set_title(f"{sample}  |  {showtype} showcase  ({mode}; DAPI+18S; {gene} transcripts)",
                 color="white", fontsize=11)
    tag = showtype.replace("/", "-")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}/{sample}__celltype_hero_{tag}.{e}", dpi=430, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[{sample}] wrote celltype hero {showtype} ({mode}, {n_shown} cells) @ {cx:.0f},{cy:.0f}", flush=True)


def main(samples, out):
    os.makedirs(out, exist_ok=True)
    bmap = build_bundle_map()
    obs = pd.read_csv(CT_CSV)
    idxcol = obs.columns[0]                     # obs index = obs_names '<LABEL>:<int>'
    obs["core"] = obs[idxcol].map(xo.core_id); obs["samp"] = obs["sample"].astype(str)
    for s in samples:
        b = bmap.get(s)
        if b is None:
            print(f"[SKIP] {s}: not in discover_samples() bundle map"); continue
        ct_map = obs.loc[obs["samp"] == s].set_index("core")["cell_type_coarse"]
        if not len(ct_map):
            print(f"[{s}] no cells in obs_celltype"); continue
        print(f"\n===== {s} ({len(ct_map)} typed cells; bundle {b}) =====", flush=True)
        for st in SHOWCASE:
            try:
                render(s, b, ct_map, st, out)
            except Exception as e:
                print(f"[ERR {s} {st}] {e}")
    print("\nALL DONE ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", nargs="*", default=None)
    a = ap.parse_args()
    default = ["SD03914_BG", "SD03614_BG", "SD03522_BG", "SD01915_BG"]
    main(a.samples or default, a.out)
