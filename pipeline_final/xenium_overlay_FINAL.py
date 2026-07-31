#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/xenium_overlay_FINAL.py
#
# The imaging layer. Puts Novae domains on top of the real DAPI image, Xenium Explorer style,
# and provides the loaders the other overlay scripts build on.
#
# Two panels per sample: a downsampled full-section overview with every cell coloured by
# domain and a 1 mm bar, and a full-resolution ventral-horn crop with DAPI plus 18S, the
# domain layer, white cell borders, cyan motor-neuron outlines and a 200 um bar.
#
# The coordinate convention is verified and should not be improvised on. origin='upper',
# polygons and centroids in microns, imshow extent in microns, pixels at micron/0.2125, no
# y-flip and no manual affine. DAPI comes from morphology_focus/ch0000_dapi.ome.tif read with
# is_ome=False so the pyramid resolves.
#
# The important piece of engineering here is resolve_boundary_join(). The _final boundary
# files are stale: only 10 to 17 percent of cells carry a polygon, and the gm-namespaced ids
# do not correspond to the imported cells, with a median polygon-to-centroid offset of several
# thousand microns. Rather than trusting them or hardcoding a workaround, this function
# measures the alignment at runtime and only turns the polygon layer on when the median offset
# and the coverage both pass. Otherwise it falls back to a domain-coloured centroid map, which
# is a legitimate spatial plot and not a degraded one, and says so loudly in the log. When the
# boundaries are re-exported the guard re-enables polygons by itself.
#
# Domain palettes are built dynamically from whatever labels the run emits, with grey reserved
# for unassigned and cyan reserved for motor neurons. Nothing is hardcoded to a domain id,
# because those change with every model.
#
# Bundles are opened read-only.
# ========================================================================================

"""
xenium_overlay_FINAL.py  --  Xenium Explorer-style figure: Novae domains on the real
DAPI image, adapted for Marcel's Ranger_procd_mw_final ("_final") segmentation.

Two panels per sample:
  (A) full-section overview   : downsampled DAPI + ALL cells coloured by Novae domain
                               (filled polygons if boundaries are valid, else centroid
                               scatter), MN markers, transcript dots, 1 mm bar.
  (B) ventral-horn detail crop: full-res DAPI + 18S + domain layer + white cell borders
                               + cyan MN outlines (or MN centroid rings in the fallback),
                               200 um bar.

Coordinate convention (UNCHANGED, verified): origin='upper', polygons/centroids in
microns, imshow extent in microns, px = micron/0.2125, NO y-flip, NO manual affine.
DAPI = morphology_focus/ch0000_dapi.ome.tif (is_ome=False -> pyramid).

=====================================================================================
_final DELTAS vs _gausss (verified read-only on SCG 2026-07-19; see crew scratch pad):
  * obs_names are '<LABEL>:<int>' (plain integer cell ids). core_id() -> '<int>'.
  * cells.parquet cell_id = integer strings '1'..'N'  -> joins obs 1:1 (integer core).
  * transcripts.parquet cell_id = int64, UNASSIGNED == 0 (NOT the "UNASSIGNED" string).
    load_transcripts() only needs feature_name/x/y/qv, so it is namespace-agnostic.
  * !!! cell_boundaries.parquet / nucleus_boundaries.parquet are STALE in _final:
        only ~10-17% of cells carry a polygon and the gm-namespaced cell_id / its
        label_id do NOT correspond to the new imported cells (polygon-vs-cells centroid
        median distance ~4000-7400 um). So a filled-polygon domain map is NOT possible
        until Marcel regenerates boundaries on the imported segmentation.
    -> resolve_boundary_join() TESTS boundary<->cells centroid alignment at runtime and
       only enables the polygon layer if the median offset <= BOUNDARY_MATCH_TOL_UM and
       coverage >= MIN_BOUNDARY_COVERAGE. Otherwise the scripts fall back to a domain-
       coloured CENTROID map (scientifically valid squidpy-style spatial plot) + MN
       centroid rings, and log the fallback + the diagnostics LOUDLY. When Marcel fixes
       the boundaries (integer cell_id OR aligned label_id) the guard auto-re-enables
       polygons with no code change.
  * Novae domain labels differ from the _gausss set (D1011..D991). The palette is now
    built DYNAMICALLY from whatever domains the _final Novae run emits (build_domain_
    palette); 'unassigned' -> grey, cyan reserved for MN. No hardcoded domain ids.

is_MN GATE: MN outlines/rings + the MN-centred detail crop degrade cleanly when
is_MN.sum()==0 (annotation pending): overview/detail + transcript overlays still render;
only MN-specific marks are suppressed. See build_polys/main.

Env: /oak/.../RK/envs/xenium_vistools/bin/python (pyarrow+tifffile+zarr). Read-only on
bundles. obs read via h5py (old anndata chokes on /uns/log1p base 'null').
"""
import os, re, sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import tifffile, h5py, zarr
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nature_style as ns

PX = 0.2125

# Novae per-sample domain obs key. MUST match what novae_persample_niches_FINAL writes.
# (confirm at launch; pick_domain_key falls back to any 'novae_domains*n6*' key.)
NOVAE_DOMAIN_KEY = "novae_domains_n6"

# ---- boundary-provenance guard thresholds (see module docstring) --------------------
BOUNDARY_MATCH_TOL_UM  = 30.0   # median polygon-vs-cells centroid offset must be <= this
MIN_BOUNDARY_COVERAGE  = 0.80   # fraction of domain cells that must have a matching polygon

MN_EDGE = "#00E5FF"  # cyan (reserved for motor neurons; never used for a domain)

# Dynamic domain palette: distinct saturated hues, cyan deliberately excluded so MN
# outlines stay unambiguous. Assigned by (numeric-aware) sorted domain label at runtime.
_DOM_PALETTE = ["#1F77B4", "#FF9E1B", "#2CA02C", "#E377C2", "#9467BD", "#D62728",
                "#8C564B", "#BCBD22", "#E6550D", "#756BB1", "#31A354", "#AD494A"]


def _dom_sort_key(d):
    """Sort D-labels by trailing integer when present (D991 < D1011), else lexicographic."""
    m = re.search(r"(\d+)\s*$", str(d))
    return (0, int(m.group(1))) if m else (1, str(d))


def build_domain_palette(domains, unassigned="unassigned"):
    """{domain_label -> colour} built from the domains actually present. Deterministic
    across runs (sorted), 'unassigned' -> grey and does not consume a palette slot."""
    doms = sorted({str(d) for d in domains if str(d) != unassigned}, key=_dom_sort_key)
    pal = {d: _DOM_PALETTE[i % len(_DOM_PALETTE)] for i, d in enumerate(doms)}
    pal[unassigned] = "0.55"
    return pal


# ---- transcript overlay (tier-based; generalises to any gene) -----------------------
#   abundant gene = small white dots (density without fog); rare gene = larger edged
#   magenta dots (pops). Covers all six _final overlay/coloc genes.
TX_COLOR = {"STMN2": "#FFFFFF", "TARDBP": "#FFFFFF",
            "CE_STMN2": "#FF19C3", "CCDC146": "#FF19C3", "MNX1": "#FF19C3", "BCL6": "#FF19C3"}
TX_TIER  = {"STMN2": "abundant", "TARDBP": "abundant",
            "CE_STMN2": "rare", "CCDC146": "rare", "MNX1": "rare", "BCL6": "rare"}
_TIER_STYLE = {  # (size, alpha, edge_lw) per panel scale
    "abundant": {"overview": (0.35, 0.35, 0.0), "detail": (1.6, 0.60, 0.0), "hero": (5.0, 0.80, 0.0)},
    "rare":     {"overview": (5.0, 0.95, 0.25), "detail": (9.0, 0.95, 0.35), "hero": (16.0, 0.95, 0.45)},
}
def _tier(g): return TX_TIER.get(g, "rare")
def _tx_style(g, scale): return _TIER_STYLE[_tier(g)][scale]
def _tx_color(g): return TX_COLOR.get(g, "#FFF200")


def load_transcripts(bundle, genes, qv_min=20):
    """genes -> {gene: Nx2 (x_location, y_location)} for qv>=qv_min. Namespace-agnostic
    (reads only feature_name/x/y/qv), so unaffected by the _final int64 cell_id."""
    df = pd.read_parquet(f"{bundle}/transcripts.parquet",
                         columns=["feature_name", "x_location", "y_location", "qv"],
                         filters=[("feature_name", "in", set(genes))])
    df = df[df["qv"] >= qv_min]
    return {g: df.loc[df["feature_name"] == g, ["x_location", "y_location"]].to_numpy() for g in genes}


def plot_transcripts(ax, txd, scale, bbox=None):
    if not txd: return
    # draw abundant tier first so rarer, larger dots land on top
    for g in sorted(txd.keys(), key=lambda g: 0 if _tier(g) == "abundant" else 1):
        xy = txd.get(g)
        if xy is None or not len(xy): continue
        if bbox is not None:
            xa, xb, ya, yb = bbox
            m = (xy[:, 0] >= xa) & (xy[:, 0] <= xb) & (xy[:, 1] >= ya) & (xy[:, 1] <= yb)
            xy = xy[m]
        if not len(xy): continue
        s, al, elw = _tx_style(g, scale)
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=_tx_color(g), alpha=al,
                   edgecolors=("black" if elw > 0 else "none"), linewidths=elw,
                   zorder=7, rasterized=True)


def core_id(s):
    """obs '<LABEL>:<int>' -> '<int>'; cells '<int>' -> '<int>'; a gm/'<int>-1' bundle id
    -> strip trailing '-<digits>'. On an int64 transcripts id -> str(int)."""
    s = str(s)
    if ":" in s: s = s.split(":", 1)[1]
    return re.sub(r"-\d+$", "", s)


def list_obs_keys(h5path):
    with h5py.File(h5path, "r") as f:
        return list(f["obs"].keys())


def pick_domain_key(h5path, preferred=NOVAE_DOMAIN_KEY):
    """Return the Novae n6 domain obs key, tolerating a naming drift in the FINAL Novae
    output. Prefers the exact key; else any 'novae_domains*6*' / 'novae_domains*' key;
    else raises a clear error listing what IS in obs."""
    keys = list_obs_keys(h5path)
    if preferred in keys:
        return preferred
    cand = [k for k in keys if re.search(r"novae.*domain", k, re.I) and "6" in k]
    if not cand:
        cand = [k for k in keys if re.search(r"novae.*domain", k, re.I)]
    if cand:
        print(f"[warn] domain key '{preferred}' absent; using '{sorted(cand)[0]}' "
              f"(available novae keys: {sorted(cand)})")
        return sorted(cand)[0]
    raise KeyError(f"No Novae domain column in {h5path}. obs keys: {sorted(keys)}")


def read_obs(h5path, cols):
    """Read obs columns into {col: Series indexed by core-id}. Categorical-aware; decodes
    bytes. Missing columns are silently skipped (caller handles absence, e.g. is_MN)."""
    out = {}
    with h5py.File(h5path, "r") as f:
        obs = f["obs"]; ik = obs.attrs.get("_index", "_index")
        if isinstance(ik, bytes): ik = ik.decode()
        index = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in obs[ik][:]])
        idx = [core_id(i) for i in index]
        for c in cols:
            if c not in obs:
                continue
            node = obs[c]
            if isinstance(node, h5py.Group):
                cats = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][:]])
                codes = node["codes"][:]; vals = np.where(codes < 0, "NA", cats[codes])
            else:
                vals = node[:]
                if vals.dtype.kind == "S": vals = np.array([x.decode() for x in vals])
            out[c] = pd.Series(vals, index=idx)
    return out


def _as_bool(series):
    """is_MN as robust bool (survives bool/int/str storage; 'True'/'1'/'1.0' -> True)."""
    return series.map(lambda v: str(v).lower() in ("true", "1", "1.0"))


def load_channel_levels(path):
    """Lazy pyramid levels (largest first) for a single-channel OME-TIFF, is_ome=False."""
    if not os.path.exists(path):
        return None
    store = tifffile.imread(path, aszarr=True, is_ome=False)
    za = zarr.open(store, mode="r")
    if hasattr(za, "array_keys"):
        arrs = [za[k] for k in sorted(za.array_keys(), key=lambda k: -int(np.prod(za[k].shape)))]
    else:
        arrs = [za]
    return [a for a in arrs if a.ndim == 2] or arrs


def load_dapi_levels(bundle):
    p = f"{bundle}/morphology_focus/ch0000_dapi.ome.tif"
    if not os.path.exists(p):
        p = f"{bundle}/morphology.ome.tif"
    return load_channel_levels(p)


def load_18s_levels(bundle):
    return load_channel_levels(f"{bundle}/morphology_focus/ch0002_18s.ome.tif")


def _stretch(a, lo, hi):
    a = np.asarray(a, dtype=float)
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - v0) / max(v1 - v0, 1e-6), 0, 1)


def composite_dapi_18s(dapi, s18):
    """RGB: DAPI -> blue nuclei, 18S (rRNA) -> amber cytoplasm. Large MN somata glow amber."""
    d = _stretch(dapi, 1, 99.5)
    if s18 is None:
        return np.stack([d*0.85, d*0.9, d], -1)   # fallback: near-grey DAPI
    r = _stretch(s18, 2, 99.8)
    rgb = np.empty(d.shape + (3,), float)
    rgb[..., 0] = np.clip(0.95*r + 0.12*d, 0, 1)   # red   <- 18S
    rgb[..., 1] = np.clip(0.55*r + 0.30*d, 0, 1)   # green <- 18S (amber) + a little DAPI
    rgb[..., 2] = np.clip(0.12*r + 0.95*d, 0, 1)   # blue  <- DAPI
    return rgb


# =====================================================================================
# CELLS (integer-id joined to obs) + boundary-provenance guard
# =====================================================================================
def load_cells(bundle, dom, mn):
    """cells.parquet -> DataFrame with core-id, domain, is_MN, centroids, cell_area.
    Integer core-id join to obs (delta D2). Rows without a domain are dropped."""
    cpath = f"{bundle}/cells.parquet"
    cols = ["cell_id", "x_centroid", "y_centroid"]
    avail = set(pq.ParquetFile(cpath).schema_arrow.names)
    if "cell_area" in avail:
        cols.append("cell_area")
    cells = pd.read_parquet(cpath, columns=cols)
    cells["core"] = cells["cell_id"].map(core_id)
    cells["domain"] = cells["core"].map(dom)
    cells["is_MN"] = cells["core"].map(mn).fillna(False).astype(bool)
    n_all = len(cells)
    cells = cells[cells["domain"].notna()].copy()
    print(f"  [cells<->obs] domain-join {len(cells)}/{n_all} = {len(cells)/max(n_all,1):.3f}", flush=True)
    return cells


def resolve_boundary_join(bundle, cells):
    """Decide whether the bundle's cell_boundaries.parquet can be trusted for a polygon
    map, and by WHICH key it joins to `cells` (which is keyed by integer core-id).

    Tries two candidate keys -- core_id(cell_id) and str(label_id) -- computes each
    key's per-cell polygon centroid, compares to the cells.parquet centroid, and picks
    the key with the smallest median offset. polys_ok is True only if that median offset
    <= BOUNDARY_MATCH_TOL_UM AND coverage >= MIN_BOUNDARY_COVERAGE.

    Returns (cb, key, polys_ok, diag) where cb has a 'k' column set to the chosen key
    (None if no boundaries file). diag is a printable dict.
    """
    path = f"{bundle}/cell_boundaries.parquet"
    if not os.path.exists(path):
        return None, None, False, {"reason": "no cell_boundaries.parquet"}
    want = ["cell_id", "vertex_x", "vertex_y"]
    schema_cols = set(pq.ParquetFile(path).schema_arrow.names)
    if "label_id" in schema_cols:
        want.append("label_id")
    cb = pd.read_parquet(path, columns=want)
    cmap = cells.set_index("core")[["x_centroid", "y_centroid"]]
    candidates = {}
    tmp = cb.copy(); tmp["k"] = tmp["cell_id"].map(core_id); candidates["cell_id_core"] = tmp
    if "label_id" in cb.columns:
        tmp = cb.copy(); tmp["k"] = tmp["label_id"].astype(str); candidates["label_id"] = tmp
    best = None
    for name, cbx in candidates.items():
        cent = cbx.groupby("k")[["vertex_x", "vertex_y"]].mean()
        common = cent.index.intersection(cmap.index)
        if len(common) == 0:
            med, cov = np.inf, 0.0
        else:
            d = np.hypot(cent.loc[common, "vertex_x"].values - cmap.loc[common, "x_centroid"].values,
                         cent.loc[common, "vertex_y"].values - cmap.loc[common, "y_centroid"].values)
            med = float(np.median(d)); cov = len(common) / max(len(cmap), 1)
        cand = {"key": name, "median_um": med, "coverage": cov, "n_match": int(len(common))}
        if best is None or med < best["median_um"]:
            best = cand; best_cbx = cbx
    polys_ok = (best["median_um"] <= BOUNDARY_MATCH_TOL_UM) and (best["coverage"] >= MIN_BOUNDARY_COVERAGE)
    diag = dict(best)
    diag["polys_ok"] = polys_ok
    diag["tol_um"] = BOUNDARY_MATCH_TOL_UM
    diag["min_cov"] = MIN_BOUNDARY_COVERAGE
    if not polys_ok:
        print(f"  [boundary-guard] STALE/MISMATCHED boundaries -> polygon layer DISABLED "
              f"(best key={best['key']} median_offset={best['median_um']:.1f}um "
              f"coverage={best['coverage']:.2f}; need <= {BOUNDARY_MATCH_TOL_UM}um & "
              f">= {MIN_BOUNDARY_COVERAGE:.2f}). Falling back to centroid rendering. "
              f"ACTION: Marcel must regenerate cell_boundaries on the imported seg.", flush=True)
        return best_cbx, best["key"], False, diag
    print(f"  [boundary-guard] boundaries OK -> polygons ENABLED "
          f"(key={best['key']} median_offset={best['median_um']:.2f}um "
          f"coverage={best['coverage']:.2f})", flush=True)
    return best_cbx, best["key"], True, diag


def build_polys(bundle, cells, palette):
    """Build domain-filled polygons for ALL cells IF boundaries are valid. Returns
    (polys, facecolors, domains, isMN_bool_array, centroids_array, polys_ok, diag).
    When boundaries are stale (polys_ok False) returns empty geometry so the caller
    uses the centroid fallback."""
    cb, key, polys_ok, diag = resolve_boundary_join(bundle, cells)
    if not polys_ok or cb is None:
        return [], [], [], np.array([], dtype=bool), np.empty((0, 2)), False, diag
    dmap = cells.set_index("core")["domain"]; mmap = cells.set_index("core")["is_MN"]
    cb = cb.copy()
    cb["domain"] = cb["k"].map(dmap)
    cb = cb[cb["domain"].notna()]
    cb["is_MN"] = cb["k"].map(mmap).fillna(False)
    polys, fcs, doms, isMN, cxy = [], [], [], [], []
    for kk, sub in cb.groupby("k", sort=False):
        d = sub["domain"].iloc[0]
        v = sub[["vertex_x", "vertex_y"]].to_numpy()
        if len(v) < 3: continue
        polys.append(v); fcs.append(palette.get(d, "0.6")); doms.append(d)
        isMN.append(bool(sub["is_MN"].iloc[0])); cxy.append(v.mean(0))
    return polys, fcs, doms, np.array(isMN), np.array(cxy), True, diag


def scalebar(ax, length_um, x0, y0, label, color="white", lw=3):
    ax.plot([x0, x0 + length_um], [y0, y0], color=color, lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(x0 + length_um / 2, y0 - length_um * 0.12, label, color=color, ha="center",
            va="bottom", fontsize=8, zorder=20)


def domain_legend(ax, present, palette, with_mn=True, tx_genes=None, swatch=11, fontsize=10):
    h = [Line2D([0],[0], marker='s', linestyle='none', markersize=swatch,
                markerfacecolor=palette.get(d, "0.6"), markeredgecolor='none', label=d) for d in present]
    if with_mn:
        h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=swatch, markerfacecolor='none',
                        markeredgecolor=MN_EDGE, markeredgewidth=1.6, label='motor neuron'))
    for g in (tx_genes or []):
        h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=swatch-3,
                        markerfacecolor=_tx_color(g), markeredgecolor=('black' if _tier(g) == 'rare' else 'none'),
                        markeredgewidth=0.5, label=f'{g}'))
    lg = ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.005, 0.5), title='Novae domain',
                   frameon=False, labelcolor='white', fontsize=fontsize, handletextpad=0.6, labelspacing=0.7)
    lg.get_title().set_color('white'); lg.get_title().set_fontsize(fontsize + 1)


def main(sample, bundle, h5, outdir, crop_center=None, genes=None):
    os.makedirs(outdir, exist_ok=True)
    ns.apply_style()
    dkey = pick_domain_key(h5)
    obs = read_obs(h5, [dkey, "is_MN"])
    dom = obs[dkey]
    mn = _as_bool(obs["is_MN"]) if "is_MN" in obs else pd.Series(False, index=dom.index)
    n_mn = int(mn.sum())
    print(f"[{sample}] domains {dom.value_counts().to_dict()}  MN {n_mn}"
          f"{'  (is_MN pending -> MN marks suppressed)' if n_mn == 0 else ''}", flush=True)
    palette = build_domain_palette(dom.dropna().unique())

    txd = None
    if genes:
        txd = load_transcripts(bundle, genes)
        print(f"[{sample}] transcripts " + ", ".join(f"{g}={len(txd[g])}" for g in genes), flush=True)

    cells = load_cells(bundle, dom, mn)
    arrs = load_dapi_levels(bundle)
    print(f"[{sample}] DAPI levels {[tuple(a.shape) for a in arrs]}", flush=True)
    full = arrs[0]; Hpx, Wpx = full.shape[-2], full.shape[-1]
    W_um, H_um = Wpx * PX, Hpx * PX

    polys, fcs, doms, isMN, cxy, polys_ok, bdiag = build_polys(bundle, cells, palette)
    if polys_ok:
        print(f"[{sample}] polygons {len(polys)}  MN polys {int(isMN.sum())}", flush=True)
        dom_set = set(doms)
    else:
        print(f"[{sample}] CENTROID FALLBACK ({len(cells)} cells) -- boundaries unusable "
              f"({bdiag.get('key')}: median {bdiag.get('median_um'):.0f}um, "
              f"cov {bdiag.get('coverage'):.2f})", flush=True)
        dom_set = set(cells["domain"].astype(str).unique())
    present = [d for d in sorted(dom_set, key=_dom_sort_key)]

    # MN centroids (always from cells.parquet integer join -> robust to boundary state)
    mn_xy_all = cells.loc[cells["is_MN"], ["x_centroid", "y_centroid"]].to_numpy()

    # ---------- (A) OVERVIEW ----------
    lvl = min(arrs, key=lambda a: abs(max(a.shape) - 5000))
    bg = np.asarray(lvl)
    vmin, vmax = np.percentile(bg, 50), np.percentile(bg, 99.8)
    fig, ax = plt.subplots(figsize=(8.2, 8.2)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", extent=[0, W_um, H_um, 0], zorder=0, alpha=0.68)
    if polys_ok:
        pc = PolyCollection(polys, facecolors=list(fcs), edgecolors="none", alpha=0.95, zorder=2)
        pc.set_rasterized(True); ax.add_collection(pc)
        ncells_shown = len(polys)
    else:
        cvals = cells["domain"].astype(str).map(palette).fillna("0.6").tolist()
        ax.scatter(cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy(),
                   s=2.0, c=cvals, marker=".", linewidths=0, alpha=0.9, zorder=2, rasterized=True)
        ncells_shown = len(cells)
    if len(mn_xy_all):
        ax.scatter(mn_xy_all[:, 0], mn_xy_all[:, 1], s=10, facecolors="none", edgecolors=MN_EDGE,
                   linewidths=0.7, zorder=4, rasterized=True)
    plot_transcripts(ax, txd, "overview")
    ax.set_xlim(0, W_um); ax.set_ylim(H_um, 0); ax.set_aspect("equal"); ax.axis("off")
    scalebar(ax, 1000, W_um * 0.04, H_um * 0.96, "1 mm")
    mode = "polygons" if polys_ok else "centroids (boundaries stale)"
    ax.set_title(f"{sample}  |  Novae domains on Xenium DAPI ({mode}; n={ncells_shown:,} cells)",
                 color="white", fontsize=10)
    domain_legend(ax, present, palette, with_mn=len(mn_xy_all) > 0, tx_genes=genes)
    fig.savefig(f"{outdir}/{sample}__overview.png", dpi=350, bbox_inches="tight", facecolor="black")
    fig.savefig(f"{outdir}/{sample}__overview.pdf", dpi=350, bbox_inches="tight", facecolor="black")
    plt.close(fig); print(f"[{sample}] wrote overview", flush=True)

    # ---------- (B) DETAIL CROP ----------
    if crop_center is None:
        cx0, cy0 = (mn_xy_all.mean(0) if len(mn_xy_all)
                    else cells[["x_centroid", "y_centroid"]].to_numpy().mean(0))
    else:
        cx0, cy0 = crop_center
    half = 500.0
    xa, xb, ya, yb = cx0-half, cx0+half, cy0-half, cy0+half
    X0, X1, Y0, Y1 = int(xa/PX), int(xb/PX), int(ya/PX), int(yb/PX)
    X0, X1 = max(0, X0), min(Wpx, X1); Y0, Y1 = max(0, Y0), min(Hpx, Y1)
    s18_arrs = load_18s_levels(bundle); s18_full = s18_arrs[0] if s18_arrs else None
    dcrop = np.asarray(full[Y0:Y1, X0:X1])
    scrop = np.asarray(s18_full[Y0:Y1, X0:X1]) if s18_full is not None else None
    rgb = composite_dapi_18s(dcrop, scrop)
    ext = [X0*PX, X1*PX, Y1*PX, Y0*PX]
    fig, ax = plt.subplots(figsize=(7.6, 7.6)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.imshow(rgb, origin="upper", extent=ext, zorder=0)
    if polys_ok:
        sel = [i for i, c in enumerate(cxy) if xa <= c[0] <= xb and ya <= c[1] <= yb]
        pc = PolyCollection([polys[i] for i in sel], facecolors=[fcs[i] for i in sel],
                            edgecolors="white", linewidths=0.3, alpha=0.38, zorder=2)
        pc.set_rasterized(True); ax.add_collection(pc)
        mnsel = [i for i in sel if isMN[i]]
        if mnsel:
            mpc = PolyCollection([polys[i] for i in mnsel], facecolors="none", edgecolors=MN_EDGE,
                                 linewidths=1.4, zorder=5); ax.add_collection(mpc)
        n_in, n_mn_in = len(sel), len(mnsel)
    else:
        inc = cells[(cells["x_centroid"].between(xa, xb)) & (cells["y_centroid"].between(ya, yb))]
        cvals = inc["domain"].astype(str).map(palette).fillna("0.6").tolist()
        ax.scatter(inc["x_centroid"].to_numpy(), inc["y_centroid"].to_numpy(), s=14, c=cvals,
                   marker="o", edgecolors="white", linewidths=0.2, alpha=0.7, zorder=2, rasterized=True)
        mnin = inc[inc["is_MN"]]
        if len(mnin):
            ax.scatter(mnin["x_centroid"].to_numpy(), mnin["y_centroid"].to_numpy(), s=90,
                       facecolors="none", edgecolors=MN_EDGE, linewidths=1.6, zorder=5)
        n_in, n_mn_in = len(inc), int(len(mnin))
    plot_transcripts(ax, txd, "detail", bbox=(xa, xb, ya, yb))
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect("equal"); ax.axis("off")
    scalebar(ax, 200, xa + (xb-xa)*0.04, yb - (yb-ya)*0.04, "200 um")
    ax.set_title(f"{sample}  |  ventral-horn detail  (DAPI + 18S; {n_in:,} cells, {n_mn_in} MN)",
                 color="white", fontsize=10)
    domain_legend(ax, present, palette, with_mn=len(mn_xy_all) > 0, tx_genes=genes)
    fig.savefig(f"{outdir}/{sample}__detail.png", dpi=400, bbox_inches="tight", facecolor="black")
    fig.savefig(f"{outdir}/{sample}__detail.pdf", dpi=400, bbox_inches="tight", facecolor="black")
    plt.close(fig); print(f"[{sample}] wrote detail", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--genes", default=None, help="comma-sep transcript overlay, e.g. STMN2,CE_STMN2")
    a = ap.parse_args()
    g = a.genes.split(",") if a.genes else None
    main(a.sample, a.bundle, a.h5, a.outdir, genes=g)
