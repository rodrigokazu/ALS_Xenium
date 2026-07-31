#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/hero_crop_FINAL.py
#
# The tight close-up of a handful of motor neurons in the ventral horn, built for the MNDA
# application. It finds a compact cluster of is_MN cells automatically and renders a small
# full-resolution DAPI and 18S window with the domain layer, white cell borders and cyan MN
# outlines.
#
# Entirely motor-neuron driven, so when is_MN.sum() is zero it skips cleanly with a message
# and writes nothing instead of producing an empty panel. The domain overview and the
# transcript overlays come from xenium_overlay_FINAL, not from here, so nothing else is lost
# while annotation is pending.
#
# Shares the boundary provenance guard with xenium_overlay_FINAL, so today the polygon layer
# is replaced by domain-coloured centroid dots with cyan rings on the motor neurons.
#
# Runs in the xenium_vistools environment.
# ========================================================================================

"""hero_crop_FINAL.py -- tight Xenium-Explorer-style close-up of a few motor neurons,
adapted for Marcel's Ranger_procd_mw_final segmentation. Auto-finds a compact cluster of
is_MN cells in the ventral horn and renders a small full-res DAPI+18S window with the
domain layer, white cell borders and cyan MN outlines. For the MNDA application.

is_MN GATE (mission requirement): the whole script is MN-driven, so if is_MN.sum()==0
(annotation pending) it SKIPS cleanly with a clear message and writes nothing -- the
domain overview/detail + transcript overlays come from xenium_overlay_FINAL, not here.

Boundary-provenance: reuses xenium_overlay_FINAL.resolve_boundary_join. When the _final
boundaries are stale (they are today; see that module's docstring) the polygon layer is
replaced by domain-coloured centroid dots + cyan MN centroid rings. Reads bundles
read-only. Env: xenium_vistools."""
import os, sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xenium_overlay_FINAL as xo   # reuse core_id, read_obs, load_cells, boundary guard, palette, PX, MN_EDGE
import nature_style as ns


def find_hero_centers(mn_xy, k=3, n=4, min_sep=450.0):
    """Return up to n centroids of tight MN groups, each >=min_sep apart. Returns [] when
    there are no MN cells (is_MN gate handled by the caller)."""
    from scipy.spatial import cKDTree
    if len(mn_xy) == 0:
        return []
    if len(mn_xy) <= k:
        return [mn_xy.mean(0)]
    tree = cKDTree(mn_xy)
    d, idx = tree.query(mn_xy, k=k)
    order = np.argsort(d[:, k-1])            # tightest groups first
    picked = []
    for i in order:
        c = mn_xy[idx[i]].mean(0)
        if all(np.hypot(*(c - p)) > min_sep for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def _render(sample, tag, cx, cy, half, full, s18_full, Wpx, Hpx, cells, cb_all, palette,
            polys_ok, outdir, txd=None):
    xa, xb, ya, yb = cx-half, cx+half, cy-half, cy+half
    X0, X1 = max(0, int(xa/xo.PX)), min(Wpx, int(xb/xo.PX))
    Y0, Y1 = max(0, int(ya/xo.PX)), min(Hpx, int(yb/xo.PX))
    dcrop = np.asarray(full[Y0:Y1, X0:X1])
    scrop = np.asarray(s18_full[Y0:Y1, X0:X1]) if s18_full is not None else None
    rgb = xo.composite_dapi_18s(dcrop, scrop)
    ext = [X0*xo.PX, X1*xo.PX, Y1*xo.PX, Y0*xo.PX]
    inc = cells[(cells["x_centroid"].between(xa, xb)) & (cells["y_centroid"].between(ya, yb))]
    incore = set(inc["core"])
    dmap = cells.set_index("core")["domain"]; mmap = cells.set_index("core")["is_MN"]

    fig, ax = plt.subplots(figsize=(7.2, 7.2)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.imshow(rgb, origin="upper", extent=ext, zorder=0)
    doms_present = set()
    if polys_ok and cb_all is not None:
        cb = cb_all[cb_all["k"].isin(incore)]
        polys, fcs, isMN = [], [], []
        for kk, sub in cb.groupby("k", sort=False):
            d = dmap.get(kk); v = sub[["vertex_x", "vertex_y"]].to_numpy()
            if d is None or len(v) < 3: continue
            polys.append(v); fcs.append(palette.get(str(d), "0.6")); isMN.append(bool(mmap.get(kk, False)))
            doms_present.add(str(d))
        ax.add_collection(PolyCollection(polys, facecolors=fcs, edgecolors="white", linewidths=0.4, alpha=0.35, zorder=2))
        mnp = [p for p, m in zip(polys, isMN) if m]
        if mnp:
            ax.add_collection(PolyCollection(mnp, facecolors="none", edgecolors=xo.MN_EDGE, linewidths=1.9, zorder=5))
        ncells, nmn = len(polys), sum(isMN)
    else:
        cvals = inc["domain"].astype(str).map(palette).fillna("0.6").tolist()
        ax.scatter(inc["x_centroid"].to_numpy(), inc["y_centroid"].to_numpy(), s=24, c=cvals,
                   marker="o", edgecolors="white", linewidths=0.25, alpha=0.7, zorder=2)
        mnin = inc[inc["is_MN"]]
        if len(mnin):
            ax.scatter(mnin["x_centroid"].to_numpy(), mnin["y_centroid"].to_numpy(), s=140,
                       facecolors="none", edgecolors=xo.MN_EDGE, linewidths=2.0, zorder=5)
        doms_present = set(inc["domain"].astype(str).unique())
        ncells, nmn = len(inc), int(len(mnin))

    xo.plot_transcripts(ax, txd, "hero", bbox=(xa, xb, ya, yb))
    ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([xa+half*0.12, xa+half*0.12+50], [yb-half*0.12, yb-half*0.12], color="white", lw=3, solid_capstyle="butt", zorder=20)
    ax.text(xa+half*0.12+25, yb-half*0.12-half*0.06, "50 um", color="white", ha="center", va="bottom", fontsize=9, zorder=20)
    present = sorted(doms_present, key=xo._dom_sort_key)
    h = [Line2D([0],[0], marker='s', linestyle='none', markersize=8, markerfacecolor=palette.get(d, "0.6"),
                markeredgecolor='none', label=d) for d in present]
    h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=8, markerfacecolor='none',
                    markeredgecolor=xo.MN_EDGE, markeredgewidth=1.6, label='motor neuron'))
    if txd:
        for g in sorted(txd.keys(), key=lambda g: 0 if xo._tier(g) == "abundant" else 1):
            h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=7,
                            markerfacecolor=xo._tx_color(g),
                            markeredgecolor=('black' if xo._tier(g) == 'rare' else 'none'),
                            markeredgewidth=0.5, label=g))
    lg = ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.005, 0.5), frameon=False,
                   labelcolor='white', fontsize=8, title='Novae domain')
    lg.get_title().set_color('white')
    mode = "" if polys_ok else "  [centroid fallback: boundaries stale]"
    ax.set_title(f"{sample}  |  motor neurons in the ventral horn (Xenium; DAPI + 18S){mode}", color="white", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(f"{outdir}/{sample}__hero_MN_{tag}.{e}", dpi=450, bbox_inches="tight", facecolor="black")
    plt.close(fig); print(f"[{sample}] wrote hero_MN_{tag}  ({ncells} cells, {nmn} MN) @ {cx:.0f},{cy:.0f}", flush=True)


def main(sample, bundle, h5, outdir, half=115.0, center=None, topk=4, genes=None, min_sep=450.0):
    os.makedirs(outdir, exist_ok=True)
    ns.apply_style()
    dkey = xo.pick_domain_key(h5)
    obs = xo.read_obs(h5, [dkey, "is_MN"])
    dom = obs[dkey]
    mn = xo._as_bool(obs["is_MN"]) if "is_MN" in obs else pd.Series(False, index=dom.index)
    cells = xo.load_cells(bundle, dom, mn)
    mn_xy = cells.loc[cells["is_MN"], ["x_centroid", "y_centroid"]].to_numpy()
    print(f"[{sample}] MN cells {len(mn_xy)}", flush=True)
    # ---- is_MN GATE ----
    if len(mn_xy) == 0:
        print(f"[{sample}] SKIP heroes: is_MN.sum()==0 (annotation pending). "
              f"Domain overview/detail + transcript overlays are produced by "
              f"xenium_overlay_FINAL; MN heroes require is_MN.", flush=True)
        return
    palette = xo.build_domain_palette(dom.dropna().unique())
    centers = [center] if center is not None else find_hero_centers(mn_xy, k=3, n=topk, min_sep=min_sep)
    print(f"[{sample}] hero centers found: {len(centers)} (requested {topk}, min_sep {min_sep})", flush=True)
    arrs = xo.load_dapi_levels(bundle); full = arrs[0]
    s18_arrs = xo.load_18s_levels(bundle); s18_full = s18_arrs[0] if s18_arrs else None
    print(f"[{sample}] 18S channel: {'present' if s18_full is not None else 'MISSING (grey DAPI fallback)'}", flush=True)
    txd = xo.load_transcripts(bundle, genes) if genes else None
    if txd:
        print(f"[{sample}] transcripts " + ", ".join(f"{g}={len(txd[g])}" for g in genes), flush=True)
    Hpx, Wpx = full.shape[-2], full.shape[-1]
    cb_all, key, polys_ok, bdiag = xo.resolve_boundary_join(bundle, cells)
    for j, (cx, cy) in enumerate(centers, 1):
        _render(sample, f"c{j}", cx, cy, half, full, s18_full, Wpx, Hpx, cells, cb_all,
                palette, polys_ok, outdir, txd=txd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--bundle", required=True)
    ap.add_argument("--h5", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--half", type=float, default=115.0)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--min-sep", dest="min_sep", type=float, default=450.0)
    ap.add_argument("--genes", default=None)
    ap.add_argument("--cx", type=float, default=None); ap.add_argument("--cy", type=float, default=None)
    a = ap.parse_args()
    ctr = (a.cx, a.cy) if a.cx is not None else None
    g = a.genes.split(",") if a.genes else None
    main(a.sample, a.bundle, a.h5, a.outdir, half=a.half, center=ctr, topk=a.topk, genes=g, min_sep=a.min_sep)
