#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/matched_triple_panel_FINAL.py
#
# The cleanest version of the three-gene comparison. It finds a motor neuron positive for
# MNX1, BCL6 and STMN2 together, centres a crop on it, and renders one panel per gene over the
# identical field with identical motor-neuron marks. Only the transcript dots change, so
# density is the only thing the eye can vary on.
#
# That constraint is what makes the figure mean anything. Three crops from three different
# fields would let composition, cell density and section quality all differ, and any apparent
# difference in transcript density would be uninterpretable.
#
# Dot sizes differ per gene (MNX1 70, BCL6 55, STMN2 18): STMN2 is far denser and would
# otherwise fill the panel solid. That is a legibility choice and it does change the visual
# weight between panels, so the sizes belong in the caption.
#
# Needs a triple-positive motor neuron, so it skips cleanly when is_MN.sum() is zero instead
# of throwing on an empty candidate list.
# ========================================================================================

"""matched_triple_panel_FINAL.py -- side-by-side co-localisation of MNX1 | BCL6 | STMN2 over
the SAME ventral-horn field. Finds a motor neuron positive for all three genes, centres a
crop on it, and renders one panel per gene (identical field, identical MN marks); only the
transcript dots differ, so DENSITY is the only visual variable. Adapted for _final.

_final deltas:
  * bundle/paths from final_config (authoritative dedup); per-gene transcript counts use the
    int64 cell_id with UNASSIGNED==0.
  * dynamic Novae-domain palette (xenium_overlay_FINAL.build_domain_palette).
  * boundary-provenance guard: filled polygons when boundaries are valid, else domain-
    coloured centroid dots + cyan MN centroid rings (identical across the 3 panels).
  * is_MN GATE: needs a triple-positive MN -> if is_MN.sum()==0 (annotation pending) it
    SKIPs cleanly (no crash on the empty-candidate .iloc[0]).

Reads bundles read-only. PRE-QC caveat applies. Env: xenium_vistools."""
import os, sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as fc
import xenium_overlay_FINAL as xo
import nature_style as ns

HDIR = str(fc.NOVAE_JOINT_DIR / "per_sample")
GENES = ["MNX1", "BCL6", "STMN2"]
GSIZE = {"MNX1": 70, "BCL6": 55, "STMN2": 18}   # dot size per gene (STMN2 dense -> smaller)
TX = "#FFF200"                                   # single yellow for all genes (density = signal)


def resolve_bundle(sample):
    return {lab: str(p) for lab, p in fc.discover_samples(verbose=False)}.get(sample)


def per_core(bundle, gene, qv_min=20):
    df = pd.read_parquet(f"{bundle}/transcripts.parquet", columns=["feature_name", "cell_id", "qv"],
                         filters=[("feature_name", "in", {gene})])
    df = df[df["qv"] >= qv_min]
    df = _drop_unassigned(df).copy()
    df["core"] = df["cell_id"].map(xo.core_id)
    return df["core"].value_counts()


def _drop_unassigned(df):
    cid = df["cell_id"]
    if np.issubdtype(cid.dtype, np.number):
        return df[cid != 0]
    s = cid.astype(str)
    return df[~s.isin(["0", "-1", "UNASSIGNED"])]


def main(sample, out, half=105.0, cx=None, cy=None):
    os.makedirs(out, exist_ok=True); ns.apply_style()
    b = resolve_bundle(sample); h5 = f"{HDIR}/{sample}__niches.h5ad"
    if b is None or not os.path.exists(h5):
        print(f"[SKIP] {sample}: bundle={b} h5exists={os.path.exists(h5)} "
              f"(Novae _FINAL per-sample h5 pending?)"); return
    dkey = xo.pick_domain_key(h5)
    obs = xo.read_obs(h5, [dkey, "is_MN"])
    dom = obs[dkey]
    mns = xo._as_bool(obs["is_MN"]) if "is_MN" in obs else pd.Series(False, index=dom.index)
    cells = xo.load_cells(b, dom, mns)
    palette = xo.build_domain_palette(dom.dropna().unique())
    counts = {g: per_core(b, g) for g in GENES}
    for g in GENES:
        cells[g] = cells["core"].map(counts[g]).fillna(0).astype(int)
    mn = cells[cells["is_MN"]].copy()
    # ---- is_MN GATE ----
    if len(mn) == 0:
        print(f"[{sample}] SKIP matched-triple: is_MN.sum()==0 (annotation pending). "
              f"This figure requires a triple-positive motor neuron.", flush=True)
        return
    if cx is None:
        cand = mn[(mn["MNX1"] >= 1) & (mn["BCL6"] >= 2) & (mn["STMN2"] >= 3)].copy()
        if not len(cand):
            cand = mn[(mn["MNX1"] >= 1) & (mn["BCL6"] >= 1)].copy()
        if not len(cand):
            print(f"[{sample}] SKIP matched-triple: no MN positive for the MNX1/BCL6/STMN2 "
                  f"combination.", flush=True)
            return
        if "cell_area" in cand.columns and cand["cell_area"].std() > 0:
            az = (cand["cell_area"] - cand["cell_area"].mean()) / max(cand["cell_area"].std(), 1e-6)
        else:
            az = pd.Series(0.0, index=cand.index)
        cand["score"] = cand["MNX1"] + cand["BCL6"] + np.log1p(cand["STMN2"]) + az.clip(-1, 3)
        tgt = cand.sort_values("score", ascending=False).iloc[0]
        cx, cy = float(tgt["x_centroid"]), float(tgt["y_centroid"])
        tcounts = {g: int(tgt[g]) for g in GENES}
        area_s = f"{tgt['cell_area']:.0f}" if "cell_area" in tgt else "NA"
        print(f"[{sample}] target MN {tgt['core']} @ {cx:.0f},{cy:.0f}  counts={tcounts}  area={area_s}", flush=True)
    else:
        tcounts = None

    arrs = xo.load_dapi_levels(b); full = arrs[0]
    s18a = xo.load_18s_levels(b); s18 = s18a[0] if s18a else None
    Hpx, Wpx = full.shape[-2], full.shape[-1]
    xa, xb, ya, yb = cx-half, cx+half, cy-half, cy+half
    X0, X1 = max(0, int(xa/xo.PX)), min(Wpx, int(xb/xo.PX)); Y0, Y1 = max(0, int(ya/xo.PX)), min(Hpx, int(yb/xo.PX))
    rgb = xo.composite_dapi_18s(np.asarray(full[Y0:Y1, X0:X1]), np.asarray(s18[Y0:Y1, X0:X1]) if s18 is not None else None)
    ext = [X0*xo.PX, X1*xo.PX, Y1*xo.PX, Y0*xo.PX]

    inc = cells[(cells["x_centroid"].between(xa, xb)) & (cells["y_centroid"].between(ya, yb))]
    incore = set(inc["core"])
    dmap = cells.set_index("core")["domain"]; mmap = cells.set_index("core")["is_MN"]
    cb_all, key, polys_ok, bdiag = xo.resolve_boundary_join(b, cells)

    polys, fcs, isMN, doms_present = [], [], [], set()
    if polys_ok and cb_all is not None:
        cbc = cb_all[cb_all["k"].isin(incore)]
        for kk, sub in cbc.groupby("k", sort=False):
            d = dmap.get(kk); v = sub[["vertex_x", "vertex_y"]].to_numpy()
            if d is None or len(v) < 3: continue
            polys.append(v); fcs.append(palette.get(str(d), "0.6")); isMN.append(bool(mmap.get(kk, False)))
            doms_present.add(str(d))
        mnp = [p for p, m in zip(polys, isMN) if m]
    else:
        doms_present = set(inc["domain"].astype(str).unique())
        mnp = None
    txd = {g: xo.load_transcripts(b, [g])[g] for g in GENES}

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.4)); fig.patch.set_facecolor("black")
    for ax, g in zip(axes, GENES):
        ax.set_facecolor("black"); ax.imshow(rgb, origin="upper", extent=ext, zorder=0)
        if polys_ok:
            ax.add_collection(PolyCollection(polys, facecolors=fcs, edgecolors="white", linewidths=0.4, alpha=0.32, zorder=2))
            if mnp: ax.add_collection(PolyCollection(mnp, facecolors="none", edgecolors=xo.MN_EDGE, linewidths=1.9, zorder=5))
        else:
            cvals = inc["domain"].astype(str).map(palette).fillna("0.6").tolist()
            ax.scatter(inc["x_centroid"].to_numpy(), inc["y_centroid"].to_numpy(), s=22, c=cvals,
                       marker="o", edgecolors="white", linewidths=0.2, alpha=0.6, zorder=2)
            mnin = inc[inc["is_MN"]]
            if len(mnin):
                ax.scatter(mnin["x_centroid"].to_numpy(), mnin["y_centroid"].to_numpy(), s=150,
                           facecolors="none", edgecolors=xo.MN_EDGE, linewidths=2.0, zorder=5)
        xy = txd[g]
        m = (xy[:, 0] >= xa) & (xy[:, 0] <= xb) & (xy[:, 1] >= ya) & (xy[:, 1] <= yb); xyc = xy[m]
        ax.scatter(xyc[:, 0], xyc[:, 1], s=GSIZE[g], c=TX, alpha=0.95, edgecolors="black", linewidths=0.4, zorder=7)
        ax.set_xlim(xa, xb); ax.set_ylim(yb, ya); ax.set_aspect("equal"); ax.axis("off")
        ttl = g if tcounts is None else f"{g}   (in-crop {len(xyc)}; in central MN {tcounts[g]})"
        ax.set_title(ttl, color="white", fontsize=15)
    axes[0].plot([xa+half*0.12, xa+half*0.12+50], [yb-half*0.12, yb-half*0.12], color="white", lw=3, zorder=20)
    axes[0].text(xa+half*0.12+25, yb-half*0.12-half*0.06, "50 um", color="white", ha="center", va="bottom", fontsize=11, zorder=20)
    present = sorted(doms_present, key=xo._dom_sort_key)
    h = [Line2D([0],[0], marker='s', linestyle='none', markersize=9, markerfacecolor=palette.get(d, "0.6"), markeredgecolor='none', label=d) for d in present]
    h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=9, markerfacecolor='none', markeredgecolor=xo.MN_EDGE, markeredgewidth=1.8, label='motor neuron'))
    h.append(Line2D([0],[0], marker='o', linestyle='none', markersize=9, markerfacecolor=TX, markeredgecolor='black', markeredgewidth=0.5, label='transcript'))
    axes[2].legend(handles=h, loc='center left', bbox_to_anchor=(1.005, 0.5), frameon=False, labelcolor='white', fontsize=10, title='Novae domain').get_title().set_color('white')
    mode = "" if polys_ok else "  [centroid fallback: boundaries stale]"
    fig.suptitle(f"{sample}  |  same ventral-horn field, three genes  (Xenium; DAPI+18S)  --  MN marker MNX1 & STMN2 vs BCL6{mode}", color="white", fontsize=16, y=0.99)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}/{sample}__MNX1_BCL6_STMN2_matched.{e}", dpi=430, bbox_inches="tight", facecolor="black")
    plt.close(fig); print(f"[{sample}] wrote matched triple panel", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=float, default=105.0)
    ap.add_argument("--cx", type=float, default=None); ap.add_argument("--cy", type=float, default=None)
    a = ap.parse_args()
    main(a.sample, a.out, half=a.half, cx=a.cx, cy=a.cy)
