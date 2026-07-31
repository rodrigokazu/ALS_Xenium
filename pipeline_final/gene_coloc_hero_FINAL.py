#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/gene_coloc_hero_FINAL.py
#
# Quantifies how much a given gene co-localises with motor neurons across the cohort, then
# renders close-ups of the best examples. Run for MNX1, BCL6 and STMN2. Those three are the
# marker panel for the FANS sorting collaboration.
#
# The metric is deliberately simple. A transcript with qv >= 20 counts as being in a motor
# neuron when Ranger assigned it to a cell that is_MN marks True. Per-motor-neuron counts
# drive both the ranking and the choice of hero fields, so the figure shows what the number
# says.
#
# Two _final specifics. transcripts.parquet cell_id is int64 with unassigned encoded as 0
# rather than the old "UNASSIGNED" string, and _drop_unassigned handles 0, "-1" and the string
# form so it survives either. Bundle resolution comes from final_config, the authoritative
# dedup, and not from local name matching.
#
# Gated on is_MN. With annotation pending the co-localisation is undefined and the script says
# so instead of emitting zeros that look like a negative result.
#
# Worth remembering when reading the output: STMN2 is broadly neuronal, not motor-neuron
# specific. That is why the abstract validated on MNX1 post hoc instead of gating on it.
# ========================================================================================

"""gene_coloc_hero_FINAL.py -- quantify a gene's co-localisation with motor neurons across
the _final Xenium cohort, then render "hero" close-ups centred on <gene>+ motor neurons for
the top-ranked samples. Adapted from gene_coloc_hero.py for Ranger_procd_mw_final.

Co-localisation metric (UNCHANGED): a <gene> transcript (qv>=20) is "in an MN" when Ranger
assigned it (transcripts.parquet cell_id) to a cell whose integer core-id is is_MN==True in
the MN-corrected Novae obs. Per-MN counts drive both the ranking and hero selection.

_final deltas:
  * transcripts.parquet cell_id is int64 with UNASSIGNED == 0 (NOT the "UNASSIGNED" string);
    _drop_unassigned() handles 0 / "-1" / "UNASSIGNED" robustly. core_id(int) -> str is fine.
  * cells.parquet integer cell_id joins obs 1:1 (via xenium_overlay_FINAL.load_cells).
  * bundle resolution + paths come from final_config (authoritative dedup).
  * is_MN GATE: this whole metric REQUIRES is_MN. Samples with is_MN.sum()==0 (annotation
    pending) get a zero stat row flagged is_MN_pending=1 and are NOT rendered; if NO sample
    has any MN the ranking/heroes are skipped with a clear message.
  * hero polygons obey xenium_overlay_FINAL's boundary-provenance guard (centroid fallback
    when the _final boundaries are stale).

Reads bundles read-only. PRE-QC caveat applies (illustrative, not QC'd quant). Env:
xenium_vistools."""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as fc
import xenium_overlay_FINAL as xo
import hero_crop_FINAL as hc
import nature_style as ns

HDIR = str(fc.NOVAE_JOINT_DIR / "per_sample")


def _sample_dir_map():
    return {lab: str(p) for lab, p in fc.discover_samples(verbose=False)}


def resolve_bundle(sample, dmap=None):
    dmap = dmap or _sample_dir_map()
    return dmap.get(sample)


def _drop_unassigned(df):
    """Drop transcripts not assigned to a cell. _final codes UNASSIGNED as integer 0;
    older bundles used the string 'UNASSIGNED' or '-1'. Handle all."""
    cid = df["cell_id"]
    if np.issubdtype(cid.dtype, np.number):
        return df[cid != 0]
    s = cid.astype(str)
    return df[~s.isin(["0", "-1", "UNASSIGNED"])]


def gene_per_core(bundle, gene, qv_min=20):
    """Series: core-id -> count of <gene> transcripts (qv>=20) Ranger assigned to that cell,
    plus g_total = all qv>=20 transcripts of that gene (assigned + unassigned)."""
    df = pd.read_parquet(f"{bundle}/transcripts.parquet",
                         columns=["feature_name", "cell_id", "qv"],
                         filters=[("feature_name", "in", {gene})])
    df = df[df["qv"] >= qv_min]
    total = len(df)
    df = _drop_unassigned(df)
    df = df.copy()
    df["core"] = df["cell_id"].map(xo.core_id)
    return df["core"].value_counts(), total


def sample_stats(sample, bundle, h5, gene):
    dkey = xo.pick_domain_key(h5)
    obs = xo.read_obs(h5, [dkey, "is_MN"])
    dom = obs[dkey]
    mn = xo._as_bool(obs["is_MN"]) if "is_MN" in obs else pd.Series(False, index=dom.index)
    cells = xo.load_cells(bundle, dom, mn)
    per_core, g_total = gene_per_core(bundle, gene)
    cells["g"] = cells["core"].map(per_core).fillna(0).astype(int)
    mn_cells = cells[cells["is_MN"]].copy()
    non_mn = cells[~cells["is_MN"]]
    n_mn = len(mn_cells); n_non = len(non_mn)
    n_mn_pos = int((mn_cells["g"] >= 1).sum())
    g_in_mn = int(mn_cells["g"].sum())
    rate_mn = g_in_mn / max(n_mn, 1)
    rate_non = int(non_mn["g"].sum()) / max(n_non, 1)
    row = dict(
        sample=sample,
        is_MN_pending=int(n_mn == 0),
        n_cells=len(cells), n_MN=n_mn,
        n_MN_pos=n_mn_pos,
        frac_MN_pos=round(n_mn_pos / max(n_mn, 1), 4),
        gene_total_qv20=g_total,
        gene_in_cells=int(cells["g"].sum()),
        gene_in_MN=g_in_mn,
        frac_gene_in_MN=round(g_in_mn / max(g_total, 1), 4),
        mean_per_MN=round(rate_mn, 3),
        mean_per_nonMN=round(rate_non, 4),
        enrich_MN_vs_nonMN=round(rate_mn / rate_non, 2) if rate_non > 0 else np.nan,
        max_in_one_MN=int(mn_cells["g"].max()) if n_mn else 0,
    )
    has_area = "cell_area" in mn_cells.columns
    cand = mn_cells[mn_cells["g"] >= 1][["core", "x_centroid", "y_centroid", "g"] +
                                        (["cell_area"] if has_area else [])].copy()
    return row, cand, has_area


def pick_hero_centers(cand, has_area, topk, min_sep, min_g=1):
    c = cand[cand["g"] >= min_g].copy()
    if not len(c):
        return []
    if has_area and c["cell_area"].std() > 0:
        z = (c["cell_area"] - c["cell_area"].mean()) / c["cell_area"].std()
        c["score"] = c["g"] * (1.0 + z.clip(-1, 3))
    else:
        c["score"] = c["g"].astype(float)
    c = c.sort_values(["score", "g"], ascending=False)
    picked = []
    for _, r in c.iterrows():
        p = np.array([r["x_centroid"], r["y_centroid"]])
        if all(np.hypot(*(p - q[:2])) > min_sep for q in picked):
            picked.append((r["x_centroid"], r["y_centroid"], int(r["g"])))
        if len(picked) >= topk:
            break
    return picked


def render_sample(sample, bundle, h5, out, centers, gene, half=120.0):
    ns.apply_style()
    dkey = xo.pick_domain_key(h5)
    obs = xo.read_obs(h5, [dkey, "is_MN"])
    dom = obs[dkey]
    mn = xo._as_bool(obs["is_MN"]) if "is_MN" in obs else pd.Series(False, index=dom.index)
    cells = xo.load_cells(bundle, dom, mn)
    palette = xo.build_domain_palette(dom.dropna().unique())
    arrs = xo.load_dapi_levels(bundle); full = arrs[0]
    s18_arrs = xo.load_18s_levels(bundle); s18_full = s18_arrs[0] if s18_arrs else None
    txd = xo.load_transcripts(bundle, [gene])
    Hpx, Wpx = full.shape[-2], full.shape[-1]
    cb_all, key, polys_ok, bdiag = xo.resolve_boundary_join(bundle, cells)
    for j, (cx, cy, n) in enumerate(centers, 1):
        hc._render(sample, f"c{j}_x{n}", cx, cy, half, full, s18_full, Wpx, Hpx,
                   cells, cb_all, palette, polys_ok, out, txd=txd)


def main(samples, out, gene, topN=8, topk=6, min_sep=250.0, half=120.0, min_g=2, render=True):
    os.makedirs(out, exist_ok=True)
    dmap = _sample_dir_map()
    rows, cand_cache = [], {}
    for s in samples:
        b = resolve_bundle(s, dmap); h5 = f"{HDIR}/{s}__niches.h5ad"
        if b is None or not os.path.exists(h5):
            print(f"[SKIP] {s}: bundle={b} h5exists={os.path.exists(h5)} "
                  f"(Novae _FINAL per-sample h5 pending?)"); continue
        try:
            row, cand, has_area = sample_stats(s, b, h5, gene)
            rows.append(row); cand_cache[s] = (b, h5, cand, has_area)
            tag = "  [is_MN PENDING]" if row["is_MN_pending"] else ""
            print(f"[STAT] {s}: MN={row['n_MN']} {gene}+MN={row['n_MN_pos']} "
                  f"({row['frac_MN_pos']:.0%}) enrich={row['enrich_MN_vs_nonMN']} "
                  f"maxInOneMN={row['max_in_one_MN']} fracInMN={row['frac_gene_in_MN']:.0%}{tag}", flush=True)
        except Exception as e:
            print(f"[ERR stat] {s}: {e}")
    if not rows:
        print(f"\n[{gene}] no samples produced stats (Novae _FINAL h5ads pending?). Nothing written.")
        return
    df = pd.DataFrame(rows).sort_values(["frac_MN_pos", "n_MN_pos"], ascending=False)
    csv = f"{out}/{gene}_MN_colocalisation_stats.csv"
    df.to_csv(csv, index=False)
    print(f"\n=== {gene}-in-MN co-localisation ranking ===")
    print(df.to_string(index=False))
    print(f"\nwrote {csv}")
    # ---- is_MN GATE ----
    total_mn = int(df["n_MN"].sum())
    if total_mn == 0:
        print(f"\n[{gene}] is_MN.sum()==0 across ALL samples (annotation pending) -> "
              f"co-localisation is is_MN-gated; NO heroes rendered. Stats CSV written "
              f"(all zero) for provenance. Re-run after Marcel's _final MN annotation lands.")
        return
    if not render:
        return
    ranked = [s for s in df["sample"].tolist() if cand_cache.get(s) and
              df.loc[df["sample"] == s, "n_MN_pos"].iloc[0] >= 3]
    top = ranked[:topN]
    print(f"\nRendering heroes for top {len(top)}: {top}")
    for s in top:
        b, h5, cand, has_area = cand_cache[s]
        centers = pick_hero_centers(cand, has_area, topk, min_sep, min_g=min_g)
        if not centers:
            centers = pick_hero_centers(cand, has_area, topk, min_sep, min_g=1)
        if not centers:
            print(f"[{s}] no {gene}+ MN centres -> skip render"); continue
        print(f"\n===== {s}: {len(centers)} {gene}+ MN heroes =====", flush=True)
        try:
            render_sample(s, b, h5, out, centers, gene, half=half)
        except Exception as e:
            print(f"[ERR render] {s}: {e}")
    print("\nALL DONE ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topN", type=int, default=8)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--min-sep", dest="min_sep", type=float, default=250.0)
    ap.add_argument("--half", type=float, default=120.0)
    ap.add_argument("--min-g", dest="min_g", type=int, default=2)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", nargs="*", default=None)
    a = ap.parse_args()
    # default sample set = the authoritative discovered labels (h5 presence checked in main)
    samples = a.samples or sorted({lab for lab, _ in fc.discover_samples(verbose=False)})
    main(samples, a.out, a.gene, topN=a.topN, topk=a.topk, min_sep=a.min_sep, half=a.half,
         min_g=a.min_g, render=not a.no_render)
