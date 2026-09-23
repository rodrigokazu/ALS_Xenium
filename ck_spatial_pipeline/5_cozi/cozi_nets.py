#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/cozi_nets.py
#
# Turns cozi_als tables into directed and undirected cell-type networks. An arc weighs the
# mean per-section z and exists when a two-sided Wilcoxon signed-rank test against 0 passes
# BH < 0.05 within its group (pooled, ALS, control). Undirected edges collapse sign-
# concordant arc pairs. It also re-runs COZI on cell_type_v2 (12 nodes, adds Fibroblast).
#
# This is the post-review script. The SCG output 20260917_113204_ncv3 came from the pre-review
# version and is provenance only. In the differential GML, padj is the pooled existence test.
# The case against control value is cc_padj.
# ========================================================================================

"""
Turn a COZI run (cozi_als.py output) into DIRECTED and UNDIRECTED cell-type networks.

Input  = the per-section ordered-pair table cozi_persection_<labelset>.csv
         (index_cell_type, neighbor_cell_type, zscore, cond_ratio, counts, sample, status, group)
         + the case x control table cozi_case_control_<labelset>.csv.
Output = per (scope, label set):
  nodes.csv                       one row per cell type: cell counts, section coverage, self z
  edges_directed_<group>.csv      one row per ORDERED pair A->B, group in pooled / ALS / control
                                  weight = mean per-section COZI z; Wilcoxon signed-rank across
                                  sections tests z != 0; BH within the arc set
  edges_undirected_<group>.csv    reciprocal arcs collapsed: z_ab, z_ba, z_mean, asymmetry
  gml/*.gml                       full / attraction / avoidance / differential graphs
  gml/persection/*.gml            one directed graph per section
  matrices/*.csv                  square z / cond_ratio / padj matrices
  figs/*.png                      z heat-maps + circular network drawings
  summary.json
Top level: granularity.json + GRANULARITY_REPORT.md.

--rerun-obj <h5ad> --rerun-celltype-col cell_type_v2   re-runs COZI per section on another
label column (astrocytes split by WDR49, exactly as cozi_als.py) so the network can be built
at a second granularity, then compares the two.

Pure-python GML writer (no networkx dependency), same block style as
Control/Nets/make_geneonly_gml.py. READ-ONLY on every input.
"""
import argparse, json, os, sys, time, math, re
import numpy as np
import pandas as pd

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def sec(m): print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78, flush=True)

def bh(p):
    p = np.asarray(p, float); n = len(p)
    q = np.full(n, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0: return q
    pp = p[ok]; m = len(pp); o = np.argsort(pp)
    r = pp[o] * m / (np.arange(m) + 1)
    qq = np.empty(m); qq[o] = np.minimum.accumulate(r[::-1])[::-1]
    q[ok] = np.clip(qq, 0, 1)
    return q

# --------------------------------------------------------------------------- GML
def _gml_val(v):
    if isinstance(v, (bool, np.bool_)): return "1" if v else "0"
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v): return '"nan"'
        return repr(float(v))
    s = str(v).replace('\\', '\\\\').replace('"', "'")
    return f'"{s}"'

def _gml_key(k):
    k = re.sub(r"[^A-Za-z0-9_]", "_", str(k))
    return k if k[0].isalpha() else "x_" + k

def write_gml(path, directed, graph_attrs, nodes, edges, ids=None):
    """nodes: list of (label, {attr}); edges: list of (src_label, dst_label, {attr}).
    ids: fixed label->id map so a label keeps the same id in every file of a run."""
    if ids is None:
        ids = {lab: i for i, (lab, _) in enumerate(nodes)}
    with open(path, "w") as fh:
        fh.write("graph [\n")
        fh.write(f"  directed {1 if directed else 0}\n")
        for k, v in graph_attrs.items():
            fh.write(f"  {_gml_key(k)} {_gml_val(v)}\n")
        for lab, at in nodes:
            fh.write(f"  node [\n    id {ids[lab]}\n    label {_gml_val(lab)}\n")
            for k, v in at.items():
                fh.write(f"    {_gml_key(k)} {_gml_val(v)}\n")
            fh.write("  ]\n")
        for s, t, at in edges:
            fh.write(f"  edge [\n    source {ids[s]}\n    target {ids[t]}\n")
            for k, v in at.items():
                fh.write(f"    {_gml_key(k)} {_gml_val(v)}\n")
            fh.write("  ]\n")
        fh.write("]\n")
    return len(nodes), len(edges)

def read_gml_counts(path):
    """Content gate: re-parse a written GML and count node / edge blocks + directed flag."""
    txt = open(path).read()
    n_nodes = len(re.findall(r"\bnode\s*\[", txt))
    n_edges = len(re.findall(r"\bedge\s*\[", txt))
    d = re.search(r"\bdirected\s+([01])", txt)
    return n_nodes, n_edges, int(d.group(1)) if d else None

# --------------------------------------------------------------------------- stats
def arc_stats(per, group_mask, alpha):
    """Per ORDERED pair: mean/sd/median z across sections, sign consistency, Wilcoxon z!=0."""
    from scipy.stats import wilcoxon
    sub = per[group_mask]
    rows = []
    for (a, b), g in sub.groupby(["index_cell_type", "neighbor_cell_type"], sort=True):
        z = g["zscore"].astype(float).dropna().values
        cr = g["cond_ratio"].astype(float).dropna().values
        n = len(z)
        p = np.nan
        if n >= 5 and np.any(z != 0):
            try: p = float(wilcoxon(z, alternative="two-sided", zero_method="wilcox").pvalue)
            except Exception: p = np.nan
        rows.append(dict(index_cell_type=a, neighbor_cell_type=b, n_sections=n,
                         min_p_achievable=float(2.0 / 2 ** n) if n else np.nan,
                         z_mean=float(np.mean(z)) if n else np.nan,
                         z_sd=float(np.std(z, ddof=1)) if n > 1 else np.nan,
                         z_median=float(np.median(z)) if n else np.nan,
                         frac_pos=float(np.mean(z > 0)) if n else np.nan,
                         cr_mean=float(np.mean(cr)) if len(cr) else np.nan,
                         n_index_cells_mean=float(g["n_index_cells"].mean()),
                         n_edges_mean=float(g["n_index_neighbor_edges"].mean()),
                         wilcoxon_p=p))
    d = pd.DataFrame(rows)
    d["is_self"] = d.index_cell_type == d.neighbor_cell_type
    arcs = ~d.is_self
    d["padj"] = np.nan
    d.loc[arcs, "padj"] = bh(d.loc[arcs, "wilcoxon_p"].values)
    d["significant"] = (d.padj < alpha)
    d["direction"] = np.where(d.z_mean > 0, "attraction", "avoidance")
    return d

def collapse_undirected(d):
    """A->B and B->A into one row with sorted (A,B); self rows dropped."""
    key = {(r.index_cell_type, r.neighbor_cell_type): r for r in d.itertuples(index=False)}
    rows, seen = [], set()
    for (a, b), r in key.items():
        if a == b: continue
        k = tuple(sorted((a, b)))
        if k in seen: continue
        seen.add(k)
        a2, b2 = k
        ab = key.get((a2, b2)); ba = key.get((b2, a2))
        if ab is None or ba is None: continue
        z_ab, z_ba = ab.z_mean, ba.z_mean
        rows.append(dict(node_a=a2, node_b=b2, z_ab=z_ab, z_ba=z_ba,
                         z_mean=(z_ab + z_ba) / 2, z_absmean=(abs(z_ab) + abs(z_ba)) / 2,
                         z_max=max(z_ab, z_ba, key=abs), asymmetry=z_ab - z_ba,
                         sign_concordant=bool(np.sign(z_ab) == np.sign(z_ba)),
                         cr_ab=ab.cr_mean, cr_ba=ba.cr_mean,
                         padj_ab=ab.padj, padj_ba=ba.padj,
                         sig_ab=bool(ab.significant), sig_ba=bool(ba.significant),
                         sig_both=bool(ab.significant and ba.significant),
                         sig_either=bool(ab.significant or ba.significant),
                         n_sections=min(ab.n_sections, ba.n_sections)))
    u = pd.DataFrame(rows)
    if len(u):
        u["direction"] = np.where(u.z_mean > 0, "attraction", "avoidance")
        u = u.sort_values("z_mean", ascending=False)
    return u

# --------------------------------------------------------------------------- figures
def circular_positions(labels):
    n = len(labels); ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return {l: (math.cos(a), math.sin(a)) for l, a in zip(labels, ang)}

def draw_network(path, title, labels, edges, directed, node_size, subtitle=""):
    """edges: list of (a, b, weight). Width ~ |w|, colour = sign. Circular layout."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    pos = circular_positions(labels)
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.set_aspect("equal"); ax.axis("off")
    wmax = max([abs(w) for _, _, w in edges] + [1e-9])
    for a, b, w in sorted(edges, key=lambda e: abs(e[2])):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        col = "#B23A48" if w > 0 else "#3A6EA5"
        lw = 0.6 + 5.0 * abs(w) / wmax
        if directed:
            p = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.15",
                                arrowstyle="-|>", mutation_scale=14, lw=lw, color=col,
                                alpha=0.85, shrinkA=16, shrinkB=16, zorder=1)
            ax.add_patch(p)
        else:
            ax.plot([x1, x2], [y1, y2], lw=lw, color=col, alpha=0.85, zorder=1,
                    solid_capstyle="round")
    smax = max(node_size.values()) if node_size else 1
    for l in labels:
        x, y = pos[l]
        s = 160 + 1400 * math.sqrt(node_size.get(l, 0) / smax)
        ax.scatter([x], [y], s=s, color="#F2E9DC", edgecolor="#2B2B2B", lw=1.0, zorder=3)
        ha = "left" if x > 0.05 else ("right" if x < -0.05 else "center")
        va = "bottom" if y > 0.05 else ("top" if y < -0.05 else "center")
        ax.text(x * 1.16, y * 1.16, l.replace("_", " "), ha=ha, va=va, fontsize=9.5,
                color="#2B2B2B", zorder=4)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)

def heatmap(path, M, title, cmap, center0=True, fmt="{:.2f}", annot=True):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.62 * M.shape[1] + 2.4, 0.55 * M.shape[0] + 1.8))
    v = np.nanmax(np.abs(M.values)) if center0 else None
    im = ax.imshow(M.values, cmap=cmap, vmin=-v if center0 else None, vmax=v if center0 else None,
                   aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([c.replace("_", " ") for c in M.columns],
                                                          rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([c.replace("_", " ") for c in M.index],
                                                          fontsize=8.5)
    ax.set_xlabel("neighbour B"); ax.set_ylabel("index A")
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                x = M.values[i, j]
                if not np.isnan(x):
                    ax.text(j, i, fmt.format(x), ha="center", va="center", fontsize=6.4,
                            color="#111111" if abs(x) < (v or 1) * 0.6 else "#FFFFFF")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title(title, loc="left", fontsize=11)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- build
def build_networks(per, cc, out, tag, meta, alpha=0.05, draw=True):
    """per = per-section table, cc = case-control table (may be None). Writes everything."""
    os.makedirs(out, exist_ok=True)
    G = os.path.join(out, "gml"); os.makedirs(os.path.join(G, "persection"), exist_ok=True)
    MX = os.path.join(out, "matrices"); os.makedirs(MX, exist_ok=True)
    FG = os.path.join(out, "figs"); os.makedirs(FG, exist_ok=True)

    per = per.copy()
    per["group"] = np.where(per["status"].astype(str) == "control", "control", "ALS")
    labels = sorted(set(per.index_cell_type) | set(per.neighbor_cell_type))
    global_ids = {l: i for i, l in enumerate(labels)}
    sections = sorted(per["sample"].unique())
    grp_of = per.drop_duplicates("sample").set_index("sample")["group"].to_dict()
    n_grp = pd.Series(grp_of).value_counts().to_dict()
    log(f"{tag}: {len(labels)} labels, {len(sections)} sections {n_grp}")

    # ---- nodes
    cnt = per.drop_duplicates(["sample", "index_cell_type"])
    piv = cnt.pivot(index="index_cell_type", columns="sample", values="n_index_cells")
    tot_per_section = piv.sum(axis=0)
    node_rows = []
    for l in labels:
        row = piv.loc[l] if l in piv.index else pd.Series(dtype=float)
        pres = row.dropna()
        node_rows.append(dict(label=l, n_cells_total=int(pres.sum()), n_sections_present=int(len(pres)),
                              n_sections_total=len(sections),
                              min_cells_per_section=int(pres.min()) if len(pres) else 0,
                              median_cells_per_section=float(pres.median()) if len(pres) else 0.0,
                              mean_fraction_of_section=float((row / tot_per_section).mean()),
                              n_sections_ALS=int(sum(grp_of[s] == "ALS" for s in pres.index)),
                              n_sections_control=int(sum(grp_of[s] == "control" for s in pres.index))))
    nodes = pd.DataFrame(node_rows).set_index("label")

    # ---- arcs per group
    groups = {"pooled": np.ones(len(per), bool),
              "ALS": (per.group == "ALS").values, "control": (per.group == "control").values}
    D, U = {}, {}
    for gname, mask in groups.items():
        d = arc_stats(per, mask, alpha)
        if cc is not None and gname == "pooled":
            c2 = cc.rename(columns={"p": "cc_p", "padj": "cc_padj", "significant": "cc_significant",
                                    "U": "cc_U"})
            d = d.merge(c2[["index_cell_type", "neighbor_cell_type", "n_ALS", "n_control", "z_ALS",
                            "z_control", "delta_z", "cr_ALS", "cr_control", "cc_U", "cc_p",
                            "cc_padj", "cc_significant"]],
                        on=["index_cell_type", "neighbor_cell_type"], how="left")
        D[gname] = d
        # self z -> node attribute
        selfz = d[d.is_self].set_index("index_cell_type")["z_mean"]
        nodes[f"self_z_{gname}"] = selfz.reindex(nodes.index)
        d[~d.is_self].drop(columns=["is_self"]).sort_values("z_mean", ascending=False) \
            .to_csv(os.path.join(out, f"edges_directed_{gname}.csv"), index=False)
        U[gname] = collapse_undirected(d)
        U[gname].to_csv(os.path.join(out, f"edges_undirected_{gname}.csv"), index=False)
        # matrices
        for col, nm in (("z_mean", "z"), ("cr_mean", "cond_ratio"), ("padj", "padj")):
            M = d.pivot(index="index_cell_type", columns="neighbor_cell_type", values=col) \
                 .reindex(index=labels, columns=labels)
            M.to_csv(os.path.join(MX, f"{nm}_{gname}.csv"))
    nodes.to_csv(os.path.join(out, "nodes.csv"))

    # ---- GML writer helpers
    base_attrs = dict(meta)
    base_attrs.update(label_set=tag, n_sections=len(sections), alpha=alpha,
                      sections_ALS=int(n_grp.get("ALS", 0)), sections_control=int(n_grp.get("control", 0)),
                      edge_weight="mean per-section COZI conditional z (normalize_zscore=True); signed",
                      note="self pairs A->A are node attributes self_z_*, not edges")
    node_attr_cols = ["n_cells_total", "n_sections_present", "min_cells_per_section",
                      "mean_fraction_of_section", "self_z_pooled", "self_z_ALS", "self_z_control"]
    def node_list(present=None):
        out_nodes = []
        for l in labels:
            if present is not None and l not in present: continue
            at = {k: nodes.loc[l, k] for k in node_attr_cols}
            at["node_type"] = "cell_type"
            out_nodes.append((l, at))
        return out_nodes

    written = {}
    def wgml(name, directed, edges, extra, drop_isolates=False):
        present = None
        if drop_isolates:
            present = set([e[0] for e in edges] + [e[1] for e in edges])
        attrs = dict(base_attrs); attrs.update(extra)
        p = os.path.join(G, f"cozi_{tag}_{name}.gml")
        n, m = write_gml(p, directed, attrs, node_list(present), edges, ids=global_ids)
        rn, rm, rd = read_gml_counts(p)
        assert (rn, rm, rd) == (n, m, int(directed)), f"GML content gate failed for {p}"
        written[name] = dict(nodes=n, edges=m, directed=bool(directed), file=os.path.relpath(p, out))
        return p

    dir_cols = ["z_mean", "z_sd", "z_median", "frac_pos", "n_sections", "cr_mean", "wilcoxon_p",
                "padj", "significant", "direction"]
    cc_cols = ["z_ALS", "z_control", "delta_z", "cr_ALS", "cr_control", "cc_p", "cc_padj",
               "cc_significant"]
    def dir_edges(d, keep):
        ed = []
        for r in d[keep & ~d.is_self].itertuples(index=False):
            at = {"weight": r.z_mean, "abs_weight": abs(r.z_mean)}
            for c in dir_cols: at[c] = getattr(r, c)
            for c in cc_cols:
                if hasattr(r, c): at[c] = getattr(r, c)
            ed.append((r.index_cell_type, r.neighbor_cell_type, at))
        return ed
    und_cols = ["z_ab", "z_ba", "z_mean", "z_absmean", "z_max", "asymmetry", "sign_concordant",
                "cr_ab", "cr_ba", "padj_ab", "padj_ba", "sig_ab", "sig_ba", "sig_both", "sig_either",
                "n_sections", "direction"]
    def und_edges(u, keep):
        ed = []
        for r in u[keep].itertuples(index=False):
            at = {"weight": r.z_mean, "abs_weight": abs(r.z_mean)}
            for c in und_cols: at[c] = getattr(r, c)
            ed.append((r.node_a, r.node_b, at))
        return ed

    power = {}
    for gname in groups:
        dd = D[gname][~D[gname].is_self]
        power[gname] = dict(n_sections_min=int(dd.n_sections.min()), n_sections_max=int(dd.n_sections.max()),
                            worst_min_p_achievable=float(dd.min_p_achievable.max()),
                            min_padj=float(dd.padj.min()), n_significant=int(dd.significant.sum()),
                            underpowered=bool(dd.significant.sum() == 0 and dd.padj.min() > alpha))
        if power[gname]["underpowered"]:
            log(f"{tag} [{gname}]: UNDERPOWERED, {power[gname]['n_sections_min']}-{power[gname]['n_sections_max']} "
                f"sections, min padj {power[gname]['min_padj']:.3f}; graphs written with 0 edges")
    for gname in groups:
        d, u = D[gname], U[gname]
        pw = dict(n_sections_min=power[gname]["n_sections_min"], n_sections_max=power[gname]["n_sections_max"],
                  worst_min_p_achievable=power[gname]["worst_min_p_achievable"],
                  underpowered=power[gname]["underpowered"])
        full = np.ones(len(d), bool)
        wgml(f"directed_{gname}_full", True, dir_edges(d, full),
             dict(group=gname, variant="full", edge_rule="every ordered pair", **pw))
        wgml(f"directed_{gname}_attraction", True,
             dir_edges(d, (d.z_mean > 0) & (d.padj < alpha)),
             dict(group=gname, variant="attraction",
                  edge_rule=f"z_mean > 0 and Wilcoxon-across-sections BH padj < {alpha}", **pw))
        wgml(f"directed_{gname}_avoidance", True,
             dir_edges(d, (d.z_mean < 0) & (d.padj < alpha)),
             dict(group=gname, variant="avoidance",
                  edge_rule=f"z_mean < 0 and Wilcoxon-across-sections BH padj < {alpha}", **pw))
        ufull = np.ones(len(u), bool)
        wgml(f"undirected_{gname}_full", False, und_edges(u, ufull),
             dict(group=gname, variant="full",
                  edge_rule="every unordered pair; weight = mean(z_ab, z_ba); sign_concordant flags pairs whose two arcs agree in sign", **pw))
        wgml(f"undirected_{gname}_attraction", False,
             und_edges(u, (u.z_mean > 0) & u.sig_either & u.sign_concordant),
             dict(group=gname, variant="attraction",
                  edge_rule="both arcs z > 0 (sign-concordant) and at least one direction significant; discordant pairs excluded", **pw))
        wgml(f"undirected_{gname}_attraction_strict", False,
             und_edges(u, (u.z_mean > 0) & u.sig_both & u.sign_concordant),
             dict(group=gname, variant="attraction_strict",
                  edge_rule="both arcs z > 0 (sign-concordant) and BOTH directions significant", **pw))

    # differential (ALS vs control) directed graph from the case-control test
    if cc is not None:
        d = D["pooled"]
        keep = d.cc_significant.fillna(False).astype(bool)
        ed = []
        for r in d[keep & ~d.is_self].itertuples(index=False):
            at = {"weight": r.delta_z, "abs_weight": abs(r.delta_z), "sign": "up_in_ALS" if r.delta_z > 0 else "down_in_ALS"}
            for c in dir_cols + cc_cols: at[c] = getattr(r, c)
            ed.append((r.index_cell_type, r.neighbor_cell_type, at))
        wgml("directed_differential_ALS_vs_control", True, ed,
             dict(group="ALS_vs_control", variant="differential",
                  edge_rule=f"Mann-Whitney U on per-section z, ALS vs control, BH padj < {alpha}; weight = delta_z"),
             drop_isolates=False)

    # per-section directed graphs
    for s in sections:
        g = per[per["sample"] == s]
        ed = []
        for r in g.itertuples(index=False):
            if r.index_cell_type == r.neighbor_cell_type or pd.isna(r.zscore): continue
            ed.append((r.index_cell_type, r.neighbor_cell_type,
                       {"weight": float(r.zscore), "abs_weight": abs(float(r.zscore)),
                        "cond_ratio": float(r.cond_ratio),
                        "n_index_cells": int(r.n_index_cells), "n_neighbor_cells": int(r.n_neighbor_cells),
                        "n_edges": int(r.n_index_neighbor_edges)}))
        present = set(g.index_cell_type) | set(g.neighbor_cell_type)
        attrs = dict(base_attrs); attrs.update(group=grp_of[s], variant="per_section", sample=s,
                                               status=str(g["status"].iloc[0]))
        p = os.path.join(G, "persection", f"cozi_{tag}_directed_{s}.gml")
        n_, m_ = write_gml(p, True, attrs, node_list(present), ed, ids=global_ids)
        rn, rm, rd = read_gml_counts(p)
        assert (rn, rm, rd) == (n_, m_, 1), f"GML content gate failed for {p}"

    # ---- figures
    if draw:
        nsz = nodes["n_cells_total"].to_dict()
        for gname in groups:
            d = D[gname]; u = U[gname]
            Mz = pd.read_csv(os.path.join(MX, f"z_{gname}.csv"), index_col=0)
            heatmap(os.path.join(FG, f"heatmap_z_{gname}.png"), Mz,
                    f"COZI z, mean of {int((per[groups[gname]]['sample']).nunique())} sections ({gname})", "RdBu_r")
            ka = (d.z_mean > 0) & (d.padj < alpha) & ~d.is_self
            draw_network(os.path.join(FG, f"net_directed_attraction_{gname}.png"),
                         f"Directed attraction network ({gname})", labels,
                         [(r.index_cell_type, r.neighbor_cell_type, r.z_mean) for r in d[ka].itertuples()],
                         True, nsz, f"arcs: z > 0 and BH padj < {alpha} across sections; width ~ z; node size ~ cells")
            kf = ~d.is_self
            draw_network(os.path.join(FG, f"net_directed_full_{gname}.png"),
                         f"Directed network, all arcs ({gname})", labels,
                         [(r.index_cell_type, r.neighbor_cell_type, r.z_mean) for r in d[kf].itertuples()],
                         True, nsz, "red = attraction (z > 0), blue = avoidance (z < 0); width ~ |z|")
            ku = (u.z_mean > 0) & u.sig_either & u.sign_concordant
            draw_network(os.path.join(FG, f"net_undirected_attraction_{gname}.png"),
                         f"Undirected attraction network ({gname})", labels,
                         [(r.node_a, r.node_b, r.z_mean) for r in u[ku].itertuples()],
                         False, nsz)
        if cc is not None:
            d = D["pooled"]
            Md = d.pivot(index="index_cell_type", columns="neighbor_cell_type", values="delta_z") \
                  .reindex(index=labels, columns=labels)
            heatmap(os.path.join(FG, "heatmap_delta_z_ALS_minus_control.png"), Md,
                    "COZI delta z, ALS minus control", "PuOr_r")
            kd = d.cc_significant.fillna(False).astype(bool) & ~d.is_self
            draw_network(os.path.join(FG, "net_directed_differential.png"),
                         "Differential network, ALS vs control", labels,
                         [(r.index_cell_type, r.neighbor_cell_type, r.delta_z) for r in d[kd].itertuples()],
                         True, nsz, f"arcs: Mann-Whitney BH padj < {alpha}; red = higher in ALS, blue = lower in ALS")

    # ---- summary
    d = D["pooled"]; arcs = d[~d.is_self]; u = U["pooled"]
    n = len(labels); n_arcs_possible = n * (n - 1)
    recip = 0
    sig = set(map(tuple, arcs.loc[arcs.significant, ["index_cell_type", "neighbor_cell_type"]].values))
    for a, b in sig:
        if (b, a) in sig: recip += 1
    summ = dict(
        label_set=tag, n_nodes=n, labels=labels, n_sections=len(sections), sections_by_group=n_grp,
        arcs_possible=n_arcs_possible, arcs_tested=int(len(arcs)),
        arcs_significant=int(arcs.significant.sum()),
        arcs_attraction_sig=int(((arcs.z_mean > 0) & arcs.significant).sum()),
        arcs_avoidance_sig=int(((arcs.z_mean < 0) & arcs.significant).sum()),
        arcs_significant_fraction=float(arcs.significant.mean()) if len(arcs) else np.nan,
        reciprocity_among_significant=float(recip / len(sig)) if sig else np.nan,
        one_way_significant_unordered=int((u.sig_ab ^ u.sig_ba).sum()) if len(u) else 0,
        both_way_significant_unordered=int(u.sig_both.sum()) if len(u) else 0,
        unordered_pairs=int(len(u)),
        undirected_sign_discordant_pairs=int((~u.sign_concordant).sum()) if len(u) else 0,
        undirected_sign_discordant_both_significant=int((~u.sign_concordant & u.sig_both).sum()) if len(u) else 0,
        undirected_attraction_edges=int(((u.z_mean > 0) & u.sig_either & u.sign_concordant).sum()) if len(u) else 0,
        undirected_attraction_strict_edges=int(((u.z_mean > 0) & u.sig_both & u.sign_concordant).sum()) if len(u) else 0,
        undirected_attraction_density=float(((u.z_mean > 0) & u.sig_either & u.sign_concordant).mean()) if len(u) else np.nan,
        group_power=power,
        differential_arcs=int((d.cc_significant.fillna(False).astype(bool) & ~d.is_self).sum()) if cc is not None else None,
        differential_self_pairs=(list(d.loc[d.cc_significant.fillna(False).astype(bool) & d.is_self, "index_cell_type"])
                                 if cc is not None else None),
        node_out_strength_pooled={l: float(arcs.loc[arcs.index_cell_type == l, "z_mean"].sum()) for l in labels},
        node_in_strength_pooled={l: float(arcs.loc[arcs.neighbor_cell_type == l, "z_mean"].sum()) for l in labels},
        node_sig_out_degree={l: int(arcs.loc[(arcs.index_cell_type == l) & arcs.significant].shape[0]) for l in labels},
        node_sig_in_degree={l: int(arcs.loc[(arcs.neighbor_cell_type == l) & arcs.significant].shape[0]) for l in labels},
        gml=written)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    log(f"{tag}: arcs sig {summ['arcs_significant']}/{summ['arcs_tested']} "
        f"(attr {summ['arcs_attraction_sig']}, avoid {summ['arcs_avoidance_sig']}); "
        f"undirected attraction {summ['undirected_attraction_edges']}/{summ['unordered_pairs']}; "
        f"differential {summ['differential_arcs']}")
    return summ, nodes, D, U

# --------------------------------------------------------------------------- COZI rerun (stage B)
def rerun_cozi(obj, celltype_col, outdir, gene="WDR49", gene_min=1, n_neighbors=10, n_perm=300,
               min_cell_count=10, seed=1, chunk=100000, restrict_vh=False, sample_col="sample"):
    """Same procedure as cozi_als.py, on another cell-type column. Returns (per, cc)."""
    import h5py
    from scipy.stats import mannwhitneyu
    from cozipy import run_cozi
    import cozipy
    os.makedirs(outdir, exist_ok=True); T = os.path.join(outdir, "tables"); os.makedirs(T, exist_ok=True)
    man = dict(object=obj, celltype_col=celltype_col, cozipy=cozipy.__version__,
               params=dict(gene=gene, gene_min=gene_min, n_neighbors=n_neighbors, n_permutations=n_perm,
                           min_cell_count=min_cell_count, seed=seed, restrict_vh=restrict_vh),
               started=time.strftime("%F %T"))
    f = h5py.File(obj, "r")
    def cat(k):
        g = f["obs"][k]
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array(c)[g["codes"][:]]
    def nb(k):
        g = f["obs"][k]; return g["values"][:] & ~g["mask"][:]
    ct = cat(celltype_col); samp = cat(sample_col); status = cat("status")
    xy = f["obsm"]["spatial"][:]
    mn_canon = nb("is_MN_v2"); in_vh = nb("in_VH_v2")
    var = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][:]]
    gj = var.index(gene)
    L = f["layers"]["counts"]; indptr = L["indptr"][:]; idx, dat = L["indices"], L["data"]
    n_obs = len(indptr) - 1; w = np.zeros(n_obs, dtype=np.int32)
    for s in range(0, n_obs, chunk):
        e = min(s + chunk, n_obs); a, b = indptr[s], indptr[e]
        ii = idx[a:b].astype(np.int64); dd = dat[a:b]; hit = ii == gj
        if hit.any():
            rows = np.repeat(np.arange(s, e), np.diff(indptr[s:e + 1]))[hit]
            np.add.at(w, rows, dd[hit].astype(np.int32))
    f.close()
    log(f"rerun on {celltype_col}: {n_obs:,} cells; {gene}+ {(w >= gene_min).sum():,}; is_MN_v2 {mn_canon.sum():,}")
    lab = ct.astype(object)
    m_astro = ct == "Astrocytes"
    lab[m_astro & (w >= gene_min)] = "Astro_WDR49pos"; lab[m_astro & (w < gene_min)] = "Astro_WDR49neg"
    # keep the curated motor neurons as their own node whatever the column calls them
    lab[mn_canon] = "Motor neurons"
    lab = lab.astype(str)
    keep = in_vh if restrict_vh else np.ones(n_obs, bool)
    lab, samp, status, xy = lab[keep], samp[keep], status[keep], xy[keep]
    vc = pd.Series(lab).value_counts(); log("label counts:\n" + vc.to_string())
    man["label_counts"] = {k: int(v) for k, v in vc.items()}; man["n_cells_used"] = int(keep.sum())
    dstat = pd.Series(status, index=samp).groupby(level=0).first()
    group = np.where(status == "control", "control", "ALS")
    dgrp = pd.Series(group, index=samp).groupby(level=0).first()
    per = []
    for s in sorted(pd.unique(samp)):
        m = samp == s; t0 = time.time()
        df = run_cozi(xy[m].astype(float), lab[m], nbh_def="knn", n_neighbors=n_neighbors,
                      n_permutations=n_perm, random_state=seed, normalize_zscore=True,
                      min_cell_count=min_cell_count)
        df["sample"] = s; df["status"] = dstat[s]; df["group"] = dgrp[s]; per.append(df)
        log(f"  {s:14s} n={m.sum():7,} pairs={len(df):5d}  {time.time() - t0:5.1f}s")
    allp = pd.concat(per, ignore_index=True)
    name = f"celltype_{celltype_col}_wdr49"
    allp.to_csv(os.path.join(T, f"cozi_persection_{name}.csv"), index=False)
    rows = []
    for (a, b), g in allp.groupby(["index_cell_type", "neighbor_cell_type"]):
        za = g.loc[g.group == "ALS", "zscore"].dropna().values
        zc = g.loc[g.group == "control", "zscore"].dropna().values
        if len(za) < 3 or len(zc) < 3: continue
        try: u, p = mannwhitneyu(za, zc, alternative="two-sided")
        except Exception: continue
        rows.append(dict(index_cell_type=a, neighbor_cell_type=b, n_ALS=len(za), n_control=len(zc),
                         z_ALS=float(np.mean(za)), z_control=float(np.mean(zc)),
                         delta_z=float(np.mean(za) - np.mean(zc)),
                         cr_ALS=float(g.loc[g.group == "ALS", "cond_ratio"].mean()),
                         cr_control=float(g.loc[g.group == "control", "cond_ratio"].mean()),
                         U=float(u), p=float(p)))
    cc = pd.DataFrame(rows)
    cc["padj"] = bh(cc["p"].values); cc["significant"] = cc["padj"] < 0.05
    cc = cc.sort_values("p"); cc.to_csv(os.path.join(T, f"cozi_case_control_{name}.csv"), index=False)
    man["results"] = dict(pairs=int(len(cc)), significant=int(cc.significant.sum()))
    man["finished"] = time.strftime("%F %T")
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1, default=str)
    return allp, cc, name, man

# --------------------------------------------------------------------------- granularity audit
def obs_label_audit(obj, cols, sample_col="sample", min_cell_count=10):
    """Per label-column: categories, per-section cell counts, how many labels fall under the
    COZI min_cell_count gate in any section. Reads obs only (fast, h5py)."""
    import h5py
    out = {}
    with h5py.File(obj, "r") as f:
        def cat(k):
            g = f["obs"][k]
            c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
            return np.array(c)[g["codes"][:]]
        samp = cat(sample_col)
        avail = [c for c in cols if c in f["obs"]]
        for c in avail:
            lab = cat(c)
            tab = pd.crosstab(pd.Series(lab, name="label"), pd.Series(samp, name="sample"))
            out[c] = dict(n_categories=int(tab.shape[0]), categories=list(tab.index),
                          n_cells={k: int(v) for k, v in tab.sum(axis=1).items()},
                          sections_present={k: int(v) for k, v in (tab > 0).sum(axis=1).items()},
                          min_cells_per_section={k: int(v) for k, v in tab.min(axis=1).items()},
                          labels_under_gate_in_any_section=[k for k, v in tab.min(axis=1).items() if v < min_cell_count],
                          per_section_table=tab.to_dict())
    return out

def granularity_report(path, audits, summaries, alpha):
    L = []
    L.append("# COZI cell-type networks: granularity report\n")
    L.append(f"Generated {time.strftime('%F %T')}. Significance = BH padj < {alpha}.\n")
    L.append("## Label columns available on the retained object\n")
    if not audits:
        L.append("Label audit not run (no `--obj` given); the column table below is therefore empty.\n")
    L.append("| column | categories | labels present in <20 sections | labels under the 10-cell gate somewhere |")
    L.append("|---|---|---|---|")
    for c, a in audits.items():
        few = [f"{k} ({v})" for k, v in a["sections_present"].items() if v < 20]
        L.append(f"| `{c}` | {a['n_categories']} | {', '.join(few) or 'none'} | "
                 f"{', '.join(a['labels_under_gate_in_any_section']) or 'none'} |")
    L.append("")
    for c, a in audits.items():
        L.append(f"### `{c}`\n")
        L.append("| label | cells | sections present | min cells / section |")
        L.append("|---|---|---|---|")
        for k in sorted(a["n_cells"], key=lambda k: -a["n_cells"][k]):
            L.append(f"| {k} | {a['n_cells'][k]:,} | {a['sections_present'][k]} | {a['min_cells_per_section'][k]:,} |")
        L.append("")
    L.append("## Network saturation per run\n")
    L.append("How much of the possible topology is already significant. When nearly every arc is "
             "significant, the cell-type level carries its information in the WEIGHTS, not in the "
             "topology, and topology-based methods (hubs, dominating sets, controllability) have "
             "nothing to choose between.\n")
    L.append("| run | nodes | arcs tested | arcs significant | attraction / avoidance | undirected attraction edges / pairs (sign-concordant) | sign-discordant pairs | one-way vs both-way | differential arcs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, s in summaries.items():
        L.append(f"| {k} | {s['n_nodes']} | {s['arcs_tested']}/{s['arcs_possible']} | "
                 f"{s['arcs_significant']} ({100 * s['arcs_significant_fraction']:.0f}%) | "
                 f"{s['arcs_attraction_sig']} / {s['arcs_avoidance_sig']} | "
                 f"{s['undirected_attraction_edges']} / {s['unordered_pairs']} | "
                 f"{s['undirected_sign_discordant_pairs']} | "
                 f"{s['one_way_significant_unordered']} / {s['both_way_significant_unordered']} | "
                 f"{s['differential_arcs']} |")
    L.append("")
    L.append("Per-group power (the Wilcoxon floor is 2/2^n for n sections; a group with all arcs at the floor "
             "and BH over the arc set cannot reach padj < alpha unless 2/2^n < alpha):\n")
    L.append("| run | group | sections per arc | worst achievable p | min padj | significant | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for k, s in summaries.items():
        for g, pw in s["group_power"].items():
            L.append(f"| {k} | {g} | {pw['n_sections_min']}-{pw['n_sections_max']} | {pw['worst_min_p_achievable']:.2e} | "
                     f"{pw['min_padj']:.3f} | {pw['n_significant']} | {'UNDERPOWERED, do not read as null' if pw['underpowered'] else 'testable'} |")
    L.append("")
    L.append("## Per-node significant degree (pooled)\n")
    for k, s in summaries.items():
        L.append(f"### {k}\n")
        L.append("| node | sig out-degree | sig in-degree | out-strength (sum z) | in-strength (sum z) |")
        L.append("|---|---|---|---|---|")
        for l in s["labels"]:
            L.append(f"| {l} | {s['node_sig_out_degree'][l]} | {s['node_sig_in_degree'][l]} | "
                     f"{s['node_out_strength_pooled'][l]:+.3f} | {s['node_in_strength_pooled'][l]:+.3f} |")
        L.append("")
    open(path, "w").write("\n".join(L))

# --------------------------------------------------------------------------- case-control from per-section rows
def case_control(per, alpha=0.05):
    """Mann-Whitney U on per-section z, ALS vs control, per ordered pair; BH over all testable pairs.
    Same procedure as cozi_als.py, used when sections are excluded and the shipped table no longer applies."""
    from scipy.stats import mannwhitneyu
    rows = []
    for (a, b), g in per.groupby(["index_cell_type", "neighbor_cell_type"]):
        za = g.loc[g.group == "ALS", "zscore"].dropna().values
        zc = g.loc[g.group == "control", "zscore"].dropna().values
        if len(za) < 3 or len(zc) < 3: continue
        try: u, p = mannwhitneyu(za, zc, alternative="two-sided")
        except Exception: continue
        rows.append(dict(index_cell_type=a, neighbor_cell_type=b, n_ALS=len(za), n_control=len(zc),
                         z_ALS=float(np.mean(za)), z_control=float(np.mean(zc)),
                         delta_z=float(np.mean(za) - np.mean(zc)),
                         cr_ALS=float(g.loc[g.group == "ALS", "cond_ratio"].mean()),
                         cr_control=float(g.loc[g.group == "control", "cond_ratio"].mean()),
                         U=float(u), p=float(p)))
    cc = pd.DataFrame(rows)
    if len(cc):
        cc["padj"] = bh(cc["p"].values); cc["significant"] = cc["padj"] < alpha
        cc = cc.sort_values("p")
    return cc

# --------------------------------------------------------------------------- main
def parse_args(a=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[],
                   help="scope=path/to/cozi_run_dir[:label_set]. Repeatable, e.g. whole=/.../20260915_022205_ncv3 "
                        "or v2=/.../cozi_rerun_cell_type_v2:celltype_cell_type_v2_wdr49")
    p.add_argument("--label-set", default="celltype_wdr49",
                   help="default cozi_persection_<label-set>.csv to use when a --run gives no :label_set")
    p.add_argument("--exclude-sections", default="",
                   help="comma-separated sample ids dropped from EVERY run before building (sensitivity analysis); "
                        "the case-control table is then recomputed from the remaining sections")
    p.add_argument("--outdir", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--obj", default=None, help="retained h5ad, for the obs label audit and --rerun")
    p.add_argument("--audit-cols", default="cell_type,cell_type_mn,cell_type_v2")
    p.add_argument("--rerun-celltype-col", default=None,
                   help="re-run COZI (whole sections) on this obs column and build its networks too")
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--n-permutations", type=int, default=300)
    p.add_argument("--min-cell-count", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(a)

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    summaries = {}
    manifest = dict(started=time.strftime("%F %T"), args=vars(args), runs={})
    excl = {s.strip() for s in args.exclude_sections.split(",") if s.strip()}
    for spec in args.run:
        scope, path = spec.split("=", 1)
        lset = args.label_set
        if ":" in path: path, lset = path.rsplit(":", 1)
        sec(f"BUILD  scope={scope}  label set={lset}  from {path}")
        T = os.path.join(path, "tables")
        per = pd.read_csv(os.path.join(T, f"cozi_persection_{lset}.csv"))
        ccp = os.path.join(T, f"cozi_case_control_{lset}.csv")
        cc = pd.read_csv(ccp) if os.path.exists(ccp) else None
        if excl:
            before = per["sample"].nunique()
            per = per[~per["sample"].isin(excl)].copy()
            per["group"] = np.where(per["status"].astype(str) == "control", "control", "ALS")
            cc = case_control(per, alpha=args.alpha)
            log(f"excluded {sorted(excl & set(per['sample'].unique()) | (excl - set(per['sample'].unique())))}: "
                f"{before} -> {per['sample'].nunique()} sections; case-control recomputed ({int(cc.significant.sum()) if len(cc) else 0} significant)")
        src_man = {}
        mp = os.path.join(path, "manifest.json")
        if os.path.exists(mp): src_man = json.load(open(mp))
        meta = dict(source_run=path, scope=scope, source_object=src_man.get("object", ""),
                    cozipy=src_man.get("cozipy", ""), n_cells_used=src_man.get("n_cells_used", ""),
                    excluded_sections=",".join(sorted(excl)) if excl else "")
        tag = f"{scope}_{lset}"
        summ, nodes, D, U = build_networks(per, cc, os.path.join(args.outdir, tag), tag, meta,
                                           alpha=args.alpha, draw=not args.no_figs)
        summaries[tag] = summ
        manifest["runs"][tag] = dict(source=path, label_set=lset, excluded_sections=sorted(excl), meta=meta)

    audits = {}
    if args.obj and os.path.exists(args.obj):
        sec("AUDIT  obs label columns")
        audits = obs_label_audit(args.obj, args.audit_cols.split(","), min_cell_count=args.min_cell_count)
        for c, a in audits.items():
            log(f"{c}: {a['n_categories']} categories; under-gate somewhere: {a['labels_under_gate_in_any_section']}")
        json.dump({c: {k: v for k, v in a.items() if k != "per_section_table"} for c, a in audits.items()},
                  open(os.path.join(args.outdir, "label_audit.json"), "w"), indent=1)
        for c, a in audits.items():
            pd.DataFrame(a["per_section_table"]).to_csv(os.path.join(args.outdir, f"label_audit_{c}_per_section.csv"))

    if args.rerun_celltype_col:
        sec(f"RERUN  COZI on {args.rerun_celltype_col} (whole sections)")
        rdir = os.path.join(args.outdir, f"cozi_rerun_{args.rerun_celltype_col}")
        per, cc, name, man = rerun_cozi(args.obj, args.rerun_celltype_col, rdir,
                                        n_neighbors=args.n_neighbors, n_perm=args.n_permutations,
                                        min_cell_count=args.min_cell_count, seed=args.seed)
        meta = dict(source_run=rdir, scope="whole", source_object=args.obj, cozipy=man["cozipy"],
                    n_cells_used=man["n_cells_used"])
        tag = f"whole_{name}"
        summ, nodes, D, U = build_networks(per, cc, os.path.join(args.outdir, tag), tag, meta,
                                           alpha=args.alpha, draw=not args.no_figs)
        summaries[tag] = summ
        manifest["runs"][tag] = dict(source=rdir, label_set=name, rerun=True, meta=meta)

    granularity_report(os.path.join(args.outdir, "GRANULARITY_REPORT.md"), audits, summaries, args.alpha)
    json.dump(dict(summaries=summaries,
                   audits={c: {k: v for k, v in a.items() if k != "per_section_table"} for c, a in audits.items()}),
              open(os.path.join(args.outdir, "granularity.json"), "w"), indent=1, default=str)
    manifest["finished"] = time.strftime("%F %T")
    json.dump(manifest, open(os.path.join(args.outdir, "manifest.json"), "w"), indent=1, default=str)
    log(f"wrote {args.outdir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
