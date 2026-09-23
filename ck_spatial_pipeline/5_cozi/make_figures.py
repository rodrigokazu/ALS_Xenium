#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/make_figures.py
#
# Figures 1 to 10 for niche DE plus COZI, with constrained layout everywhere. FIG9 is the
# WDR49+ and MN case against control plot and FIG10 the COZI niche plot. It feeds the Niche
# DE and COZI Explorer.
# ========================================================================================

"""Nature-style despined figures: niche DE v2 + COZI case x control.

Layout rules, applied everywhere so text can never sit on the data:
  * every figure uses matplotlib constrained layout, which reserves space for
    tick labels, axis labels and outside legends instead of letting them collide;
  * at most ONE figure-level legend per figure, always loc="outside lower center";
  * anything that can be direct-labelled is direct-labelled rather than legended;
  * despined: only the axes that carry a scale keep a spine.

Emits vector PDF, 600 dpi PNG and SVG (svg.fonttype=none, so text stays text and the
figure zooms losslessly when embedded in an artifact).
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": False,
    "figure.dpi": 110, "savefig.bbox": None, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.h_pad": 0.05,
    "figure.constrained_layout.wspace": 0.09,
})
INK="#1a1a1a"; MID="#555555"; MUT="#8a8a8a"; LINE="#cfcfcf"
BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; RED="#D55E00"
PURPLE="#7B52AB"; PINK="#CC79A7"; TEAL="#56B4E9"; YELLOW="#F0E442"; GREY="#9a9a9a"
DIV = LinearSegmentedColormap.from_list("div", ["#2166AC","#87B0D6","#F5F5F5","#E8A07A","#B2182B"])

def despine(ax, keep=("left","bottom")):
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(s in keep)

def save(fig, outdir, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600}), ("svg", {})):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), **kw)
    plt.close(fig); print("   ", name)

def panel(ax, letter):
    ax.set_title(letter, loc="left", fontsize=9, fontweight="bold", color=INK, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de", required=True); ap.add_argument("--cozi", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    T = os.path.join(a.de, "tables")
    def rd(p, **kw):
        f = os.path.join(T, p)
        return pd.read_csv(f, **kw) if os.path.exists(f) else None
    print("figures ->", a.outdir)

    # ============================================= FIG1 WM/GM annotation
    ann = rd("A_wm_gm_annotation.csv", index_col=0)
    comp = rd("A_niche_composition.csv", index_col=0)
    if ann is not None and comp is not None:
        ann.index = ann.index.astype(str); comp.index = comp.index.astype(str)
        order = list(ann.sort_values("frac_oligo", ascending=False).index)
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                gridspec_kw={"width_ratios": [1.25, 1]})
        pal = {"Oligodendrocytes": BLUE, "OPCs": TEAL, "Astrocytes": PURPLE,
               "Excitatory neurons": RED, "Inhibitory neurons": ORANGE,
               "Other neurons": PINK, "Motor neurons": "#7F0000",
               "Microglia": GREEN, "Macrophages": YELLOW, "Endothelial": GREY}
        ax = axs[0]; bot = np.zeros(len(order))
        for c in comp.columns:
            v = comp.loc[order, c].values * 100
            ax.bar(range(len(order)), v, bottom=bot, width=.72,
                   color=pal.get(c, "#bbbbbb"), label=c, linewidth=0)
            bot += v
        ax.set_xticks(range(len(order))); ax.set_xticklabels([f"n{n}" for n in order])
        ax.set_ylabel("cells (%)"); ax.set_ylim(0, 100); ax.set_xlim(-.62, len(order)-.38)
        ax.set_xlabel("niche"); despine(ax); panel(ax, "a")
        # ONE figure-level legend, outside, so it cannot touch either panel
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=5.9,
                   handlelength=.85, handleheight=.85, columnspacing=.9, labelspacing=.25)
        # panel b: direct labelling, no legend
        ax = axs[1]
        HL = {"clear white matter": BLUE, "clear grey matter": RED}
        for n in ann.index:
            r = ann.loc[n]
            c = HL.get(r["call"], "#c8c8c8")
            ax.scatter(r.WM_score, r.GM_score, s=26 if r["call"] in HL else 16,
                       color=c, edgecolor="white", linewidth=.5,
                       zorder=4 if r["call"] in HL else 3)
        ax.axhline(0, color=LINE, lw=.6, zorder=1); ax.axvline(0, color=LINE, lw=.6, zorder=1)
        # deterministic dodge for the crowded low-score cluster
        dodge = {}
        for n in ann.index:
            r = ann.loc[n]
            key = (round(r.WM_score, 1), round(r.GM_score, 1))
            i = dodge.get(key, 0); dodge[key] = i + 1
            off = [(6, -2), (6, 6), (6, -10), (-7, 6), (-7, -10), (0, 8)][i % 6]
            ax.annotate(f"n{n}", (r.WM_score, r.GM_score), fontsize=6.2,
                        ha="left" if off[0] >= 0 else "right",
                        textcoords="offset points", xytext=off, color=INK, zorder=5)
        for n, txt, dx, dy, ha in (
                (ann.index[ann["call"] == "clear white matter"], "clear\nwhite matter", -10, -14, "right"),
                (ann.index[ann["call"] == "clear grey matter"], "clear\ngrey matter", -12, -4, "right")):
            if len(n):
                r = ann.loc[n[0]]
                ax.annotate(txt, (r.WM_score, r.GM_score), fontsize=6,
                            textcoords="offset points", xytext=(dx, dy), ha=ha,
                            color=HL[r["call"]], fontweight="bold", zorder=5)
        ax.set_xlabel("white-matter panel score (mean $z$)")
        ax.set_ylabel("grey-matter panel score (mean $z$)")
        xl = ax.get_xlim(); yl = ax.get_ylim()
        ax.set_xlim(xl[0]-.18, xl[1]+.22); ax.set_ylim(yl[0]-.12, yl[1]+.30)
        despine(ax); panel(ax, "b")
        save(fig, a.outdir, "FIG1_wm_gm_niche_annotation")

    # ============================================= FIG2 design
    pbg = rd("pseudobulk_niche_by_group.csv", index_col=0)
    pbm = rd("pseudobulk_metadata.csv", index_col=0)
    if pbg is not None:
        pbg.index = pbg.index.astype(str)
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
        idx = list(pbg.index); x = np.arange(len(idx)); w = .38
        ax = axs[0]
        for k, (g, col) in enumerate((("control", GREY), ("ALS", RED))):
            if g in pbg.columns:
                ax.bar(x + (k-.5)*w, pbg[g].values, w, color=col, label=g, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f"n{i}" for i in idx])
        ax.set_xlabel("niche"); ax.set_ylabel("pseudobulks (donors)")
        despine(ax); panel(ax, "a")
        h, l = ax.get_legend_handles_labels()
        ax = axs[1]
        if pbm is not None:
            cc = [c for c in pbm.columns if c.startswith("psbulk") and "cell" in c]
            if cc:
                col = cc[0]
                for k, (g, c2) in enumerate((("control", GREY), ("ALS", RED))):
                    sub = pbm[pbm["group"] == g] if "group" in pbm.columns else pbm
                    for i, n in enumerate(idx):
                        v = sub[sub["niche"].astype(str) == n][col].values
                        if not len(v): continue
                        jit = (np.random.default_rng(i*10+k).random(len(v))-.5)*.20
                        ax.scatter(np.full(len(v), i+(k-.5)*.28)+jit, v, s=5.5,
                                   color=c2, alpha=.85, linewidth=0, zorder=3)
                ax.set_yscale("log")
                ax.set_xticks(range(len(idx))); ax.set_xticklabels([f"n{i}" for i in idx])
                ax.set_xlabel("niche")
                ax.set_ylabel("WDR49+ astrocytes per pseudobulk")
                ax.axhline(20, color=INK, ls=":", lw=.7)
                ax.annotate("QC floor (20 cells)", (0.99, 20), xycoords=("axes fraction", "data"),
                            fontsize=5.6, color=INK, ha="right", va="bottom")
                despine(ax)
        panel(ax, "b")
        fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=6.2, handlelength=.9)
        save(fig, a.outdir, "FIG2_pseudobulk_design")

    # ============================================= FIG3/4 DEG counts
    summ = rd("DE_summary.csv")
    if summ is not None and len(summ):
        summ["kind"] = np.where(summ.contrast.str.startswith("caseControl"),
                                "case vs control", "niche vs niche")
        for kind, tag in (("case vs control", "FIG3_case_control_DEG_counts"),
                          ("niche vs niche", "FIG4_niche_vs_niche_DEG_counts")):
            s = summ[summ.kind == kind].copy()
            if not len(s): continue
            s["lab"] = (s.contrast.str.replace("caseControl_", "", regex=False)
                        .str.replace("nicheXniche_", "n", regex=False)
                        .str.replace("_vs_", " vs n", regex=False)
                        .str.replace("vsControl", " vs control", regex=False)
                        .str.replace("_", " ", regex=False))
            s = s.sort_values("significant")
            fig, ax = plt.subplots(figsize=(4.6, max(1.9, .245*len(s)+1.0)))
            y = np.arange(len(s))
            ax.barh(y, s.significant, .62, color="#dcdcdc", linewidth=0, label="significant")
            ax.barh(y, s.seg_pass, .62, color=BLUE, linewidth=0,
                    label="survives segmentation filters")
            mx = max(1, s.significant.max())
            for i, r in enumerate(s.itertuples()):
                ax.text(max(r.significant, 0) + mx*.02, i,
                        f"{int(r.seg_pass)} / {int(r.significant)}", va="center",
                        fontsize=5.8, color=MID)
            ax.set_yticks(y); ax.set_yticklabels(s.lab, fontsize=6)
            ax.set_xlabel("differentially expressed genes")
            ax.set_xlim(0, mx*1.20)
            despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=6.2, handlelength=.9)
            save(fig, a.outdir, tag)

    # ============================================= FIG5 filter action
    f1 = rd("F1_global_detection_filter.csv")
    if f1 is not None and summ is not None:
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
        ax = axs[0]
        bars = {"F1\nglobal": int(f1.f1_fail.sum())}
        for n in ("0", "1", "2", "4", "5"):
            d = rd(f"F2_flags_niche{n}.csv")
            if d is not None:
                k = [c for c in d.columns if c.startswith("f2_fail")][0]
                bars[f"F2\nn{n}"] = int(d[k].sum())
        ax.bar(range(len(bars)), list(bars.values()), .66,
               color=[RED] + [BLUE]*(len(bars)-1), linewidth=0)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels(list(bars), fontsize=5.8)
        ax.set_ylabel(f"panel genes flagged (of {len(f1)})")
        for i, v in enumerate(bars.values()):
            ax.text(i, v + len(f1)*.014, str(v), ha="center", fontsize=5.8, color=MID)
        ax.set_ylim(0, max(bars.values())*1.16)
        despine(ax); panel(ax, "a")
        ax = axs[1]
        s = summ[summ.significant > 0].copy()
        s["pct"] = 100*s.removed/s.significant
        s = s.sort_values("pct")
        col = np.where(s.contrast.str.startswith("caseControl"), ORANGE, BLUE)
        ax.hlines(range(len(s)), 0, s.pct, color=LINE, lw=.9, zorder=1)
        ax.scatter(s.pct, range(len(s)), s=14, color=col, zorder=3, linewidth=0)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.contrast.str.replace("caseControl_", "", regex=False)
                           .str.replace("nicheXniche_", "n", regex=False)
                           .str.replace("_vs_", " vs n", regex=False)
                           .str.replace("vsControl", " vs ctrl", regex=False)
                           .str.replace("_", " ", regex=False), fontsize=5.3)
        ax.set_xlabel("significant DEGs removed by filters (%)"); ax.set_xlim(-2, 104)
        despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
        panel(ax, "b")
        hs = [plt.Line2D([], [], marker="o", ls="", ms=3.2, color=ORANGE, label="case vs control"),
              plt.Line2D([], [], marker="o", ls="", ms=3.2, color=BLUE, label="niche vs niche")]
        fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=6.2,
                   handletextpad=.35)
        save(fig, a.outdir, "FIG5_segmentation_filter_action")

    # ============================================= FIG6 top case/control DEGs
    sp = rd("DE_significant_seg_pass.csv")
    if sp is not None and len(sp):
        cc = sp[sp.contrast.str.startswith("caseControl")].copy()
        if len(cc):
            top = cc.sort_values("padj").head(22).sort_values("log2FoldChange")
            fig, ax = plt.subplots(figsize=(5.0, max(2.0, .235*len(top)+1.0)))
            y = np.arange(len(top))
            ax.barh(y, top.log2FoldChange, .64,
                    color=np.where(top.log2FoldChange > 0, RED, BLUE), linewidth=0)
            ax.axvline(0, color=INK, lw=.6)
            labs = [f"{g}   {c.replace('caseControl_','').replace('vsControl','').replace('_',' ')}"
                    for g, c in zip(top.gene, top.contrast)]
            ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.7)
            ax.set_xlabel("log$_2$ fold change")
            xl = ax.get_xlim(); ax.set_xlim(xl[0]-.25, xl[1]+.25)
            ax.annotate("higher in controls", (0.02, 1.012), xycoords="axes fraction",
                        fontsize=5.8, color=BLUE, ha="left", va="bottom")
            ax.annotate("higher in ALS", (0.98, 1.012), xycoords="axes fraction",
                        fontsize=5.8, color=RED, ha="right", va="bottom")
            despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
            save(fig, a.outdir, "FIG6_case_control_top_DEGs")

    # ============================================= FIG7 marker heatmap
    z = rd("interniche_zscores.csv", index_col=0)
    if z is not None and sp is not None and len(sp):
        nn = sp[sp.contrast.str.startswith("nicheXniche")]
        genes = (nn.groupby("gene").size().sort_values(ascending=False).head(32).index.tolist()
                 if len(nn) else [])
        genes = [g for g in genes if g in z.index]
        if len(genes) >= 4:
            sub = z.loc[genes]
            sub = sub.iloc[np.lexsort((sub.values.max(axis=1), sub.values.argmax(axis=1)))]
            fig, ax = plt.subplots(figsize=(0.40*len(sub.columns)+2.3, 0.155*len(sub)+1.0))
            v = float(np.nanmax(np.abs(sub.values)))
            im = ax.imshow(sub.values, cmap=DIV, aspect="auto",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
            ax.set_xticks(range(len(sub.columns)))
            ax.set_xticklabels([f"n{c}" for c in sub.columns])
            ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=5.7)
            ax.set_xlabel("niche")
            for s2 in ax.spines.values(): s2.set_visible(False)
            ax.tick_params(length=0)
            cb = fig.colorbar(im, ax=ax, pad=.03, fraction=.055, aspect=14)
            cb.set_label("expression $z$ across niches", fontsize=6)
            cb.ax.tick_params(labelsize=5.7, length=1.6); cb.outline.set_visible(False)
            save(fig, a.outdir, "FIG7_interniche_marker_heatmap")

    # ============================================= FIG8 cilia flag
    if sp is not None and len(sp):
        cil = sp[sp.gene.isin(["ZBBX", "CCDC146", "CRYM"])].copy()
        if len(cil):
            fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4),
                                    gridspec_kw={"width_ratios": [1.3, 1]})
            ax = axs[0]
            gs = sorted(cil.gene.unique())
            for i, g in enumerate(gs):
                sub = cil[cil.gene == g]
                ax.scatter(sub.log2FoldChange, [i]*len(sub), s=17, color=BLUE,
                           alpha=.85, linewidth=0, zorder=3)
                ax.annotate(f"{len(sub)} contrasts", (sub.log2FoldChange.min(), i),
                            textcoords="offset points", xytext=(-6, 0), ha="right",
                            va="center", fontsize=5.6, color=MUT)
            ax.axvline(0, color=INK, lw=.6)
            ax.set_yticks(range(len(gs))); ax.set_yticklabels(gs, fontsize=6.6)
            ax.set_ylim(-.7, len(gs)-.3)
            ax.set_xlabel("log$_2$ fold change per contrast")
            xl = ax.get_xlim(); ax.set_xlim(xl[0]-1.5, xl[1]+.4)
            despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
            panel(ax, "a")
            ax = axs[1]
            u = cil.drop_duplicates("gene").set_index("gene").loc[gs].reset_index()
            xx = np.arange(len(u)); w = .34
            ax.bar(xx-w/2, u.det_focal, w, color=PURPLE, linewidth=0,
                   label="WDR49+ astrocytes")
            ax.bar(xx+w/2, u.det_nonastro, w, color=GREY, linewidth=0, label="non-astrocytes")
            ax.set_xticks(xx); ax.set_xticklabels(u.gene, fontsize=6.6)
            ax.set_ylabel("detection rate")
            ax.set_ylim(0, max(u.det_focal.max(), u.det_nonastro.max())*1.25)
            despine(ax); panel(ax, "b")
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=6.2, handlelength=.9)
            save(fig, a.outdir, "FIG8_niche5_cilia_flag")

    # ============================================= FIG9/10 COZI
    if a.cozi:
        CT = os.path.join(a.cozi, "tables")
        def rc(p):
            f = os.path.join(CT, p)
            return pd.read_csv(f) if os.path.exists(f) else None
        import glob as _glob
        for fp in sorted(_glob.glob(os.path.join(CT, "cozi_case_control_*_focus.csv"))):
            fname = os.path.basename(fp).replace("cozi_case_control_", "").replace("_focus.csv", "")
            foc = pd.read_csv(fp)
            if not len(foc): continue
            d = foc.reindex(foc.delta_z.abs().sort_values(ascending=False).index).head(19)
            d = d.sort_values("delta_z")
            fig, ax = plt.subplots(figsize=(5.4, max(2.0, .235*len(d)+1.15)))
            y = np.arange(len(d))
            ax.barh(y, d.delta_z, .64,
                    color=np.where(d.delta_z > 0, RED, BLUE), linewidth=0)
            ax.axvline(0, color=INK, lw=.6)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{r.index_cell_type} $\\rightarrow$ {r.neighbor_cell_type}"
                                for r in d.itertuples()], fontsize=5.7)
            ax.set_xlabel("$\\Delta z$ (ALS $-$ control)")
            xl = ax.get_xlim(); pad = (xl[1]-xl[0])*.06
            ax.set_xlim(xl[0]-pad, xl[1]+pad)
            ax.annotate("lower in ALS", (0.02, 1.012), xycoords="axes fraction",
                        fontsize=5.8, color=BLUE, ha="left", va="bottom")
            ax.annotate("higher in ALS", (0.98, 1.012), xycoords="axes fraction",
                        fontsize=5.8, color=RED, ha="right", va="bottom")
            nsig = int(d.significant.sum()) if "significant" in d.columns else 0
            tot = int(foc.significant.sum()) if "significant" in foc.columns else 0
            msg = (f"{tot} of {len(foc)} ordered pairs significant at BH < 0.05"
                   if tot else f"no pair significant at BH < 0.05 (0 of {len(foc)})")
            ax.annotate(msg, (0.5, -0.155), xycoords="axes fraction", fontsize=6.2,
                        color=INK if tot else MID, ha="center", va="top")
            for i, r in enumerate(d.itertuples()):
                if getattr(r, "significant", False):
                    ax.annotate("*", (r.delta_z, i), textcoords="offset points",
                                xytext=(4 if r.delta_z > 0 else -4, -2),
                                ha="left" if r.delta_z > 0 else "right",
                                fontsize=9, color=INK)
            despine(ax, keep=("bottom",)); ax.tick_params(axis="y", length=0)
            save(fig, a.outdir, f"FIG9_cozi_{fname}_case_control")
        nic = rc("cozi_case_control_niche.csv")
        if nic is not None and len(nic):
            piv = nic.pivot_table(index="index_cell_type", columns="neighbor_cell_type",
                                  values="delta_z")
            sig = nic[nic.significant] if "significant" in nic.columns else nic.iloc[:0]
            fig, ax = plt.subplots(figsize=(0.40*len(piv.columns)+2.4, 0.36*len(piv)+1.2))
            v = float(np.nanmax(np.abs(piv.values)))
            im = ax.imshow(piv.values, cmap=DIV, aspect="auto",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([c.replace("niche", "n") for c in piv.columns])
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels([c.replace("niche", "n") for c in piv.index])
            ax.set_xlabel("neighbour niche"); ax.set_ylabel("index niche")
            for r in sig.itertuples():
                if r.index_cell_type in piv.index and r.neighbor_cell_type in piv.columns:
                    ax.text(list(piv.columns).index(r.neighbor_cell_type),
                            list(piv.index).index(r.index_cell_type), "*",
                            ha="center", va="center", fontsize=8, color=INK)
            for s2 in ax.spines.values(): s2.set_visible(False)
            ax.tick_params(length=0)
            cb = fig.colorbar(im, ax=ax, pad=.03, fraction=.05, aspect=14)
            cb.set_label("$\\Delta z$ (ALS $-$ control)", fontsize=6)
            cb.ax.tick_params(labelsize=5.7, length=1.6); cb.outline.set_visible(False)
            ax.annotate("* BH < 0.05", (0.0, -0.20), xycoords="axes fraction",
                        fontsize=6, color=INK, ha="left", va="top")
            save(fig, a.outdir, "FIG10_cozi_niche_case_control")
    print("done")

if __name__ == "__main__":
    sys.exit(main())
