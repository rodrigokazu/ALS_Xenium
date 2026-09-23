#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/niche_de_v2.py
#
# Niche DE on the integrated run. pyDESeq2 on raw-count pseudobulks per niche and donor,
# FB_SPI filters F1 (2x) and F2 (4x), alpha 0.05 and |LFC| >= 1. It shares the object, the
# cell_type_mn labels and the WDR49 >= 1 focal definition with cozi_als.py, which is why it
# ships in this folder. Runs on the lianaplus env.
# ========================================================================================

"""
v2 niche analysis on the WHOLE-SECTION NicheCompass run (integrated_whole), whose niches
are comparable across samples. Three things:

  A. WM / GM niche annotation, driven by cell-type composition AND marker-panel expression,
     so the call is auditable rather than asserted.
  B. Niche x niche pseudobulk DE for WDR49+ astrocytes (as v1).
  C. CASE x CONTROL pseudobulk DE for WDR49+ astrocytes -- the comparison of primary interest.
     Per niche (ALS vs control, c9 vs control, sporadic vs control) and pooled across niches
     with niche as a covariate, which is the highest-powered version.

Every DE table carries the FB_SPI segmentation-artefact flags (F1 global, F2 per niche).

Deliberate deviation from FB_SPI wording, as in v1: pseudobulks are RAW counts summed and
pyDESeq2 fits its own size factors, because DESeq2 requires integer counts. The
size-factor-normalised values feed only the inter-niche Z matrix.

READ-ONLY on the input object.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

OBJ = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC/"
       "runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad")

# marker panels, restricted at runtime to what is actually on the 480-plex
WM_PANEL = ["MOG","MBP","MAG","MOBP","CLDN11","OPALIN","ERMN","SOX10","OLIG1","OLIG2",
            "ST18","CTNNA3","MYRF"]
GM_PANEL = ["SLC17A7","GAD1","GAD2","RBFOX3","VIP","SST","RORB","GRIK3","BRINP3","GPC5"]
CILIA_PANEL = ["ZBBX","CCDC146","CRYM"]
ASTRO_PANEL = ["AQP4","GJA1","SERPINA3","CHI3L1","THSD4","SPON1","SLC1A2"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def sec(m): print("\n"+"="*78+f"\n{m}\n"+"="*78, flush=True)


def parse_args(a=None):
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default=OBJ)
    p.add_argument("--outdir", required=True)
    p.add_argument("--niche-col", default="latent_leiden_0.3")
    p.add_argument("--celltype-col", default="cell_type_mn",
                   help="ALWAYS cell_type_mn. Its 'Motor neurons' category == is_MN_v2 "
                        "(844 cells); the coarse cell_type 'Motor neurons' class "
                        "(101,660) is DEPRECATED and must never be used.")
    p.add_argument("--sample-col", default="sample")
    p.add_argument("--focal-mode", default="wdr49_astro",
                   choices=["wdr49_astro", "mn_canonical"],
                   help="wdr49_astro = cell_type=='Astrocytes' & WDR49>=n; "
                        "mn_canonical = obs['is_MN_v2'] is True")
    p.add_argument("--focal-type", default="Astrocytes")
    p.add_argument("--focal-label", default=None, help="name used in outputs")
    p.add_argument("--gene", default="WDR49")
    p.add_argument("--gene-min", type=int, default=1)
    p.add_argument("--niches", default="0,1,2,4,5")
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--min-counts", type=int, default=1000)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--min-total-count", type=int, default=10)
    p.add_argument("--large-n", type=int, default=3)
    p.add_argument("--min-prop", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lfc", type=float, default=1.0)
    p.add_argument("--f1-fold", type=float, default=2.0)
    p.add_argument("--f2-fold", type=float, default=4.0)
    p.add_argument("--min-per-group", type=int, default=3)
    p.add_argument("--chunk", type=int, default=100000)
    return p.parse_args(a)


def stream_read(path, cols, gene, chunk, nbcols=()):
    import h5py
    f = h5py.File(path, "r")
    obs = {}
    for k in cols:
        g = f["obs"][k]
        if isinstance(g, h5py.Group):
            cats = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
            obs[k] = pd.Categorical.from_codes(g["codes"][:], categories=cats).astype(str)
        else:
            obs[k] = g[:]
    # anndata nullable-boolean: values + mask, mask True == NA
    for k in nbcols:
        g = f["obs"][k]
        obs[k] = g["values"][:] & ~g["mask"][:]
        obs[k + "_isna"] = g["mask"][:]
    var = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][:]]
    L = f["layers"]["counts"]; indptr = L["indptr"][:]
    n_obs, n_var = len(indptr)-1, len(var)
    gj = var.index(gene)
    gc = np.zeros(n_obs, dtype=np.int32)
    Xd = np.zeros((n_obs, n_var), dtype=np.float32)
    idx, dat = L["indices"], L["data"]
    for s in range(0, n_obs, chunk):
        e = min(s+chunk, n_obs); a, b = indptr[s], indptr[e]
        ii = idx[a:b].astype(np.int64); dd = dat[a:b]
        rows = np.repeat(np.arange(s, e), np.diff(indptr[s:e+1]))
        Xd[rows, ii] = dd
        hit = ii == gj
        if hit.any(): np.add.at(gc, rows[hit], dd[hit].astype(np.int32))
    f.close()
    log(f"read {n_obs:,} x {n_var}")
    return pd.DataFrame(obs), np.array(var), Xd, gc


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    T = os.path.join(args.outdir, "tables"); os.makedirs(T, exist_ok=True)
    niches = [x.strip() for x in args.niches.split(",")]

    import anndata as ad, scipy.sparse as sp, decoupler as dc, scanpy as scp
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    man = {"object": args.obj, "params": vars(args), "started": time.strftime("%F %T")}

    sec("LOAD")
    nbc = ("is_MN_v2",) if args.focal_mode == "mn_canonical" else ()
    obs, var, Xd, gc = stream_read(args.obj, [args.niche_col, args.celltype_col,
                                              args.sample_col, "status"], args.gene,
                                   args.chunk, nbcols=nbc)
    obs["niche"] = obs[args.niche_col].astype(str)
    ct = obs[args.celltype_col].values.astype(str)
    nl = obs["niche"].values.astype(str)
    if args.focal_mode == "mn_canonical":
        # canonical curated motor-neuron call; the LINEAGE comparator for the
        # segmentation filters is the coarse 'Motor neurons' class
        focal = obs["is_MN_v2"].values.astype(bool)
        # cell_type_mn 'Motor neurons' IS is_MN_v2, so the lineage class and the focal
        # set coincide and the F1/F2 comparator is simply every non-canonical-MN cell.
        # The deprecated coarse cell_type 'Motor neurons' class is never used.
        is_ft = ct == "Motor neurons"
        assert is_ft.sum() == focal.sum(), (
            f"celltype-col must be cell_type_mn: got {is_ft.sum()} 'Motor neurons' "
            f"vs {focal.sum()} is_MN_v2. The coarse cell_type class is DEPRECATED.")
        FOCAL_NAME = args.focal_label or "MN_canonical"
    else:
        is_ft = ct == args.focal_type
        focal = is_ft & (gc >= args.gene_min)
        FOCAL_NAME = args.focal_label or f"{args.focal_type}_{args.gene}pos"
    man["focal_mode"] = args.focal_mode
    man["focal_name"] = FOCAL_NAME
    man["n_focal"] = int(focal.sum())
    # ALS = c9 + sporadic
    obs["group"] = np.where(obs["status"].values == "control", "control", "ALS")
    log(f"focal set [{FOCAL_NAME}] = {focal.sum():,} cells; lineage comparator class = {is_ft.sum():,} cells")
    dstat = obs.groupby(args.sample_col)["status"].first()
    log("donors by status: " + str(dict(dstat.value_counts())))
    man["donors_by_status"] = {k: int(v) for k, v in dstat.value_counts().items()}

    # ============================================================ A. WM/GM annotation
    sec("A. WM / GM NICHE ANNOTATION")
    panels = {"WM": [g for g in WM_PANEL if g in set(var)],
              "GM": [g for g in GM_PANEL if g in set(var)],
              "cilia": [g for g in CILIA_PANEL if g in set(var)],
              "astro": [g for g in ASTRO_PANEL if g in set(var)]}
    for k, v in panels.items(): log(f"panel {k}: {len(v)} genes on the 480-plex -> {v}")
    man["panels"] = panels

    # composition
    comp = pd.crosstab(nl, ct, normalize="index")
    comp.to_csv(os.path.join(T, "A_niche_composition.csv"))
    oligo = comp.get("Oligodendrocytes", pd.Series(0, index=comp.index))
    opc = comp.get("OPCs", pd.Series(0, index=comp.index))
    neuro_cols = [c for c in comp.columns if "neuron" in c.lower()]
    neuro = comp[neuro_cols].sum(axis=1)
    astroF = comp.get("Astrocytes", pd.Series(0, index=comp.index))

    # niche-level mean expression (ALL cells in the niche), then z across niches
    gi = {g: i for i, g in enumerate(var)}
    rows = {}
    for n in comp.index:
        m = nl == n
        if m.sum() == 0: continue
        rows[n] = Xd[m].mean(axis=0)
    nmean = pd.DataFrame(rows, index=var).T
    nmean.to_csv(os.path.join(T, "A_niche_mean_expression.csv"))
    z = (nmean - nmean.mean(axis=0)) / nmean.std(axis=0, ddof=0).replace(0, np.nan)
    scores = pd.DataFrame({k: z[v].mean(axis=1) for k, v in panels.items() if v})

    ev = pd.DataFrame({
        "n_cells": pd.Series({n: int((nl == n).sum()) for n in comp.index}),
        "frac_oligo": oligo, "frac_OPC": opc, "frac_neurons": neuro, "frac_astro": astroF,
        "WM_score": scores["WM"], "GM_score": scores["GM"],
        "cilia_score": scores.get("cilia"), "astro_score": scores.get("astro"),
    })

    def call(r):
        if r.frac_oligo >= 0.45 and r.frac_neurons <= 0.10 and r.WM_score > 0.4:
            return "clear white matter"
        if r.frac_oligo >= 0.35 and r.frac_neurons <= 0.22 and r.WM_score > 0:
            return "white matter (leaning)"
        if r.frac_neurons >= 0.28 and r.frac_oligo <= 0.25 and r.GM_score > 0.2:
            return "clear grey matter"
        if r.frac_neurons >= 0.20 and r.frac_oligo <= 0.20:
            return "grey matter (leaning)"
        return "mixed / other"
    ev["call"] = ev.apply(call, axis=1)
    ev["in_DE"] = [n in niches for n in ev.index]
    ev = ev.round(4)
    ev.to_csv(os.path.join(T, "A_wm_gm_annotation.csv"))
    print(ev.to_string())
    man["annotation"] = ev.reset_index().rename(columns={"index": "niche"}).to_dict("records")

    # ================================================== F1 / F2 segmentation filters
    sec("SEGMENTATION FILTERS")
    det_f = (Xd[focal] > 0).mean(axis=0)
    det_n = (Xd[~is_ft] > 0).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(det_f > 0, det_n / det_f, np.inf)
    f1 = pd.DataFrame({"gene": var, "det_focal": det_f, "det_nonastro": det_n,
                       "ratio_nonastro_over_focal": ratio, "f1_fail": ratio > args.f1_fold})
    f1.to_csv(os.path.join(T, "F1_global_detection_filter.csv"), index=False)
    log(f"F1 flags {int(f1.f1_fail.sum())}/{len(var)}")

    grp = np.where(focal, "FOCAL", ct)
    keyed = np.array([f"{a}||{b}" for a, b in zip(grp, nl)])
    dg, mg = {}, {}
    for g in pd.unique(keyed):
        m = keyed == g
        dg[g] = (Xd[m] > 0).mean(axis=0); mg[g] = Xd[m].mean(axis=0)
    det_g = pd.DataFrame(dg, index=var).T; mean_g = pd.DataFrame(mg, index=var).T
    det_g.to_csv(os.path.join(T, "F2_group_detection.csv"))
    mean_g.to_csv(os.path.join(T, "F2_group_mean.csv"))

    def f2_for(n):
        fk = f"FOCAL||{n}"
        if fk not in det_g.index: return None
        fd, fm = det_g.loc[fk], mean_g.loc[fk]
        oth = [i for i in det_g.index if i.endswith(f"||{n}") and not i.startswith("FOCAL||")]
        if not oth: return None
        od, om = det_g.loc[oth], mean_g.loc[oth]
        top = om.idxmax(axis=0)
        td = pd.Series([od.loc[top[g], g] for g in om.columns], index=om.columns)
        tm = pd.Series([om.loc[top[g], g] for g in om.columns], index=om.columns)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = np.where(fd.values > 0, td.values / fd.values, np.inf)
            rm = np.where(fm.values > 0, tm.values / fm.values, np.inf)
        return pd.DataFrame({"gene": om.columns,
                             f"top_nonastro_{n}": [t.split("||")[0] for t in top.values],
                             f"det_ratio_{n}": rd, f"mean_ratio_{n}": rm,
                             f"f2_fail_{n}": (rd > args.f2_fold) | (rm > args.f2_fold)}
                            ).set_index("gene")
    f2 = {n: f2_for(n) for n in niches}
    for n, d in f2.items():
        if d is not None:
            d.to_csv(os.path.join(T, f"F2_flags_niche{n}.csv"))
            log(f"  F2 niche {n}: {int(d[f'f2_fail_{n}'].sum())}/{len(d)}")
    del Xd

    # ==================================================================== pseudobulk
    sec("PSEUDOBULK per (niche, donor)")
    A = ad.read_h5ad(args.obj)
    A = A[focal].copy()
    A.obs["niche"] = A.obs[args.niche_col].astype(str)
    A = A[A.obs["niche"].isin(niches)].copy()
    A.obs["group"] = np.where(A.obs["status"].values == "control", "control", "ALS")
    A.X = A.layers["counts"].copy()
    log(f"focal cells in tested niches: {A.n_obs:,}")
    pb = dc.pp.pseudobulk(A, sample_col=args.sample_col, groups_col="niche",
                          mode="sum", skip_checks=True)
    dc.pp.filter_samples(pb, min_cells=args.min_cells, min_counts=args.min_counts, inplace=True)
    pb.obs["group"] = np.where(pb.obs["status"].values == "control", "control", "ALS")
    log(f"pseudobulks: {pb.n_obs}")
    xt = pd.crosstab(pb.obs["niche"], pb.obs["group"])
    print(xt.to_string())
    xt.to_csv(os.path.join(T, "pseudobulk_niche_by_group.csv"))
    meta_cols = ["niche", args.sample_col, "status", "group"] + \
                [c for c in pb.obs.columns if c.startswith("psbulk")]
    pb.obs[meta_cols].to_csv(os.path.join(T, "pseudobulk_metadata.csv"))
    man["pseudobulk_niche_by_group"] = xt.to_dict()

    # inter-niche Z
    zb = pb.copy(); scp.pp.normalize_total(zb); scp.pp.log1p(zb)
    M = pd.DataFrame(np.asarray(zb.X), index=zb.obs_names, columns=zb.var_names)
    M["niche"] = zb.obs["niche"].values
    pn = M.groupby("niche", observed=True).mean()
    ((pn - pn.mean()) / pn.std(ddof=0).replace(0, np.nan)).T.to_csv(
        os.path.join(T, "interniche_zscores.csv"))

    def annotate(r, ns):
        r = r.merge(f1[["gene", "det_focal", "det_nonastro", "f1_fail"]], on="gene", how="left")
        for n in ns:
            d = f2.get(n)
            if d is not None: r = r.merge(d.reset_index(), on="gene", how="left")
        fc = [c for c in r.columns if c.startswith("f2_fail_")]
        r["f2_fail_any"] = r[fc].fillna(False).any(axis=1) if fc else False
        r["seg_pass"] = ~(r["f1_fail"].fillna(False) | r["f2_fail_any"])
        r["significant"] = (r["padj"] < args.alpha) & (r["log2FoldChange"].abs() >= args.lfc)
        r["significant_seg_pass"] = r["significant"] & r["seg_pass"]
        return r.sort_values("padj")

    def run_de(sub, design, contrast, tag, ns, up_lab, dn_lab):
        n0 = sub.n_vars
        try:
            dc.pp.filter_by_expr(sub, group=design.replace("~", "").split("+")[0].strip(),
                                 min_count=args.min_count, min_total_count=args.min_total_count,
                                 large_n=args.large_n, min_prop=args.min_prop, inplace=True)
        except Exception as e:
            log(f"  {tag}: filter_by_expr skipped ({type(e).__name__})")
        counts = pd.DataFrame(np.rint(np.asarray(sub.X.todense() if sp.issparse(sub.X) else sub.X)
                                      ).astype(int), index=sub.obs_names, columns=sub.var_names)
        meta = sub.obs.copy()
        try:
            dds = DeseqDataSet(counts=counts, metadata=meta, design=design,
                               refit_cooks=True, quiet=True)
            dds.deseq2()
            st = DeseqStats(dds, contrast=contrast, quiet=True); st.summary()
            r = st.results_df.copy()
        except Exception as e:
            log(f"  FAILED {tag}: {type(e).__name__}: {e}"); return None
        r.index.name = "gene"; r = r.reset_index(); r["contrast"] = tag
        r["up_in"] = np.where(r["log2FoldChange"] > 0, up_lab, dn_lab)
        r = annotate(r, ns)
        r.to_csv(os.path.join(T, f"DE_{tag}.csv"), index=False)
        log(f"  {tag}: {sub.n_obs} pb, {n0}->{sub.n_vars} genes | "
            f"{int(r.significant.sum())} sig, {int(r.significant_seg_pass.sum())} seg-pass")
        return r

    # ============================================================ B. niche x niche
    sec("B. NICHE x NICHE DE")
    res_b = []
    use = [n for n in niches if int(xt.sum(axis=1).get(n, 0)) >= 6]
    for i in range(len(use)):
        for j in range(i+1, len(use)):
            a, b = use[i], use[j]
            sub = pb[pb.obs["niche"].isin([a, b])].copy()
            sub.obs["niche"] = pd.Categorical(sub.obs["niche"].astype(str), categories=[b, a])
            r = run_de(sub, "~niche", ["niche", a, b], f"nicheXniche_{a}_vs_{b}", [a, b],
                       f"niche{a}", f"niche{b}")
            if r is not None: res_b.append(r)
    if res_b:
        pd.concat(res_b, ignore_index=True).to_csv(
            os.path.join(T, "DE_B_niche_vs_niche_all.csv"), index=False)

    # ========================================================= C. case x control
    sec("C. CASE x CONTROL DE  (the comparison of primary interest)")
    res_c = []
    # C1 pooled across niches, niche as covariate
    sub = pb.copy()
    sub.obs["group"] = pd.Categorical(sub.obs["group"].astype(str), categories=["control", "ALS"])
    sub.obs["niche"] = pd.Categorical(sub.obs["niche"].astype(str))
    r = run_de(sub, "~group + niche", ["group", "ALS", "control"],
               "caseControl_POOLED_ALSvsControl", use, "ALS", "control")
    if r is not None: res_c.append(r)
    # C2 per niche
    for n in use:
        for lab, pos in (("ALS", None), ("c9", "c9"), ("sporadic", "sporadic")):
            s = pb[pb.obs["niche"] == n].copy()
            if pos is None:
                s.obs["grp"] = np.where(s.obs["status"].values == "control", "control", "ALS")
            else:
                keep = np.isin(s.obs["status"].values, [pos, "control"])
                s = s[keep].copy()
                s.obs["grp"] = np.where(s.obs["status"].values == "control", "control", pos)
            vc = pd.Series(s.obs["grp"]).value_counts()
            if len(vc) < 2 or vc.min() < args.min_per_group:
                log(f"  niche{n} {lab}: SKIP (groups {dict(vc)}, need >={args.min_per_group})")
                continue
            s.obs["grp"] = pd.Categorical(s.obs["grp"].astype(str), categories=["control", lab])
            r = run_de(s, "~grp", ["grp", lab, "control"],
                       f"caseControl_niche{n}_{lab}vsControl", [n], lab, "control")
            if r is not None: res_c.append(r)
    if res_c:
        pd.concat(res_c, ignore_index=True).to_csv(
            os.path.join(T, "DE_C_case_control_all.csv"), index=False)

    # ==================================================================== summary
    sec("SUMMARY")
    allr = pd.concat(res_b + res_c, ignore_index=True) if (res_b or res_c) else pd.DataFrame()
    if len(allr):
        allr.to_csv(os.path.join(T, "DE_all.csv"), index=False)
        summ = (allr.groupby("contrast")
                .agg(genes_tested=("gene", "size"), significant=("significant", "sum"),
                     seg_pass=("significant_seg_pass", "sum"))
                .assign(removed=lambda d: d.significant - d.seg_pass)
                .sort_index())
        print(summ.to_string())
        summ.to_csv(os.path.join(T, "DE_summary.csv"))
        man["summary"] = summ.reset_index().to_dict("records")
        sp2 = allr[allr.significant_seg_pass].sort_values("padj")
        sp2.to_csv(os.path.join(T, "DE_significant_seg_pass.csv"), index=False)
        cc = sp2[sp2.contrast.str.startswith("caseControl")]
        print(f"\nCase x control seg-passing DEGs: {len(cc)}")
        if len(cc):
            print(cc[["contrast", "gene", "up_in", "log2FoldChange", "padj",
                      "det_focal", "det_nonastro"]].head(30).to_string(index=False))
    man["finished"] = time.strftime("%F %T")
    json.dump(man, open(os.path.join(args.outdir, "manifest.json"), "w"), indent=1, default=str)
    log(f"wrote {args.outdir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
