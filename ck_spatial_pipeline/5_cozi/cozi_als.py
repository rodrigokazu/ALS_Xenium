#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/5_cozi/cozi_als.py
#
# COZI per section. For each section it builds a kNN graph on cell centroids (k 10), counts
# A to B neighbour edges for every ordered label pair and compares them with 300 label
# permutations (seed 1). The z score is conditional: it normalises by the A cells with at
# least one B neighbour, so A to B and B to A differ. cond_ratio is the fraction of A cells
# touching a B cell.
#
# Three label sets: cell types with astrocytes split by WDR49 raw count >= 1, niches, and the
# cell types again with MNs named MN_canonical. Case against control is a two-sided
# Mann-Whitney on per-section z, ALS against control, BH corrected. --restrict-vh limits
# the scope to in_VH_v2.
# ========================================================================================

"""
COZI directional neighbour preference on the ALS human Xenium cohort, with a
CASE x CONTROL comparison. Runs on WHOLE SECTIONS of the integrated_whole
NicheCompass object, whose niches are comparable across samples.

Two label sets, both per section:
  L1  cell types, with Astrocytes split by WDR49 status
      -> Astro_WDR49pos / Astro_WDR49neg + the other 8 types (+ Motor neurons)
      This is the primary run: it asks who the WDR49+ astrocytes prefer to sit next to.
  L2  NicheCompass niches -> directional niche-to-niche preference.

COZI (Schiller et al. 2026, Nat Commun) gives, for an ORDERED pair A -> B:
  zscore     conditional z of observed vs permuted neighbour counts, normalised by the
             number of A cells that have >=1 B neighbour (so A->B != B->A)
  cond_ratio fraction of A cells that actually neighbour a B cell, separating local
             enrichment from tissue-wide infiltration

Case x control: per ordered pair, Mann-Whitney U on the per-section z between ALS
(c9 + sporadic) and control sections, Benjamini-Hochberg corrected.

cozipy 0.1.4 has normalize_zscore=True, which is the sqrt(n cells) division FB_SPI
applied by hand with 0.1.1.

READ-ONLY on the input object.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

OBJ = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC/"
       "runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def sec(m): print("\n"+"="*78+f"\n{m}\n"+"="*78, flush=True)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); r = p[o] * n / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(r[::-1])[::-1]
    return np.clip(q, 0, 1)

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
    p.add_argument("--gene", default="WDR49")
    p.add_argument("--gene-min", type=int, default=1)
    p.add_argument("--n-neighbors", type=int, default=10)     # FB_SPI value
    p.add_argument("--n-permutations", type=int, default=300)  # FB_SPI value
    p.add_argument("--min-cell-count", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--restrict-vh", action="store_true",
                   help="restrict to obs['in_VH_v2'] -- the ventral-horn mask")
    p.add_argument("--chunk", type=int, default=100000)
    return p.parse_args(a)

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    T = os.path.join(args.outdir, "tables"); os.makedirs(T, exist_ok=True)
    import h5py
    from scipy.stats import mannwhitneyu
    from cozipy import run_cozi
    import cozipy
    man = {"object": args.obj, "params": vars(args), "cozipy": cozipy.__version__,
           "started": time.strftime("%F %T")}
    log(f"cozipy {cozipy.__version__}")

    sec("LOAD  obs + spatial + WDR49 counts")
    f = h5py.File(args.obj, "r")
    def cat(k):
        g = f["obs"][k]
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array(c)[g["codes"][:]]
    ct = cat(args.celltype_col); niche = cat(args.niche_col)
    samp = cat(args.sample_col); status = cat("status")
    xy = f["obsm"]["spatial"][:]
    def nb(k):
        g = f["obs"][k]; return g["values"][:] & ~g["mask"][:]
    mn_canon = nb("is_MN_v2")
    in_vh = nb("in_VH_v2")
    var = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][:]]
    gj = var.index(args.gene)
    L = f["layers"]["counts"]; indptr = L["indptr"][:]
    idx, dat = L["indices"], L["data"]
    n_obs = len(indptr) - 1
    w = np.zeros(n_obs, dtype=np.int32)
    for s in range(0, n_obs, args.chunk):
        e = min(s + args.chunk, n_obs); a, b = indptr[s], indptr[e]
        ii = idx[a:b].astype(np.int64); dd = dat[a:b]
        hit = ii == gj
        if hit.any():
            rows = np.repeat(np.arange(s, e), np.diff(indptr[s:e+1]))[hit]
            np.add.at(w, rows, dd[hit].astype(np.int32))
    f.close()
    log(f"{n_obs:,} cells; spatial {xy.shape}; {args.gene}+ overall {(w>=args.gene_min).sum():,}")
    log(f"canonical MN (is_MN_v2) {mn_canon.sum():,}; in_VH_v2 {in_vh.sum():,}")

    keep = np.ones(n_obs, dtype=bool)
    if args.restrict_vh:
        keep = in_vh
        log(f"RESTRICTED to the ventral-horn mask: {keep.sum():,} cells, "
            f"{pd.Series(samp[keep]).nunique()} sections")
    man["restrict_vh"] = bool(args.restrict_vh)
    man["n_cells_used"] = int(keep.sum())

    # L1 labels: cell type, astrocytes split by WDR49
    lab1 = ct.copy().astype(object)
    m_astro = ct == "Astrocytes"
    lab1[m_astro & (w >= args.gene_min)] = "Astro_WDR49pos"
    lab1[m_astro & (w < args.gene_min)] = "Astro_WDR49neg"
    lab1 = lab1.astype(str)
    # L3: as L1 but with the CANONICAL curated motor neurons pulled out of the
    # coarse 'Motor neurons' class, so MN_canonical is its own node
    # With cell_type_mn, 'Motor neurons' already == is_MN_v2 and the rest are
    # 'Other neurons'. L3 just renames it so the node is unambiguous in the output.
    lab3 = lab1.copy().astype(object)
    lab3[lab3 == "Motor neurons"] = "MN_canonical"
    lab3[mn_canon] = "MN_canonical"
    lab3 = lab3.astype(str)
    log("L1 label counts:\n" + pd.Series(lab1[keep]).value_counts().to_string())
    log("L3 label counts:\n" + pd.Series(lab3[keep]).value_counts().to_string())
    lab2 = np.array(["niche" + x for x in niche])
    # apply the spatial restriction to everything downstream
    ct, niche, samp, status = ct[keep], niche[keep], samp[keep], status[keep]
    xy = xy[keep]; lab1, lab2, lab3 = lab1[keep], lab2[keep], lab3[keep]

    dstat = pd.Series(status, index=samp).groupby(level=0).first()
    group = np.where(status == "control", "control", "ALS")
    dgrp = pd.Series(group, index=samp).groupby(level=0).first()
    log("sections by group: " + str(dict(dgrp.value_counts())))
    man["sections_by_group"] = {k: int(v) for k, v in dgrp.value_counts().items()}

    sections = sorted(pd.unique(samp))
    results = {}
    for name, labels in (("celltype_wdr49", lab1), ("niche", lab2),
                         ("celltype_mn_canonical", lab3)):
        sec(f"COZI  labels = {name}  (k={args.n_neighbors}, {args.n_permutations} perms)")
        per = []
        for s in sections:
            m = samp == s
            co = xy[m].astype(float); la = labels[m]
            t0 = time.time()
            df = run_cozi(co, la, nbh_def="knn", n_neighbors=args.n_neighbors,
                          n_permutations=args.n_permutations, random_state=args.seed,
                          normalize_zscore=True, min_cell_count=args.min_cell_count)
            df["sample"] = s; df["status"] = dstat[s]; df["group"] = dgrp[s]
            per.append(df)
            log(f"  {s:14s} n={m.sum():7,} pairs={len(df):5d}  {time.time()-t0:5.1f}s")
        allp = pd.concat(per, ignore_index=True)
        allp.to_csv(os.path.join(T, f"cozi_persection_{name}.csv"), index=False)
        log(f"wrote per-section table: {len(allp):,} rows")

        # case x control per ordered pair
        rows = []
        for (a, b), g in allp.groupby(["index_cell_type", "neighbor_cell_type"]):
            za = g.loc[g.group == "ALS", "zscore"].dropna().values
            zc = g.loc[g.group == "control", "zscore"].dropna().values
            if len(za) < 3 or len(zc) < 3: continue
            try: u, p = mannwhitneyu(za, zc, alternative="two-sided")
            except Exception: continue
            rows.append(dict(index_cell_type=a, neighbor_cell_type=b,
                             n_ALS=len(za), n_control=len(zc),
                             z_ALS=float(np.mean(za)), z_control=float(np.mean(zc)),
                             delta_z=float(np.mean(za) - np.mean(zc)),
                             cr_ALS=float(g.loc[g.group == "ALS", "cond_ratio"].mean()),
                             cr_control=float(g.loc[g.group == "control", "cond_ratio"].mean()),
                             U=float(u), p=float(p)))
        cc = pd.DataFrame(rows)
        if len(cc):
            cc["padj"] = bh(cc["p"].values)
            cc["significant"] = cc["padj"] < 0.05
            cc = cc.sort_values("p")
            cc.to_csv(os.path.join(T, f"cozi_case_control_{name}.csv"), index=False)
            nsig = int(cc.significant.sum())
            log(f"case x control: {len(cc)} testable ordered pairs, {nsig} significant at BH<0.05")
            print(cc.head(20).to_string(index=False))
            results[name] = dict(pairs=len(cc), significant=nsig)
            # focal view: anything involving the WDR49+ astrocytes
            for focal_lab, fname in (("Astro_WDR49pos", "WDR49pos"),
                                     ("MN_canonical", "MNcanonical")):
                if focal_lab not in set(cc.index_cell_type) | set(cc.neighbor_cell_type):
                    continue
                foc = cc[(cc.index_cell_type == focal_lab) |
                         (cc.neighbor_cell_type == focal_lab)].copy()
                foc.to_csv(os.path.join(T, f"cozi_case_control_{fname}_focus.csv"), index=False)
                print(f"\n{focal_lab} pairs: {len(foc)}, "
                      f"{int(foc.significant.sum())} significant at BH<0.05")
                print(foc.head(20).to_string(index=False))
        else:
            log("no testable pairs")

    man["results"] = results
    man["finished"] = time.strftime("%F %T")
    json.dump(man, open(os.path.join(args.outdir, "manifest.json"), "w"), indent=1, default=str)
    log(f"wrote {args.outdir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
