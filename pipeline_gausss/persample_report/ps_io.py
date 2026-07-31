#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_io.py
#
# The original robust loader for the INDEPENDENT per-sample niche h5ads, working around
# anndata 0.10.8 refusing to read /uns/log1p/base when a newer anndata wrote it with
# encoding_type='null'. Builds the object from h5py directly.
#
# pipeline_final/ps_io_FINAL_fix.py is the descendant of this file and adds CSR/CSC dispatch
# on the on-disk encoding attribute. This version does not have it. If you run these sweep
# scripts against _FINAL output you will need that fix, since the _FINAL writer emits CSC
# counts and this loader would mis-shape the matrix.
# ========================================================================================

"""ps_io.py -- robust loader for the INDEPENDENT per-sample niches h5ads.

anndata 0.10.8 (SCG xenium_vistools) cannot read these files via read_h5ad:
/uns/log1p/base was written with encoding_type='null' by a newer anndata and
raises IORegistryError. We therefore build the AnnData directly from h5py:
X (csr), layers/counts (csr), var index, the obs columns we need, obsm/spatial_orig.
"""
import h5py, re, numpy as np, pandas as pd
import scipy.sparse as sp
import anndata as ad


def _decode(a):
    return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])


def _read_csr(g):
    data = g["data"][:]; indices = g["indices"][:]; indptr = g["indptr"][:]
    shape = tuple(g.attrs["shape"])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _read_obs_col(node):
    if isinstance(node, h5py.Group):  # categorical
        cats = list(_decode(node["categories"][:])); codes = np.asarray(node["codes"][:])
        # map unmapped codes (-1) to an explicit label matching the overlay reader,
        # so a missing code never becomes the string 'nan' and gets treated as a real niche
        if (codes < 0).any():
            fill = "unassigned" if "unassigned" in cats else "NA"
            if fill not in cats:
                cats = cats + [fill]
            codes = np.where(codes < 0, cats.index(fill), codes)
        return pd.Categorical.from_codes(codes, categories=cats)
    v = node[:]
    if v.dtype.kind == "S":
        return _decode(v)
    return v


def core_id(s):
    s = str(s)
    if ":" in s:
        s = s.split(":", 1)[1]
    return re.sub(r"-\d+$", "", s)


def load_indep(h5, obs_cols=None):
    """Return an AnnData with X (log1p-normalized), layers['counts'], var, the
    requested obs columns, and obsm['spatial_orig']. obs index = original obs_names."""
    default = ["novae_domains_n4", "novae_domains_n6", "novae_domains_n10",
               "is_MN", "status", "sample", "total_counts", "n_counts", "xenium_cell_id"]
    cols = obs_cols or default
    with h5py.File(h5, "r") as f:
        X = _read_csr(f["X"])
        counts = _read_csr(f["layers/counts"]) if "counts" in f["layers"] else None
        ik = f["var"].attrs.get("_index", "_index")
        if isinstance(ik, bytes):
            ik = ik.decode()
        var_index = _decode(f["var/" + ik][:])
        oik = f["obs"].attrs.get("_index", "_index")
        if isinstance(oik, bytes):
            oik = oik.decode()
        obs_index = _decode(f["obs/" + oik][:])
        obs = {}
        for c in cols:
            if c in f["obs"]:
                obs[c] = _read_obs_col(f["obs"][c])
        so = np.asarray(f["obsm/spatial_orig"]) if "spatial_orig" in f["obsm"] else None
        sp_ = np.asarray(f["obsm/spatial"]) if "spatial" in f["obsm"] else None
        # bundle path lives in uns
        bundle = None
        for k in ("xenium_outs_path", "sample_dir"):
            if k in f["uns"]:
                v = f["uns"][k][()]
                bundle = v.decode() if isinstance(v, bytes) else str(v)
                break
    var = pd.DataFrame(index=pd.Index(var_index, name=None))
    obs_df = pd.DataFrame(obs, index=pd.Index(obs_index, name=None))
    A = ad.AnnData(X=X, obs=obs_df, var=var)
    if counts is not None:
        A.layers["counts"] = counts
    if so is not None:
        A.obsm["spatial_orig"] = so
    if sp_ is not None:
        A.obsm["spatial"] = sp_
    A.uns["bundle"] = bundle
    A.obs["core"] = [core_id(i) for i in obs_index]
    return A


if __name__ == "__main__":
    import sys, scanpy as sc
    H = sys.argv[1] if len(sys.argv) > 1 else \
        "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_INDEPENDENT_MNcorrected/per_sample/SD03614_BG__niches_independent.h5ad"
    a = load_indep(H)
    print("built AnnData:", a.shape, "bundle:", a.uns["bundle"])
    print("X max (log1p?):", round(float(a.X[:2000].max()), 3),
          "counts max:", round(float(a.layers["counts"][:2000].max()), 1))
    for r in ("novae_domains_n4", "novae_domains_n6", "novae_domains_n10"):
        print(r, "->", a.obs[r].value_counts().to_dict())
    sub = a[a.obs["novae_domains_n6"] != "unassigned"].copy()
    sub.obs["novae_domains_n6"] = sub.obs["novae_domains_n6"].astype("category")
    sc.tl.rank_genes_groups(sub, "novae_domains_n6", method="wilcoxon", n_genes=8)
    df = sc.get.rank_genes_groups_df(sub, group=None)
    print("rank_genes OK. top per group:")
    for g, gg in df.groupby("group"):
        print("  ", g, list(gg.sort_values("scores", ascending=False)["names"].head(6)))
