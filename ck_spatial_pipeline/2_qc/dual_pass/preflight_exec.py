#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/preflight_exec.py
#
# Executes the injected relabel cell for all 20 samples against the real parquet and a real
# minimal AnnData before any notebook gets patched or any array goes in. py_compile and
# dry runs had all passed on code that then failed on real data. This gate catches that
# class of failure in minutes on the login node.
# ========================================================================================

"""EXECUTE the injected relabel cell for all 20 samples against the real parquet
and a real minimal AnnData, before any notebook is patched or any array submitted.

Why this exists. py_compile, the render-and-compile pass, --self-test and --dry-run
all passed on a cell that raised `KeyError: 'vh_annotated'` the moment it touched
the parquet, because every one of those checks is static. A 20-task SLURM array is
a very slow way to discover a missing dict key. This runs the actual cell body.

It builds a small but genuine AnnData per sample from that sample's niche h5ad:
real obs_names (so cell_index is real), real obsm['spatial'] (so the coordinate
gate is real), the real pristine obs['is_MN']/obs['is_neuron'] (so the
EXPECTED_NATIVE_MN tripwire is real), and the real GAD1/GAD2 counts columns (so
the GAD cross-check is real). Everything the cell actually reads is genuine; only
the 478 unused genes are dropped, for speed.

usage: preflight_exec.py [SAMPLE ...]        (default: all 20)
exit 0 = every sample executed the cell and the trace cell cleanly.
"""
import importlib.util
import subprocess
import sys
import traceback

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

CODE = "/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun/relabel_v6_at_source.py"
NICHE = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
         "Novae_persample_INDEPENDENT_FINAL/per_sample/{s}__niches_independent.h5ad")
GENES = ("GAD1", "GAD2")

spec = importlib.util.spec_from_file_location("rl", CODE)
RL = importlib.util.module_from_spec(spec)
sys.modules["rl"] = RL
spec.loader.exec_module(RL)


def read_str(h5, path):
    node = h5[path]
    if isinstance(node, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in node["categories"][:]]
        return np.asarray(cats, dtype=object)[node["codes"][:]]
    arr = node[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr], dtype=object)


def build(sample):
    """A real AnnData carrying exactly what the injected cell reads."""
    with h5py.File(NICHE.format(s=sample), "r") as h:
        idx = h["obs"].attrs.get("_index", b"_index")
        idx = idx.decode() if isinstance(idx, bytes) else idx
        obs_names = read_str(h, f"obs/{idx}")
        n = len(obs_names)

        vk = h["var"].attrs.get("_index", b"_index")
        vk = vk.decode() if isinstance(vk, bytes) else vk
        var_names = read_str(h, f"var/{vk}")

        obs = {}
        for c in ("is_MN", "is_neuron"):
            if c in h["obs"]:
                obs[c] = h[f"obs/{c}"][:].astype(bool)
        spatial = h["obsm/spatial"][:]

        # pull only the GAD columns out of layers/counts, whatever its encoding
        g = h["layers/counts"]
        enc = g.attrs.get("encoding-type", b"")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        shape = tuple(g.attrs["shape"])
        from scipy.sparse import csc_matrix
        parts = (g["data"][:], g["indices"][:], g["indptr"][:])
        M = (csc_matrix(parts, shape=shape) if "csc" in enc
             else csr_matrix(parts, shape=shape)).tocsc()
        cols, keep = [], []
        for gene in GENES:
            w = np.where(var_names == gene)[0]
            if len(w):
                cols.append(np.asarray(M[:, w[0]].todense()).ravel())
                keep.append(gene)
        X = np.vstack(cols).T if cols else np.zeros((n, 0))

    a = ad.AnnData(X=csr_matrix(X.astype(np.float32)),
                   obs=pd.DataFrame(obs, index=pd.Index(obs_names, name="cell")),
                   var=pd.DataFrame(index=pd.Index(keep, name="gene")))
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = np.asarray(spatial, dtype=float)
    return a


def run_one(s):
    """Execute the injected cell + trace cell for one sample. Raises on failure."""
    adata = build(s)
    ns = {"adata": adata, "__name__": "preflight"}
    exec(compile(RL.render(RL.CELL, s), f"<cell:{s}>", "exec"), ns)
    ns["sample"] = ns["adata"].copy()      # the trace cell reads `sample` too
    exec(compile(RL.render(RL.TRACE_EXTRA, s), f"<trace:{s}>", "exec"), ns)
    t = ns.get("__TRACE__", {})
    return (f"n_obs={adata.n_obs} MN={t.get('n_is_MN_raw')} "
            f"NA={t.get('n_is_MN_na_raw')} vh={t.get('has_vh_data')} "
            f"coord_delta={t.get('coord_gate_max_delta_um')}")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("-")]

    # --one: the child. Execute a single sample in this process and exit.
    if "--one" in sys.argv:
        try:
            print(f"RESULT {args[0]} OK {run_one(args[0])}")
            return 0
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"RESULT {args[0]} FAIL {type(exc).__name__}: {exc}")
            return 2

    # driver: one SUBPROCESS PER SAMPLE. The cell reads the whole 2,063,633-row
    # parquet every time, and twenty of those in one interpreter exhausts the login
    # node -- which surfaces as MemoryError/ArrowMemoryError and is indistinguishable
    # from a real code fault. A fresh process per sample also matches how the SLURM
    # array actually runs it, so a pass here means more.
    todo = args or RL.SAMPLES
    fails, out = [], []
    print(f"{'sample':<12} result")
    print("-" * 92)
    for s in todo:
        p = subprocess.run([sys.executable, __file__, "--one", s],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), "")
        if " OK " in line:
            print(f"{s:<12} OK   {line.split(' OK ', 1)[1]}")
        else:
            detail = (line.split(" FAIL ", 1)[1] if " FAIL " in line
                      else (p.stderr.strip().splitlines() or ["no output"])[-1])
            print(f"{s:<12} FAIL {detail}")
            fails.append((s, detail))
            out.append(p.stderr)
    print("-" * 92)
    if fails:
        print(f"{len(fails)} / {len(todo)} FAILED:")
        for s, e in fails:
            print(f"  {s}: {e}")
        if out:
            print("\nfirst traceback:\n" + out[0][-2500:])
        return 2
    print(f"all {len(todo)} samples executed the injected cell cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
