# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/novae_INDEPENDENT_FINAL.py
#
# The same tool pointed at the opposite question. Here every sample gets its own freshly
# trained Novae model, so domains are NOT comparable across sections and any attempt to
# compare composition between samples using this output is a mistake.
#
# The reason to do it this way is QC. A model that only ever sees one section will happily
# give a segmentation smear, a tissue edge or a patch of debris its own domain, because within
# that section the artefact really is a distinct neighbourhood. Pooling hides exactly that.
# This output is what the dual-pass QC consumes.
#
# Run as a SLURM array, one sample per task. That is the main structural change from the
# _gausss version, where all 20 looped inside a single job. A failure now costs one sample
# instead of the run.
#
# num_workers is 2 here, not 8, and there is a reason for it. Fork-heavy GPU dataloader
# workers sharing a process with the HDF5 writer produced a segfault; the backup of the
# previous version still sits on SCG as novae_INDEPENDENT_FINAL.py.bak_segfault_20260725.
# Raise it only if you also separate the write.
#
# Targets are 4, 6, 8 and 10 domains with 6 primary.
# ========================================================================================

"""
novae_INDEPENDENT_FINAL.py
============================================================
PER-SAMPLE INDEPENDENT Novae niches on the MN-corrected _FINAL (Marcel's
Ranger_procd_mw_final segmentation) data — one FRESH Novae model per sample, NO
shared/comparable domains across samples.

Purpose: QC. A per-sample independent model makes segmentation "smear" and
tissue-edge / debris artefacts show up as their own spurious domain within a
section, so these maps are for eyeballing and for the dual-pass QC that runs on
this output — NOT for cross-sample composition comparison (use the JOINT run for
that: novae_JOINT_FINAL.py -> Novae_persample_niches_FINAL/).

*** RUN AS A SLURM JOB ARRAY (delta vs the _gausss reference) ***
The proven _gausss INDEPENDENT script looped all 20 samples inside ONE job. For
_FINAL it is a SLURM ARRAY: ONE sample per SLURM_ARRAY_TASK_ID (runner --array=0-19%5),
so samples fit in parallel and one failure does not sink the batch.

  * INPUT per task: the per-sample _FINAL h5ad (cfg.persample_h5ad(label) =
    H5AD_DIR/<LABEL>__MNcorrected_FINAL.h5ad). The 20 per-sample files are sorted by
    label; SLURM_ARRAY_TASK_ID indexes into that sorted list. FALLBACK (honours
    "Input = combined FINAL h5ad"): if the 20 per-sample files are not present, the
    task loads cfg.COMBINED_H5AD and subsets to the task_id-th sample (labels sorted
    the SAME way, so the index->sample mapping is identical in both modes).
  * Two run modes (dispatched in __main__):
       python novae_INDEPENDENT_FINAL.py <task_id>   -> process ONE sample
       python novae_INDEPENDENT_FINAL.py aggregate   -> stitch the 20 per-sample
                                                        summaries into the cohort
                                                        summary + run_manifest.json
    Each task writes ONLY its own files (no shared-file write races under %5); the
    aggregate step is a tiny post-array pass (see run_novae_INDEPENDENT_FINAL.sh).

Same Novae conventions as the JOINT run (spatial-crew review):
  * Graph on TRUE microns (obsm['spatial_orig'] -> obsm['spatial']), Delaunay,
    percentile edge pruning. NO technology='xenium' (it rebuilds coords). Single
    slide per task -> no slide_key. (The spatial<-spatial_orig swap ALSO undoes the
    grid offset in the combined-subset fallback, which is why it is unconditional.)
  * Only filtering: min_counts>=1 (drop empty cells). Raw counts kept in layers.
  * Multi-resolution: hit sensible domain COUNTS via a per-sample level scan
    (assign_domains exact-matches-or-raises -> scan levels, pick nearest per target).
    NaN domain -> 'unassigned'; `del adata.obs[col]` not .pop. SEED=0.
  * INDEP domain sweep: TARGET_N_DOMAINS = [4, 6, 8, 10] (write novae_domains_n{tgt}).
  * MNs (obs['is_MN']) overlaid on every map + per-sample MN x domain summary;
    degrades gracefully to placeholders when is_MN is all-False (annotation pending).

Env: novae_env (Python 3.11 venv), modules python/3.11.1 + gcc/13.3.0, GPU.
"""

import os
import sys

sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
FINAL_CODE_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code"
if FINAL_CODE_DIR not in sys.path:
    sys.path.insert(0, FINAL_CODE_DIR)

import glob
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import novae
import torch

import final_config as cfg
try:
    from mn_annotation_join import MN_LABEL_SPEC as _MN_LABEL_SPEC
except Exception:
    _MN_LABEL_SPEC = {"column": "MN_motorneuron", "positive_values": ["yes"]}

# ============================================================
# CONFIG  (paths imported from final_config; NOT re-derived)
# ============================================================
INPUT_DIR = str(cfg.H5AD_DIR)
PERSAMPLE_SUFFIX = cfg.PERSAMPLE_SUFFIX             # "__MNcorrected_FINAL.h5ad"
INPUT_GLOB = f"*{PERSAMPLE_SUFFIX}"
COMBINED_H5AD = str(cfg.COMBINED_H5AD)              # fallback input
OUTPUT_DIR = str(cfg.NOVAE_INDEP_DIR)
EXPECTED_SAMPLE_COUNT = int(cfg.EXPECTED_SAMPLE_COUNT)   # 20

SAMPLE_KEY = "sample"
STATUS_KEY = "status"
MN_KEY = "is_MN"

DELAUNAY = True
EDGE_PERCENTILE = 99

TARGET_N_DOMAINS = [4, 6, 8, 10]    # per-sample INDEP resolutions (write novae_domains_n{tgt})
PRIMARY_TARGET = 6
SCAN_LEVELS = list(range(1, 13))
MAX_EPOCHS = 50
NUM_WORKERS = 2   # was 8: fork-heavy GPU dataloader workers + HDF5 writer in one process was
                  # segfaulting mid-write on 10/20 samples (SIGSEGV right after "Computing
                  # representations: 100%", corrupting the h5ad); lower worker count + the
                  # explicit teardown/atomic-write below are the fix (2026-07-25).
MIN_COUNTS = 1
UNASSIGNED = "unassigned"
POINT_SIZE = 2.0
DPI = 200
SEED = 0

# Output subdirs (per_sample h5ads, plots, per-sample manifests, per-sample summary rows).
SUB_PERSAMPLE = "per_sample"
SUB_PLOTS = "plots"
SUB_SUMMARY = "summary"
SUB_MANIFESTS = os.path.join(SUB_SUMMARY, "persample")
SUB_ROWS = os.path.join(SUB_SUMMARY, "persample_rows")
for sub in (SUB_PERSAMPLE, SUB_PLOTS, SUB_SUMMARY, SUB_MANIFESTS, SUB_ROWS):
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def log(m): print(m, flush=True)


def coerce_is_MN(series_or_none, n):
    """REAL numpy bool of length n; degrade safely (see novae_JOINT_FINAL.coerce_is_MN)."""
    if series_or_none is None:
        log("  [MN] WARNING obs['is_MN'] MISSING -> is_MN all-False (annotation pending)")
        return np.zeros(n, dtype=bool)
    s = series_or_none
    dt = s.dtype
    if dt == bool:
        return np.asarray(s.to_numpy(), dtype=bool)
    if str(dt) in ("category", "object") or getattr(dt, "kind", "") in ("U", "S", "O"):
        vals = s.astype("object").astype(str).str.strip().str.lower()
        out = vals.isin({"true", "1", "yes", "t"}).to_numpy()
        log(f"  [MN] NOTE obs['is_MN'] was {dt}; coerced truthy-token -> bool (True={int(out.sum())})")
        return np.asarray(out, dtype=bool)
    return np.asarray(pd.to_numeric(s, errors="coerce").fillna(0).to_numpy() != 0, dtype=bool)


def _setup_compute():
    """GPU/seed setup (called once per task; kept out of aggregate())."""
    use_cuda = torch.cuda.is_available()
    accelerator = "cuda" if use_cuda else "cpu"
    log(f"[novae] torch={torch.__version__} cuda_available={use_cuda} -> accelerator='{accelerator}'")
    if not use_cuda:
        log("[novae] WARNING: no GPU detected — training on CPU will be slow.")
    import random as _random
    np.random.seed(SEED); _random.seed(SEED); torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(SEED)
    try:
        novae.utils.set_seed(SEED)
    except Exception:
        pass
    log(f"[seed] SEED={SEED}")
    return accelerator


def assign_target_domains(model, ad):
    """Level-scan -> pick the level nearest each target count; return {target: alias_key}, chosen, counts.

    Preserves the reference gotchas: assign_domains exact-matches-or-raises so we SCAN levels,
    NaN domain -> 'unassigned', and drop raw scan cols with `del` (NOT .pop, which would
    TypeError with no default).
    """
    level_keys, level_counts = {}, {}
    for L in SCAN_LEVELS:
        try:
            k = model.assign_domains(ad, level=L)
        except Exception as e:
            log(f"    [scan] level {L} unavailable: {type(e).__name__}")
            continue
        level_keys[L] = k
        level_counts[L] = int(ad.obs[k].dropna().astype(str).nunique())
    if not level_counts:
        raise RuntimeError("no Novae levels could be assigned")
    chosen, used, dkeys = {}, set(), {}
    for tgt in TARGET_N_DOMAINS:
        cand = sorted(level_counts.items(), key=lambda kv: (abs(kv[1] - tgt), kv[0]))
        lvl = next((L for L, _ in cand if L not in used), cand[0][0])
        used.add(lvl); chosen[tgt] = lvl
    for tgt, lvl in chosen.items():
        src = level_keys[lvl]
        s = ad.obs[src].astype("object")
        s = s.where(ad.obs[src].notna(), UNASSIGNED).astype(str)
        alias = f"novae_domains_n{tgt}"
        ad.obs[alias] = pd.Categorical(s)
        dkeys[tgt] = alias
    for L, k in level_keys.items():
        if (k in ad.obs) and (k not in set(dkeys.values())):
            del ad.obs[k]
    return dkeys, chosen, level_counts


def spatial_plot(ad, dom_key, sample_name, n_dom, path):
    xy = np.asarray(ad.obsm["spatial_orig"], dtype=float)
    domains = sorted(ad.obs[dom_key].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    dom_to_color = {d: ("0.8" if d == UNASSIGNED else cmap(i % 20)) for i, d in enumerate(domains)}
    dvals = ad.obs[dom_key].astype(str).values
    fig, ax = plt.subplots(figsize=(7, 7))
    for d in domains:
        m = dvals == d
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, c=[dom_to_color[d]], label=d, linewidths=0)
    mmn = ad.obs[MN_KEY].values.astype(bool)
    if mmn.any():
        ax.scatter(xy[mmn, 0], xy[mmn, 1], s=POINT_SIZE * 6, facecolors="none",
                   edgecolors="k", linewidths=0.4, label=f"MN (n={int(mmn.sum())})", zorder=5)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{sample_name} — INDEPENDENT niches (~{n_dom} domains) + MNs")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(markerscale=4, fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5), title="domain")
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)


# ============================================================
# INPUT RESOLUTION  (per-sample file primary; combined-subset fallback)
# ============================================================
def _persample_files():
    return sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_GLOB)))


def resolve_input(task_id):
    """Return (adata, label, source_desc) for this array task.

    Primary: the sorted per-sample _FINAL h5ads (index by task_id). Fallback: subset the
    combined _FINAL h5ad to the task_id-th sample (labels sorted the SAME way -> identical
    index->sample mapping). Raises loudly if neither input is available for this index.
    """
    files = _persample_files()
    if len(files) == EXPECTED_SAMPLE_COUNT:
        if task_id < 0 or task_id >= len(files):
            raise IndexError(f"task_id {task_id} out of range for {len(files)} per-sample files")
        f = files[task_id]
        base = os.path.basename(f).replace(PERSAMPLE_SUFFIX, "")
        log(f"[input] per-sample file [{task_id}/{len(files)-1}]: {os.path.basename(f)}")
        ad = sc.read_h5ad(f)
        return ad, base, f"persample:{os.path.basename(f)}"

    # --- fallback: subset the combined FINAL h5ad -----------------------------
    log(f"[input] per-sample files present: {len(files)} != {EXPECTED_SAMPLE_COUNT} expected "
        f"-> FALLBACK to subsetting the combined _FINAL h5ad.")
    if not os.path.exists(COMBINED_H5AD):
        raise FileNotFoundError(
            f"Neither {EXPECTED_SAMPLE_COUNT} per-sample files ({INPUT_DIR}/{INPUT_GLOB}) nor the "
            f"combined h5ad ({COMBINED_H5AD}) are available. Run the concat builder first.")
    log(f"[input] loading combined {COMBINED_H5AD} (heavy) to subset one sample ...")
    adata_all = sc.read_h5ad(COMBINED_H5AD)
    labels = sorted(adata_all.obs[SAMPLE_KEY].astype(str).unique())
    if len(labels) != EXPECTED_SAMPLE_COUNT:
        log(f"[input] WARNING combined has {len(labels)} samples (expected {EXPECTED_SAMPLE_COUNT}); "
            f"index->sample mapping uses the sorted label list as-is.")
    if task_id < 0 or task_id >= len(labels):
        raise IndexError(f"task_id {task_id} out of range for {len(labels)} samples in combined h5ad")
    label = labels[task_id]
    log(f"[input] combined-subset [{task_id}/{len(labels)-1}]: sample '{label}'")
    ad = adata_all[adata_all.obs[SAMPLE_KEY].astype(str) == label].copy()
    del adata_all
    return ad, label, f"combined-subset:{label}"


# ============================================================
# PROCESS ONE SAMPLE  (one array task)
# ============================================================
def process_one(task_id):
    log(f"\n==================== ARRAY TASK {task_id} ====================")
    accelerator = _setup_compute()

    ad, base, source_desc = resolve_input(task_id)
    log(f"[task {task_id}] {base}: n_obs={ad.n_obs}  source={source_desc}")

    assert "spatial_orig" in ad.obsm, f"{base}: spatial_orig missing"
    ad.obs[MN_KEY] = coerce_is_MN(ad.obs[MN_KEY] if MN_KEY in ad.obs else None, ad.n_obs)
    n_mn = int(ad.obs[MN_KEY].sum())
    is_MN_all_false = (n_mn == 0)
    sample_name = str(ad.obs[SAMPLE_KEY].iloc[0]) if SAMPLE_KEY in ad.obs else base
    status = str(ad.obs[STATUS_KEY].iloc[0]) if STATUS_KEY in ad.obs else "NA"

    # coords: TRUE microns for the graph (also undoes the grid offset in combined-subset mode)
    ad.obsm["spatial_grid_offset"] = np.asarray(ad.obsm["spatial"], dtype=float)
    ad.obsm["spatial"] = np.asarray(ad.obsm["spatial_orig"], dtype=float)

    n0 = ad.n_obs
    sc.pp.filter_cells(ad, min_counts=MIN_COUNTS)
    log(f"  [filter] min_counts>={MIN_COUNTS}: {n0} -> {ad.n_obs}  | MN={int(ad.obs[MN_KEY].sum())}")
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    # independent graph (single slide -> no slide_key) + fresh model
    novae.spatial_neighbors(ad, coord_type="generic", delaunay=DELAUNAY, percentile=EDGE_PERCENTILE)
    model = novae.Novae(ad)
    model.fit(ad, max_epochs=MAX_EPOCHS, accelerator=accelerator, num_workers=NUM_WORKERS)
    model.compute_representations(ad, accelerator=accelerator, num_workers=NUM_WORKERS)

    dkeys, chosen, level_counts = assign_target_domains(model, ad)
    primary_key = dkeys.get(PRIMARY_TARGET, dkeys[TARGET_N_DOMAINS[0]])
    log(f"  [domains] targets->levels {chosen} | primary '{primary_key}' "
        f"({ad.obs[primary_key].nunique()} labels)")

    # Tear down the GPU model/dataloader workers BEFORE the HDF5 write. The corruption seen on
    # 10/20 samples was a SIGSEGV firing right after "Computing representations: 100%", i.e.
    # while lingering CUDA context / forked dataloader worker cleanup raced the h5py write in
    # the same process -> truncated ("bad object header version") files. Explicit teardown +
    # writing to a temp path and atomically renaming means a crash (if it still happens) leaves
    # no corrupt file at the real path -- the next rerun just sees the sample as still missing.
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # save lean per-sample object
    ad.obsp.clear()
    safe = base.replace("/", "_")
    final_path = os.path.join(OUTPUT_DIR, SUB_PERSAMPLE, f"{safe}__niches_independent.h5ad")
    tmp_path = final_path + ".tmp"
    ad.write_h5ad(tmp_path, compression="gzip")
    os.replace(tmp_path, final_path)

    # one spatial map per target resolution (with MN overlay) + per-sample summary rows
    mn = ad.obs[MN_KEY].values.astype(bool)
    rows = []
    for tgt, key in dkeys.items():
        n_dom = ad.obs[key].astype(str).nunique()
        spatial_plot(ad, key, sample_name, n_dom,
                     os.path.join(OUTPUT_DIR, SUB_PLOTS, f"{safe}__niches_n{tgt}.png"))
        # per-sample MN concentration at this resolution (QC signal)
        if mn.any():
            vc = ad.obs.loc[mn, key].astype(str).value_counts(normalize=True)
            top_dom, top_frac = vc.index[0], float(vc.iloc[0])
        else:
            top_dom, top_frac = "NA", np.nan
        rows.append({
            "sample": sample_name, "status": status, "target_n_domains": tgt,
            "level_chosen": chosen[tgt], "n_domains_actual": int(n_dom),
            "n_cells": int(ad.n_obs), "n_MN": int(mn.sum()),
            "is_MN_all_false": bool(is_MN_all_false),
            "MN_top_domain": top_dom, "MN_top_domain_frac": top_frac,
            "input_source": source_desc,
        })

    # per-sample summary rows (aggregate stitches these; no shared-file write race)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, SUB_ROWS, f"{safe}__rows.csv"), index=False)

    # per-sample manifest
    with open(os.path.join(OUTPUT_DIR, SUB_MANIFESTS, f"{safe}__manifest.json"), "w") as fh:
        json.dump({
            "task_id": int(task_id), "sample": sample_name, "status": status,
            "n_cells": int(ad.n_obs), "n_MN": int(mn.sum()),
            "is_MN_all_false": bool(is_MN_all_false),
            "input_source": source_desc, "target_n_domains": TARGET_N_DOMAINS,
            "primary_target": PRIMARY_TARGET, "chosen_level_per_target": chosen,
            "level_scan_counts": level_counts, "domain_keys_by_target": dkeys,
            "graph": {"delaunay": DELAUNAY, "edge_percentile": EDGE_PERCENTILE,
                      "coords": "spatial_orig"},
            "min_counts_filter": MIN_COUNTS, "accelerator": accelerator, "seed": SEED,
            "MN_LABEL_SPEC": _MN_LABEL_SPEC,
        }, fh, indent=2, default=str)

    log(f"  [OK] task {task_id} {sample_name}  (n_MN={n_mn}"
        + ("  [is_MN all-False -> MN outputs are placeholders]" if is_MN_all_false else "")
        + ")")


# ============================================================
# AGGREGATE  (tiny post-array pass: stitch per-sample summaries)
# ============================================================
def aggregate():
    log("[aggregate] stitching per-sample INDEPENDENT summaries")
    rows_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, SUB_ROWS, "*__rows.csv")))
    frames = []
    for f in rows_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            log(f"  [aggregate] WARN could not read {os.path.basename(f)}: {type(e).__name__}: {e}")
    if frames:
        df = pd.concat(frames, ignore_index=True).sort_values(["sample", "target_n_domains"])
        out_csv = os.path.join(OUTPUT_DIR, SUB_SUMMARY, "persample_independent_summary.csv")
        df.to_csv(out_csv, index=False)
        log(f"  [aggregate] wrote {out_csv} ({df.shape[0]} rows)")
        log(df.to_string(index=False))
    else:
        log("  [aggregate] no per-sample rows found yet (run the array first).")

    present = sorted({os.path.basename(f).replace("__rows.csv", "") for f in rows_files})
    input_files = _persample_files()
    expected = (sorted({os.path.basename(f).replace(PERSAMPLE_SUFFIX, "") for f in input_files})
                if input_files else present)
    missing = sorted(set(expected) - set(present))

    with open(os.path.join(OUTPUT_DIR, SUB_SUMMARY, "run_manifest.json"), "w") as fh:
        json.dump({
            "mode": "per-sample INDEPENDENT (SLURM array; no shared domains) — for smear/QC + dual-pass QC",
            "input_dir": INPUT_DIR, "input_glob": INPUT_GLOB,
            "combined_fallback": COMBINED_H5AD, "output_dir": OUTPUT_DIR,
            "expected_sample_count": EXPECTED_SAMPLE_COUNT,
            "n_present": len(present), "present": present,
            "n_missing": len(missing), "missing": missing,
            "target_n_domains": TARGET_N_DOMAINS, "primary_target": PRIMARY_TARGET,
            "graph": {"delaunay": DELAUNAY, "edge_percentile": EDGE_PERCENTILE, "coords": "spatial_orig"},
            "min_counts_filter": MIN_COUNTS, "seed": SEED,
            "note": ("MN_top_domain labels are PER-SAMPLE and NOT comparable across samples (independent "
                     "models); only MN_top_domain_frac is cross-comparable. For smear/QC eyeballing + the "
                     "dual-pass QC that consumes per_sample/*__niches_independent.h5ad, not cross-sample "
                     "domain identity. 'missing' = expected samples with no per-sample rows (a failed/"
                     "unrun array task -- check the SLURM logs)."),
        }, fh, indent=2, default=str)
    log(f"[aggregate] present={len(present)}/{len(expected)}  missing={missing}")
    log("[aggregate] wrote summary/run_manifest.json")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SLURM_ARRAY_TASK_ID")
    if arg is None:
        raise SystemExit("usage: python novae_INDEPENDENT_FINAL.py <task_id | aggregate>")
    if str(arg).strip().lower() == "aggregate":
        aggregate()
    else:
        process_one(int(arg))
