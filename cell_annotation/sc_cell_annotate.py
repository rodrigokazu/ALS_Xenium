#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/sc_cell_annotate.py
#
# Asks Claude to name the Leiden clusters from their marker genes, via cell-annotator 0.3.0.
#
# What leaves the machine is a per-cluster list of marker gene symbols and nothing else. No
# cell-level data, no patient information, no counts. Say so plainly if anyone asks.
#
# The key is read from the environment or a .env file through python-dotenv and is never
# hardcoded. The script fails loudly if ANTHROPIC_API_KEY is absent instead of falling back to
# anything.
#
# One configuration trap with real cost. sample_key makes cell-annotator loop per sample, so
# setting it to the sample column turns a handful of API calls into roughly twenty times as
# many. Use a single global key.
#
# --dry-run prints the marker lists and stops before any call. Use it to see what would be
# sent.
#
# The labels are first-pass hypotheses to validate against markers and the Novae domains, not
# ground truth. Motor neurons are about 0.6 percent of cells and will not form their own
# Leiden cluster, so this will not call them; is_MN remains authoritative and the sensible
# check is a crosstab of is_MN against leiden afterwards.
# ========================================================================================

"""
sc_cell_annotate.py

First-pass LLM cell-type annotation of the clustered ALS spinal-cord Xenium object,
using cell-annotator 0.3.0 (https://github.com/quadbio/cell-annotator).

cell-annotator does NOT cluster. It computes per-cluster marker genes then asks an LLM
to label each existing Leiden cluster. Run sc_cluster.py first to produce the clustered,
log-normalized .h5ad this script consumes.

Backend = Anthropic/Claude (provider='anthropic'). The API key is read from the
environment / a .env file via python-dotenv. The key is NEVER hardcoded; the script
FAILS LOUDLY if ANTHROPIC_API_KEY is absent.

Only external egress = per-cluster marker-gene lists + the expected-cell-types prompt
sent to Anthropic (one call per cluster + one expected-types call). No cell-level data
leaves the machine.

Outputs:
  - obs['cell_type_llm'] written onto the object (first-pass hypothesis layer; is_MN
    stays authoritative for motor neurons)
  - per-cluster annotation CSV (cluster, cell_type, confidence, top markers, n_cells)
  - expected-cell-types provenance CSV
  - annotated .h5ad (NEW file; never overwrites the pre-QC input)
  - sanity crosstabs: is_MN x leiden, is_MN x cell_type_llm, and (if present) Novae domain

Usage:
  python sc_cell_annotate.py --input <clustered.h5ad> --out <annotated.h5ad> \
      [--cluster-key leiden] [--sample-key none] [--model claude-haiku-4-5] \
      [--csv <table.csv>] [--dry-run]

--dry-run stops BEFORE any LLM call and just prints the clusters + top markers it WOULD send.
"""

import argparse
import json
import os
import sys
from datetime import date


def log(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


DEFAULT_CSV_DIR = "/home/rodrigok/SLURM_jobs/Spatial/CellAnnot"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Clustered .h5ad from sc_cluster.py (X = log-normalized).")
    p.add_argument("--out", default=None, help="Output annotated .h5ad (NEW file). Default: alongside input.")
    p.add_argument("--cluster-key", default="leiden", help="obs key with cluster labels. Default 'leiden'.")
    p.add_argument("--sample-key", default="none",
                   help="obs key for per-sample annotation. Default 'none' = GLOBAL first pass: "
                        "clusters are annotated ONCE across all cells (~n_clusters + 1 LLM calls). "
                        "This is what the full 1.88M-cell run wants. Passing a real key (e.g. "
                        "'sample') runs cell-annotator PER sample (one SampleAnnotator per unique "
                        "value): with 20 samples that is ~20x the LLM calls/cost AND runs "
                        "rank_genes_groups on per-sample subsets while the Leiden categorical "
                        "still carries all global clusters, so clusters with 0-1 cells in a "
                        "sample (guaranteed under the spinal-level x disease confound) can error "
                        "or yield empty markers mid-job. Use per-sample ONLY for genuinely "
                        "independent per-sample annotation.")
    p.add_argument("--species", default="human")
    p.add_argument("--tissue", default="spinal cord")
    p.add_argument("--stage", default="adult")
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-haiku-4-5",
                   help="Anthropic model string. Default claude-haiku-4-5 (cheap). "
                        "Upgrade to a Sonnet-class model for higher-quality labels.")
    p.add_argument("--key-added", default="cell_type_llm",
                   help="obs key to write LLM labels to. Default 'cell_type_llm'.")
    p.add_argument("--csv", default=None,
                   help=f"Per-cluster annotation CSV path. Default under {DEFAULT_CSV_DIR}.")
    p.add_argument("--n-markers", type=int, default=5, help="Expected markers per cell type. Default 5.")
    p.add_argument("--max-markers", type=int, default=7, help="Max DE markers per cluster. Default 7.")
    p.add_argument("--dry-run", action="store_true",
                   help="Stop before any LLM call; print clusters + top markers that WOULD be sent.")
    return p.parse_args()


def load_key_or_die(provider):
    """Load .env then require the provider's API key. Fail loudly if absent."""
    try:
        from dotenv import load_dotenv
        env_file = os.environ.get("CELL_ANNOTATOR_ENV_FILE", os.path.expanduser("~/.env"))
        loaded = load_dotenv(dotenv_path=env_file)
        log(f"[key] load_dotenv({env_file}) -> {loaded}")
    except Exception as e:
        log(f"[key] python-dotenv unavailable ({e}); relying on shell environment only.")

    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "ANTHROPIC_API_KEY")

    key = os.environ.get(env_var)
    if not key:
        sys.exit(
            f"FATAL: {env_var} not set. Export it or add it to your .env "
            f"(or CELL_ANNOTATOR_ENV_FILE) before running cell-annotator with provider='{provider}'."
        )
    log(f"[key] {env_var} found (len={len(key)}); not printed.")
    return env_var


def print_dry_run_markers(adata, cluster_key, max_markers):
    """Compute per-cluster top markers with scanpy (NO LLM, NO egress) and print them."""
    import scanpy as sc
    import pandas as pd  # noqa: F401

    log(f"[dry-run] computing rank_genes_groups on '{cluster_key}' (wilcoxon) -- no LLM, no egress ...")
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon", use_raw=False)
    groups = adata.obs[cluster_key].cat.categories if hasattr(adata.obs[cluster_key], "cat") \
        else sorted(adata.obs[cluster_key].unique())
    names = adata.uns["rank_genes_groups"]["names"]
    sizes = adata.obs[cluster_key].value_counts()
    log("[dry-run] clusters + top markers cell-annotator WOULD send to the LLM:")
    for g in groups:
        try:
            top = [names[str(g)][i] for i in range(min(max_markers, len(names[str(g)])))]
        except Exception:
            top = list(names[g][:max_markers])
        log(f"  cluster {g} (n={int(sizes.get(g, 0)):,}): {', '.join(map(str, top))}")
    log("[dry-run] STOP — no LLM call made. Re-run without --dry-run to annotate.")


def main():
    args = parse_args()

    log("[import] loading scanpy / anndata / numpy / pandas ...")
    import numpy as np  # noqa: F401
    import pandas as pd
    import scanpy as sc

    log(f"[env] scanpy {sc.__version__}")

    log(f"[load] reading {args.input}")
    adata = sc.read_h5ad(args.input)
    log(f"[load] {adata.n_obs:,} cells x {adata.n_vars} genes")

    if args.cluster_key not in adata.obs:
        sys.exit(f"FATAL: cluster key '{args.cluster_key}' not in obs. "
                 f"Run sc_cluster.py first. Available: {list(adata.obs.columns)}")

    # Ensure cluster key is categorical (cell-annotator + rank_genes_groups expect this).
    if not hasattr(adata.obs[args.cluster_key], "cat"):
        adata.obs[args.cluster_key] = adata.obs[args.cluster_key].astype("category")

    # --------------------------------------------------------------- DRY RUN
    if args.dry_run:
        print_dry_run_markers(adata, args.cluster_key, args.max_markers)
        return

    # ----------------------------------------------------------- KEY / IMPORT
    load_key_or_die(args.provider)

    try:
        from cell_annotator import CellAnnotator
    except Exception as e:
        sys.exit(f"FATAL: cannot import cell_annotator ({e}). Is cell-annotator installed in this env?")

    sample_key = None if str(args.sample_key).lower() in ("none", "", "null") else args.sample_key
    if sample_key is not None and sample_key not in adata.obs:
        sys.exit(f"FATAL: sample key '{sample_key}' not in obs. Pass --sample-key none for single-sample mode.")

    log(f"[cell-annotator] CellAnnotator(species={args.species!r}, tissue={args.tissue!r}, "
        f"cluster_key={args.cluster_key!r}, sample_key={sample_key!r}, "
        f"provider={args.provider!r}, model={args.model!r})")
    ca = CellAnnotator(
        adata,
        species=args.species,
        tissue=args.tissue,
        stage=args.stage,
        cluster_key=args.cluster_key,
        sample_key=sample_key,
        provider=args.provider,
        model=args.model,
        api_key=None,  # read from environment; never hardcoded
    )

    # Cheap key-PRESENCE check for the chosen provider before the full run.
    # NOTE: check_api_access(provider=...) tests env-var presence for that provider, not live
    # connectivity; a revoked/invalid key still passes here and only fails inside the real LLM
    # calls. The CellAnnotator constructor above already validated the Anthropic key, so this is
    # a belt-and-braces presence check rather than a connectivity probe.
    log(f"[cell-annotator] check_api_access(provider={args.provider!r}) ...")
    try:
        ok = ca.check_api_access(provider=args.provider)
        if ok is False:
            sys.exit("FATAL: cell-annotator check_api_access() returned False. "
                     "Check ANTHROPIC_API_KEY / provider / model / connectivity.")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"FATAL: cell-annotator API access check failed: {e}")

    # 1) Expected cell types + markers for the tissue/species (restricted to panel genes).
    log("[cell-annotator] get_expected_cell_type_markers(...) [1 LLM call] ...")
    ca.get_expected_cell_type_markers(
        n_markers=args.n_markers, filter_to_var_names=True, provide_var_names=True
    )

    # 2) Per-cluster differential markers (NO LLM).
    log("[cell-annotator] get_cluster_markers(...) [no LLM] ...")
    ca.get_cluster_markers(
        method="wilcoxon", min_specificity=0.75, min_auc=0.7,
        max_markers=args.max_markers, use_raw=False, use_rapids=False,
    )

    # 3) Label each cluster (one LLM call per cluster). Writes obs[key_added].
    log(f"[cell-annotator] annotate_clusters(key_added={args.key_added!r}) [1 LLM call/cluster] ...")
    ca.annotate_clusters(min_markers=2, restrict_to_expected=False, key_added=args.key_added)

    if args.key_added not in adata.obs:
        # Some versions write onto ca.adata rather than the passed handle.
        if hasattr(ca, "adata") and args.key_added in ca.adata.obs:
            adata.obs[args.key_added] = ca.adata.obs[args.key_added].values
        else:
            sys.exit(f"FATAL: annotate_clusters did not produce obs['{args.key_added}'].")
    log(f"[cell-annotator] wrote obs['{args.key_added}']:")
    log(adata.obs[args.key_added].value_counts(dropna=False).to_string())

    # ------------------------------------------------- PER-CLUSTER CSV TABLE
    today = date.today().strftime("%Y%m%d")
    csv_path = args.csv or os.path.join(DEFAULT_CSV_DIR, f"cluster_annotation_table_{today}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    # Pull annotation_df + marker_dfs from ca.sample_annotators (dict; single entry when sample_key=None).
    frames = []
    marker_lookup = {}
    try:
        sa_map = ca.sample_annotators
        if isinstance(sa_map, dict):
            sa_items = list(sa_map.items())
        else:
            sa_items = [("all", sa_map)]
        for sname, sa in sa_items:
            adf = getattr(sa, "annotation_df", None)
            if adf is not None and len(adf):
                adf = adf.copy()
                # annotation_df is INDEXED by cluster label; reset_index() so the cluster id
                # survives the concat below (ignore_index would otherwise drop it entirely).
                adf = adf.reset_index()
                first = adf.columns[0]  # former index = the cluster label
                if first != "cluster":
                    adf = adf.rename(columns={first: "cluster"})
                adf.insert(0, "sample", sname)
                frames.append(adf)
            mdfs = getattr(sa, "marker_dfs", None)
            if isinstance(mdfs, dict):
                for cl, mdf in mdfs.items():
                    try:
                        genes = list(mdf["gene"])[: args.max_markers] if "gene" in mdf.columns \
                            else list(mdf.index)[: args.max_markers]
                    except Exception:
                        genes = []
                    marker_lookup[str(cl)] = ", ".join(map(str, genes))
    except Exception as e:
        log(f"[csv] WARNING: could not read ca.sample_annotators cleanly ({e}); "
            "falling back to obs-derived table.")

    sizes = adata.obs[args.cluster_key].value_counts()

    if frames:
        # Keep the per-frame columns (incl. the restored 'cluster'); do NOT ignore_index,
        # which would discard the cluster identity we just preserved.
        table = pd.concat(frames).reset_index(drop=True)
        # Attach n_cells + top markers per cluster if a 'cluster' column exists.
        cl_col = "cluster" if "cluster" in table.columns else None
        if cl_col:
            table["n_cells"] = table[cl_col].map(lambda c: int(sizes.get(str(c), sizes.get(c, 0))))
            table["top_markers"] = table[cl_col].map(lambda c: marker_lookup.get(str(c), ""))
    else:
        # Minimal fallback table from obs mapping.
        rows = []
        grp = adata.obs.groupby(args.cluster_key, observed=True)[args.key_added]
        for cl, s in grp:
            lab = s.mode().iloc[0] if len(s.mode()) else ""
            rows.append({
                "cluster": cl, "cell_type": lab,
                "n_cells": int(sizes.get(cl, 0)),
                "top_markers": marker_lookup.get(str(cl), ""),
            })
        table = pd.DataFrame(rows)

    table.to_csv(csv_path, index=False)
    log(f"[csv] wrote per-cluster annotation table -> {csv_path} ({len(table)} rows)")

    # Expected cell types / markers provenance dump.
    prov_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                             f"expected_cell_types_{today}.json")
    try:
        prov = {
            "expected_cell_types": list(getattr(ca, "expected_cell_types", []) or []),
            "expected_marker_genes": dict(getattr(ca, "expected_marker_genes", {}) or {}),
        }
        with open(prov_path, "w") as fh:
            json.dump(prov, fh, indent=2, default=str)
        log(f"[csv] wrote expected-cell-types provenance -> {prov_path}")
    except Exception as e:
        log(f"[csv] WARNING: could not dump expected cell types ({e}).")

    # ----------------------------------------------------------- SANITY CHECKS
    if "is_MN" in adata.obs:
        log("[sanity] is_MN x " + args.cluster_key + " crosstab "
            "(MNs ~0.6%/11.2k expected to smear into the dominant neuron cluster, NOT their own):")
        log(pd.crosstab(adata.obs[args.cluster_key], adata.obs["is_MN"]).to_string())
        log(f"[sanity] is_MN x {args.key_added} crosstab:")
        log(pd.crosstab(adata.obs[args.key_added], adata.obs["is_MN"]).to_string())
        # Auto-surface the cluster hosting the majority of is_MN cells + its LLM label so a
        # red flag (MNs landing in a NON-neuronal cluster) can't hide in a large crosstab.
        try:
            mn = adata.obs["is_MN"].astype(str).str.lower().isin(("true", "1", "yes"))
            n_mn = int(mn.sum())
            if n_mn > 0:
                host = adata.obs.loc[mn, args.cluster_key].value_counts()
                top_cl = host.index[0]
                frac = host.iloc[0] / n_mn
                lab = adata.obs.loc[adata.obs[args.cluster_key] == top_cl, args.key_added]
                host_label = lab.mode().iloc[0] if len(lab.mode()) else "?"
                log(f"[sanity] is_MN host cluster = {top_cl} (captures {frac:.1%} of {n_mn:,} "
                    f"MNs) -> {args.key_added}='{host_label}'")
                if "neuron" in str(host_label).lower():
                    log("[sanity] OK: is_MN host cluster carries a NEURONAL label as expected.")
                else:
                    log(f"[sanity] WARNING: is_MN host cluster labelled '{host_label}', NOT a "
                        "neuronal type -- investigate (MNs are expected to fall in a neuron cluster).")
            else:
                log("[sanity] is_MN present but no positive cells; skipping host-cluster check.")
        except Exception as e:
            log(f"[sanity] MN host-cluster check skipped: {e}")
    else:
        log("[sanity] no is_MN column; skipping MN crosstabs.")

    # Novae domain cross-check if such a column is present.
    novae_cols = [c for c in adata.obs.columns
                  if "novae" in c.lower() or c.lower() in ("domain", "niche", "spatial_domain")]
    if novae_cols:
        nc = novae_cols[0]
        log(f"[sanity] Novae/domain column '{nc}' present -> {args.key_added} x {nc} crosstab "
            "(check cell types track spatial domains: grey vs white matter, ventral horn):")
        log(pd.crosstab(adata.obs[args.key_added], adata.obs[nc]).to_string())
    else:
        log("[sanity] no Novae/domain column found; skipping spatial-domain crosstab.")

    log("[sanity] REMINDER: all cell_type_llm labels are first-pass HYPOTHESES to validate "
        "against markers/domains, not ground truth. is_MN stays authoritative for motor neurons. "
        "Batch caveat: spinal-level x disease confound (controls cervical / ALS lumbar) -> "
        "coarse TYPE labels tolerable, state/disease splits NOT trustworthy from this pass.")

    # -------------------------------------------------------------- PROVENANCE
    adata.uns["cellannotator_provenance"] = {
        "cell_annotator_version": "0.3.0",
        "provider": args.provider,
        "model": args.model,
        "species": args.species,
        "tissue": args.tissue,
        "stage": args.stage,
        "cluster_key": args.cluster_key,
        "sample_key": str(sample_key),
        "key_added": args.key_added,
        "csv": csv_path,
        "date": today,
    }

    # ------------------------------------------------------------------- SAVE
    if args.out:
        out_path = args.out
    else:
        base = os.path.basename(args.input).replace(".h5ad", "")
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                f"{base}_annotated_cellannotator_{today}.h5ad")
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("FATAL: output path equals input path; refusing to overwrite the clustered input.")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    adata.write_h5ad(out_path)
    log(f"[save] wrote annotated .h5ad -> {out_path}")
    log("[done] cell-annotator first-pass annotation complete.")


if __name__ == "__main__":
    main()
