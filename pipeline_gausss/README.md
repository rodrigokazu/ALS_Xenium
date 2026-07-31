# pipeline_gausss

The generation before `pipeline_final`, built on Marcel's Gaussian re-segmentation
(`Ranger_procd_mw_gausss`). Only the parts with no successor in `pipeline_final` are here.
Anything that was ported was left out, because shipping two copies of the same logic is how
the wrong one gets run.

So everything in this folder is still the only copy of what it does.

Source on SCG: `/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/`.

## Domain characterisation

`novae_downstream.py` and its runner. Differential expression between domains, spatially
variable genes, PAGA, batch-corrected UMAP, and a pathway score heatmap over 19 curated gene
sets. This is what turns a domain map into a statement about biology. Nothing in
`pipeline_final` replaces it yet; point it at the `_FINAL` niche object when porting.

## Ventral horn sub-domains, a three-stage chain

Run in order, chained with `--dependency=afterok`.

1. `subset_domains_stage1.py`: keep D1014 and D1018, the two domains covering the ventral
   horn, and write a clean subset. Came out at 537,469 cells.
2. `novae_resubset_stage2.py`: train a fresh Novae model on just those cells to resolve
   finer structure. Uses a hard 100 um radius cap on the Delaunay graph instead of a
   percentile prune, because on a gappy subset a percentile keeps edges that bridge the gaps.
3. `downstream_subdomains_stage3.py`: characterise the sub-domains.

It found five distinguishable sub-niches inside two parent domains: synaptic and motor
neuron, myelin-leaning grey matter, reactive astrocyte, vascular, and microglial. The
by-status comparison is doubly confounded here and should not be used for an
ALS-versus-control claim.

## Niche-number sweep reports

`persample_report/` holds `ps_sweep.py` and its cohort and audio roll-ups. These ask how much
the picture changes when you request 4, 6 or 10 domains, and quantify the nesting between
them with contingency tables, adjusted Rand index and purity. Useful when someone asks why a
particular domain count was chosen.

`ps_io.py` is the loader these use. It predates the CSR/CSC fix in
`pipeline_final/ps_io_FINAL_fix.py`, so running these scripts against `_FINAL` output needs
that fix first.

## Re-plotting utilities

`replot_nature.py` and `replot_nature_smallpts.py` regenerate figures from finished runs in
publication style without recomputing anything. Use the small-points version; the first pass
used markers heavy enough to hide the tissue.

## Export helpers

`export_niche_data_joblib.py` sends the coordinates, domain labels and Novae embedding to a
collaborator as arrays. `convert_plots_npy_joblib_tif.py` converts finished PNGs to joblib or
TIFF. Prefer the first one, since it hands over data instead of a picture.

## nature_style.py

A copy of the shared plotting helpers, byte-identical to the one in `pipeline_final`. It lives here because every script in this folder imports it by bare name, resolving against its own directory. Change one, change both.
