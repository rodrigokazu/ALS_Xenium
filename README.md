# Cooper-Knock Spatial Pipeline

Analysis code for the ALS spinal cord Xenium cohort (University of Sheffield, Cooper-Knock lab, with
Stanford): 20 sections, 10 ALS donors and 10 controls, 480-gene panel. The pipeline lives in
`ck_spatial_pipeline/`, one folder per tool, in the order the tools ran.

- **Novae** (`1_novae/`) trains a fresh model on each section to get spatial domains at 4, 6, 8
  and 10 levels. The independent fit makes smear show up as its own domain and feeds QC. A joint
  fit shares domains across sections for composition and the imaging overlays.
- **QC** (`2_qc/`) scores every n6 domain for smear (KEEP, REVIEW or REMOVE), then runs 20 patched
  per-section notebooks: a depth cut of 10 transcripts on raw counts, a cut at 8 neighbours within
  100 µm, and removal of REMOVE domains. The cohort goes from 1,694,362 to 1,097,669 cells.
- **Cell annotation** (`3_cell_annotation/`) layers marker-score cell types, the ventral-horn v2
  motor-neuron call (`is_MN_v2 = is_neuron_v3 & ~is_GAD & in_VH_v2`, 844 cells) and an audit of
  Other neurons that moves 48,063 cells into real classes, adding Fibroblast (`cell_type_v2`).
- **NicheCompass** (`4_nichecompass/`) learns gene-programme niches (77 programmes survive the
  panel). After the audits it drops niches 3, 7 and 8 plus three junk sub-clusters and retrains on
  965,285 cells, giving 7 niches (`niche_v2`) in the canonical object
  `NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad`.
- **COZI** (`5_cozi/`) scores directional cell-type neighbour preference per section against 300
  permutations and compares ALS with control. On the whole cord 13 of 121 cell-type pairs differ,
  and inside the ventral horn none do. Niche DE ships alongside it.

How to use this repo:

- The scripts carry absolute SCG paths and run from flat folders on the cluster, where they import
  each other by bare name. `ck_spatial_pipeline/deploy/deploy.sh` copies each file to its SCG path
  from `manifest.tsv`, and a dry run flags any file changed on SCG since the last pull.
- Every script opens with a commentary header covering what it does and what has gone wrong with it
  before. `ck_spatial_pipeline/README.md` gives the dated run order, the canonical outputs and the
  known traps, and `archive/` keeps retired code.
- Spinal level confounds disease: all ten controls are cervical and eight of ten ALS cases are
  lumbar. Treat every ALS against control contrast with that in mind.
