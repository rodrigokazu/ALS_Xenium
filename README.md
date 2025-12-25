# Hybrid Segmentation for Xenium Spatial Data

## Summary

This repository implements a **hybrid segmentation workflow** for Xenium spatial transcriptomics data. The goal is to preserve high-quality automated cell segmentation produced by Xenium Ranger while allowing targeted manual addition of cells that are missed by automated methods (e.g., large motor neurons).

The workflow is designed to be minimally invasive, reproducible, and fully compatible with downstream Xenium analysis tools.

---

## Motivation

Automated segmentation performs well for most cell types but can occasionally miss large or morphologically atypical cells. Fully re-segmenting an entire dataset is unnecessary and risks degrading otherwise correct results. This workflow provides a controlled way to **add only missing cells** without altering existing segmentation.

---

## Approach

The strategy is based on three principles:

1. **Preserve all Xenium Ranger segmentations**
2. **Manually annotate only missing cells**
3. **Reintegrate additions into the Xenium ecosystem**

Existing cell boundaries are used as a visual reference during manual annotation. New cells are added downstream and merged with the original segmentation.

---

## Workflow Overview

1. **Start with Xenium Ranger outputs**

   * Automated cell segmentation
   * Morphology image

2. **Generate a reference overlay**

   * Convert existing cell boundaries into a lightweight visual guide
   * Used only for annotation, not analysis

3. **Manual annotation**

   * Missing cells are annotated in QuPath
   * Existing cells are not modified

4. **Reintegration**

   * Manually added cells are merged with original segmentation
   * Data are re-imported using Xenium Ranger tooling

5. **Downstream analysis**

   * Hybrid segmentation can be explored in Xenium Explorer
   * All standard analysis workflows remain supported

---

## Key Features

* Preserves original Xenium Ranger segmentation
* Adds only manually curated cells
* Scales to large Xenium datasets
* Avoids interactive editing of existing boundaries
* Compatible with multiple QuPath versions
* Fully reproducible

---

## Intended Use

This workflow is intended for research applications where:

* Automated segmentation is mostly correct
* A small subset of biologically important cells is missing
* Manual correction is required without disrupting the rest of the dataset

---

## Notes

This workflow augments Xenium Ranger segmentation but does not modify raw imaging data or transcript localisation. All changes are applied downstream in a controlled and reversible manner.
