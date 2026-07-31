# ========================================================================================
# ALS_Xenium repo commentary | superseded/hybrid_segmentation/Xenium_Ranger_outlineTIFF.py
#
# Part of the hybrid segmentation experiment. That whole line of work was abandoned once
# Marcel's Gaussian and then _final segmentations arrived.
#
# Reads cell_boundaries.parquet and rasterises the polygon perimeters into a TIFF so outlines
# can be inspected against the morphology images or loaded into external tools. Uses
# skimage.draw.polygon_perimeter.
#
# Written to run on a Windows workstation against a Dropbox path for one hard-coded sample
# (SD01915_BG). It will not run anywhere else without editing BASE. Historical.
# ========================================================================================

# Getting cell outlines from Xenium Ranger resegmentation of large cells
# Written by Dr Rodrigo Kazu @ JCK lab - University of Sheffield

import os
import pandas as pd
import numpy as np
import tifffile
from skimage.draw import polygon_perimeter

BASE = os.path.join(
    r"C:\Users\md1rss\Dropbox\NeuralPathways\Current_projects",
    r"Cooper-Knock_lab\Analysis_2025\Spatial",
    r"ALS2_SD01915_BG__largecells_reseg_20251215_064234_job49421202_4",
    r"outs"
)

cell_boundaries_path = os.path.join(BASE, "cell_boundaries.parquet")
output_tif = os.path.join(BASE, "ranger_cells_edges.tif")

cells = pd.read_parquet(cell_boundaries_path)

H = int(cells["vertex_y"].max()) + 5
W = int(cells["vertex_x"].max()) + 5

edges = np.zeros((H, W), dtype=np.uint8)

for _, grp in cells.groupby("cell_id"):
    rr, cc = polygon_perimeter(
        grp["vertex_y"].values.astype(int),
        grp["vertex_x"].values.astype(int),
        shape=edges.shape,
        clip=True
    )
    edges[rr, cc] = 255  # white edges

tifffile.imwrite(output_tif, edges)
print(f"Wrote edge-only outline to: {output_tif}")
