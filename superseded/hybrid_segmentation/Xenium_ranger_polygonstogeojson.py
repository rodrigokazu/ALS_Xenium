# ========================================================================================
# ALS_Xenium repo commentary | superseded/hybrid_segmentation/Xenium_ranger_polygonstogeojson.py
#
# Converts Xenium Ranger cell boundary polygons into GeoJSON, one feature per cell, so they
# could be handed to GIS-style tooling or a viewer that speaks GeoJSON.
#
# Thirty lines, one hard-coded input path, no CLI. It did its job during the hybrid
# segmentation work and has no current caller.
#
# Boundary provenance, not boundary format, is what went wrong later: on _final only about 10
# to 17 percent of cells carry a polygon and the ids are in the wrong namespace, so a
# converter is not the missing piece.
# ========================================================================================

# Converting Xenium Ranger polygons to GeoJSON
# Written by Dr Rodrigo Kazu @ JCK lab - University of Sheffield

import pandas as pd
import geojson

cells = pd.read_parquet(
    r"C:\Users\md1rss\Dropbox\NeuralPathways\Current_projects\Cooper-Knock_lab\Analysis_2025\Spatial\ALS2_SD01915_BG__largecells_reseg_20251215_064234_job49421202_4\outs\cell_boundaries.parquet"
)

features = []

for cid, grp in cells.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))

    # Ensure polygon closure
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    features.append(
        geojson.Feature(
            geometry=geojson.Polygon([coords]),
            properties={"cell_id": cid}
        )
    )

fc = geojson.FeatureCollection(features)

with open("ranger_cells.geojson", "w") as f:
    geojson.dump(fc, f, indent=2)

print(f"Wrote {len(features)} Ranger cell polygons")
