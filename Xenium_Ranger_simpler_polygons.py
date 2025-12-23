import pandas as pd
import geojson
from shapely.geometry import Polygon

# === INPUT ===
cell_boundaries_path = (
    r"C:\Users\md1rss\Dropbox\NeuralPathways\Current_projects"
    r"\Cooper-Knock_lab\Analysis_2025\Spatial"
    r"\ALS2_SD01915_BG__largecells_reseg_20251215_064234_job49421202_4"
    r"\outs\cell_boundaries.parquet"
)

# === OUTPUT ===
output_geojson = "ranger_cells_simplified.geojson"

# === LOAD ===
cells = pd.read_parquet(cell_boundaries_path)

features = []
dropped = 0

for cid, grp in cells.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))

    # Ensure polygon closure
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)

    # Simplify geometry aggressively (visual reference only)
    poly = poly.simplify(tolerance=2.0, preserve_topology=True)

    if not poly.is_valid or poly.is_empty:
        dropped += 1
        continue

    features.append(
        geojson.Feature(
            geometry=geojson.mapping(poly),
            properties={"cell_id": cid}
        )
    )

fc = geojson.FeatureCollection(features)

with open(output_geojson, "w") as f:
    geojson.dump(fc, f)

print(f"Wrote {len(features)} simplified Ranger polygons")
print(f"Dropped {dropped} invalid polygons")
