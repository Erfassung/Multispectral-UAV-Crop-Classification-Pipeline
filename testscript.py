import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiLineString

# 1. Paths
ORTHO_GLOB     = "data/raw/230607_reflectance_ortho.tif"
VECTOR_GEOJSON = "data/shapefiles/final_shape.geojson"
OUT_MASK       = "data/shapefiles/crop_mask_aligned.tif"

# 2. Read vector and convert any closed LineStrings → Polygons
gdf = gpd.read_file(VECTOR_GEOJSON)
def to_poly(g):
    if g.geom_type == "LineString":
        return Polygon(g.coords)
    if g.geom_type == "MultiLineString":
        for part in g.geoms:
            if part.coords[0] == part.coords[-1]:
                return Polygon(part.coords)
    return g
gdf["geometry"] = gdf.geometry.map(to_poly)

# 3. Reproject vector to match ortho CRS (if needed)
with rasterio.open(ORTHO_GLOB) as src:
    ortho_crs   = src.crs
    ortho_transform = src.transform
    ortho_w, ortho_h = src.width, src.height

if gdf.crs != ortho_crs:
    gdf = gdf.to_crs(ortho_crs)

# 4. Prepare shapes (skip background=8 if you like)
shapes = [
    (geom, int(lbl))
    for geom, lbl in zip(gdf.geometry, gdf.crop_label)
    if lbl != 8
]

# 5. Rasterize onto the ortho’s grid
mask_arr = rasterize(
    shapes,
    out_shape=(ortho_h, ortho_w),
    transform=ortho_transform,
    fill=0,           # background label = 0
    dtype="uint8",
    all_touched=True  # fill every pixel touched by the polygon
)

# 6. Save the mask with identical metadata to the ortho
meta = src.meta.copy()
meta.update({
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})
with rasterio.open(OUT_MASK, "w", **meta) as dst:
    dst.write(mask_arr, 1)

print("Written aligned mask to", OUT_MASK)
