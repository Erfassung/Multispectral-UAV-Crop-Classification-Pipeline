import rioxarray                   
import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, MultiLineString
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import os

#ensure output folder exists
os.makedirs("data/shapefiles", exist_ok=True)

#load the geojson
gdf = gpd.read_file("data/shapefiles/md_FieldSHP.geojson")

#LineStrings to Polygons
def to_polygon(geom):
    if geom.geom_type == "LineString":
        return Polygon(geom.coords)
    if geom.geom_type == "MultiLineString":
        #merge closed rings
        for part in geom:
            if part.coords[0] == part.coords[-1]:
                return Polygon(part.coords)
    return geom

gdf["geometry"] = gdf.geometry.apply(to_polygon)

#filter out the background label
gdf = gdf[gdf["crop_label"] != 8] #here you guys have to check if deeplearning model needs a label for background or not, if yes you need to keep it or change the encoding script!

#prepare rasterization
geoms  = gdf.geometry.values
labels = gdf["crop_label"].astype(int).values
crs    = gdf.crs

#set 3 cm resolution
pixel_size = 0.03

#calc bounds+ output grid size
minx, miny, maxx, maxy = gdf.total_bounds
width  = int(np.ceil((maxx - minx) / pixel_size))
height = int(np.ceil((maxy - miny) / pixel_size))
transform = from_bounds(minx, miny, maxx, maxy, width, height)

#rasterize polygons we use 255 for background 
mask_arr = rasterize(
    ((geom, val) for geom, val in zip(geoms, labels)),
    out_shape=(height, width),
    fill=255,           # background value outside polygons
    dtype="uint8",
    transform=transform
)

#wrap in xarray + attach georef metadata
da = xr.DataArray(mask_arr, dims=("y", "x"), name="crop_mask")
da = da.rio.write_crs(crs, inplace=True)
da = da.rio.write_transform(transform, inplace=True)
da = da.rio.write_nodata(255, inplace=True)  # mark 255 as nodata

#save the final 3cm-resolution mask
output_path = "data/shapefiles/crop_mask.tif"
da.rio.to_raster(output_path, driver="GTiff", dtype=da.dtype)

print(f"Written the mask to: {output_path}")
