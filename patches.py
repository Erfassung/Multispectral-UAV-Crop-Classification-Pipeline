import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from GeoPatch import TrainPatch
import argparse
#https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html #Reading/Writing Vector Files
#https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html #Reprojecting
#https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.buffer.html #Geometry Operations #Minkowski sum is just sum of euclidean distances like always.
#https://rasterio.readthedocs.io/en/latest/api/rasterio.html#rasterio.open #Opening & Inspecting Rasters
#https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize #Rasterizing Vector Geometries
#https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetWriter.write # Writing to a New GeoTIFF
#https://rasterio.readthedocs.io/en/latest/api/rasterio.windows.html#rasterio.windows.transform # Handling Windows & Transforms
"""
To check if the created mask is okay:
In QGIS, load your orthophoto (e.g. 230607_reflectance_ortho.tif).
Then load the single-band mask (e.g. data/shapefiles/crop_mask.tif) that was generated.
In the Layers panel, drop the mask layer on top of the ortho, set its styling to “Singleband pseudocolor” (or just render it as a semi-transparent polygon), and reduce its opacity (to ~50 %)
Zoom around if the colored mask pixels line up exactly with the crop boundaries on the ortho its okay.
If not you may need to check/change the coordinate system of the shapefiles provided in the metadata from EPSG:25832 to EPSG:32632 and export as geojson.

"""
"""
#example call:
# extractor = PatchExtractor(
#     vector_path="data/shapefiles/fieldshape.geojson",
#     ortho_dir="data/raw",
#     output_dir="patches_npy",
#     patch_size=256,
#     stride=128,
#     channel_first=True
# )
# extractor.extract_all_orthos()
# can just provide 2 args: shape and ortho directory rest will be default"""

class PatchExtractor:
    def __init__(
        self,
        vector_path: str,          # geojson or shapefile with crop column
        ortho_dir: str,            # directory containing orthos
        mask_path: str = None,     # if provided skip rasterization
        output_dir: str = "patches_npy",
        patch_size: int = 256,
        stride: int = 128,
        channel_first: bool = True):

        self.vector_path   = vector_path
        self.ortho_dir     = ortho_dir
        self.output_dir    = output_dir
        self.patch_size    = patch_size
        self.stride        = stride
        self.channel_first= channel_first
        os.makedirs(self.output_dir, exist_ok=True)

        # load and encode vector
        self._load_and_encode_vector()

        # if mask exists skip rasterization
        if mask_path and os.path.isfile(mask_path):
            self.mask_path = mask_path
        else:
            self.mask_path = None

    def _load_and_encode_vector(self):
        gdf = gpd.read_file(self.vector_path)
        # encode crop with pandas categorical
        bad = gdf["crop"].str.contains(r"Mixture", na=False) #exclude my fav polygons because for classifcation mixed crop polygons are useless for patch wise classification 
        if bad.any():
            print(f"Dropping {bad.sum()} cross‐crop mixtures:")
            print(gdf.loc[bad, ["plot_ID", "crop"]])
            gdf = gdf.loc[~bad].copy()

        gdf["crop_label"] = gdf["crop"].astype("category").cat.codes

        def to_poly(g):
            if g.geom_type == "LineString":
                coords = list(g.coords)
                if coords[0] == coords[-1]:
                    return Polygon(coords)   
                return g                   
            if g.geom_type == "MultiLineString":
                for part in g.geoms:
                    part_coords = list(part.coords)
                    if part_coords[0] == part_coords[-1]:
                        return Polygon(part_coords)
            return g

        # use to_poly()
        gdf["geometry"] = gdf.geometry.map(to_poly)
        self.gdf = gdf

    def rasterize_mask(self, reference_ortho: str, mask_out: str):
        # use first ortho as reference for crs transform
        with rasterio.open(reference_ortho) as src:
            crs, transform, width, height, meta = (src.crs, src.transform, src.width, src.height, src.meta.copy())

        # reproject vector if needed
        if self.gdf.crs != crs:
            gdf_reproj = self.gdf.to_crs(crs)
        else:
            gdf_reproj = self.gdf

        # build (geometry, value) list
        shapes = [(geom, int(label)) for geom, label in zip(gdf_reproj.geometry, gdf_reproj["crop_label"])]

        # rasterize into single band mask
        mask_arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8", all_touched=True,)

        # save mask
        mask_meta = meta.copy()
        mask_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
        with rasterio.open(mask_out, "w", **mask_meta) as dst:
            dst.write(mask_arr, 1)
        self.mask_path = mask_out
        print(f"created mask: '{mask_out}'")

    def extract_all_orthos(self):
        if not self.mask_path:
            # rasterize once using the first ortho as reference (can be changed to make a mask for each ortho)
            first_ortho = sorted(glob.glob(os.path.join(self.ortho_dir, "*.tif")))[0]
            mask_out = os.path.join(os.path.dirname(self.vector_path), "crop_mask.tif")
            self.rasterize_mask(first_ortho, mask_out)

        ortho_files = sorted(glob.glob(os.path.join(self.ortho_dir, "*.tif")))
        for ortho_path in ortho_files:
            self._process_single_ortho(ortho_path)


    def _process_single_ortho(self, ortho_path: str):
        name = os.path.splitext(os.path.basename(ortho_path))[0]
        out_dir = os.path.join(self.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        patcher = TrainPatch(image=ortho_path, label=self.mask_path, patch_size=self.patch_size, stride=self.stride, channel_first=self.channel_first)
        patcher.data_dimension()
        patcher.patch_info()
    

        img_stack, mask_stack = patcher.save_numpy(folder_name="", only_label=False, return_stacked=True, save_stack=False, V_flip=False, H_flip=False, Rotation=False) # here the patches are flipped and rotated so each patch is multiplyed by 4 for more training data 

        for i in range(img_stack.shape[0]):
            img_patch = img_stack[i]
            mask_patch = mask_stack[i]
            pixel_values, count_pixel = np.unique(mask_patch, return_counts=True)
            nonzero = pixel_values != 0
            pixel_values, count_pixel = pixel_values[nonzero], count_pixel[nonzero]
            if pixel_values.size == 0:
                continue

            label = int(pixel_values[np.argmax(count_pixel)])
            file_name = f"{i:05d}_{label}.npy" #5 digits for sorting
            out_fp     = os.path.join(out_dir, file_name)
            np.savez_compressed(out_fp, image=img_patch, label=label)

        print(f"Saved patches for '{name}' in '{out_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Generate labeled patch arrays from multispectral orthos.")
    parser.add_argument("vector_path", help="Path to GeoJSON/shapefile containing a 'crop' column.")
    parser.add_argument("ortho_dir", help="Directory containing aligned multispectral orthos (*.tif).")
    parser.add_argument("--mask_path", default=None, help="Precomputed mask TIFF. If omitted, one mask is rasterized from the vector.")
    parser.add_argument("--output_dir", default="patches_npy", help="Directory where patch .npy files will be saved.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch height/width in pixels (default: 256).")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride in pixels (default: 128).")
    parser.add_argument("--channel_first", type=lambda x: bool(int(x)),default=True, help="Store patches as (C,H,W) if true, else (H,W,C). Use 1 for True, 0 for False.")

    args = parser.parse_args()
    extractor = PatchExtractor(vector_path=args.vector_path, ortho_dir=args.ortho_dir, mask_path=args.mask_path, output_dir=args.output_dir, patch_size=args.patch_size, stride=args.stride, channel_first=args.channel_first)
    extractor.extract_all_orthos()