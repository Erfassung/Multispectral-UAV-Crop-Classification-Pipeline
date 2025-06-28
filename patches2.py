import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from GeoPatch import TrainPatch
import argparse

class PatchExtractor:
    def __init__(
        self,
        vector_path: str,          # geojson or shapefile with crop column
        ortho_dir: str,            # directory containing orthos
        mask_path: str = None,     # if provided skip rasterization
        output_dir: str = "patches_npy",
        patch_size: int = 128,
        stride: int = 64,
        channel_first: bool = True):

        self.vector_path   = vector_path
        self.ortho_dir     = ortho_dir
        self.output_dir    = output_dir
        self.patch_size    = patch_size
        self.stride        = stride
        self.channel_first= channel_first
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Configuration: {self.patch_size=}, {self.stride=}, {self.channel_first=}")

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
        bad = gdf["crop"].str.contains(r"Mixture", na=False) & gdf["crop"].str.contains(r"-", na=False)
        if bad.any():
            print(f"Dropping {bad.sum()} cross‐crop mixtures:")
            print(gdf.loc[bad, ["plot_ID", "crop"]])
            gdf = gdf.loc[~bad].copy()

        gdf["crop_label"] = gdf["crop"].astype("category").cat.codes + 1 # 1-based index
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

        shapes = [(geom, int(label)) for geom, label in zip(gdf_reproj.geometry, gdf_reproj["crop_label"])]
        mask_arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8", all_touched=True)

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
        patcher = TrainPatch(
            image=ortho_path,
            label=self.mask_path,
            patch_size=self.patch_size,
            stride=self.stride,
            channel_first=self.channel_first
        )
        patcher.data_dimension()
        patcher.patch_info()
        patcher.save_Geotif(folder_name=out_dir, only_label=True)
        rename_patches_with_label(out_dir)
        print(f"Saved and renamed GeoTIFF patches for '{name}' in '{out_dir}'")

def rename_patches_with_label(patch_folder):
    patch_files = glob.glob(os.path.join(patch_folder, "*.tif"))
    for patch_fp in patch_files:
        with rasterio.open(patch_fp) as src:
            label_patch = src.read(1)  # read first band
            unique, counts = np.unique(label_patch[label_patch != 0], return_counts=True)
            if len(unique) == 0:
                continue
            label = unique[np.argmax(counts)]
        base = os.path.splitext(os.path.basename(patch_fp))[0]
        new_fp = os.path.join(patch_folder, f"{base}_{label}.tif")
        os.rename(patch_fp, new_fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled patch GeoTIFFs from multispectral orthos.")
    parser.add_argument("vector_path", help="Path to GeoJSON/shapefile containing a 'crop' column.")
    parser.add_argument("ortho_dir", help="Directory containing aligned multispectral orthos (*.tif).")
    parser.add_argument("--mask_path", default=None, help="Precomputed mask TIFF. If omitted, one mask is rasterized from the vector.")
    parser.add_argument("--output_dir", default="patches_npy", help="Directory where patch .tif files will be saved.")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch height/width in pixels (default: 64).")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride in pixels (default: 32).")
    parser.add_argument("--channel_first", type=lambda x: bool(int(x)), default=True, help="Store patches as (C,H,W) if true, else (H,W,C). Use 1 for True, 0 for False.")

    args = parser.parse_args()
    extractor = PatchExtractor(
        vector_path=args.vector_path,
        ortho_dir=args.ortho_dir,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        channel_first=args.channel_first
    )
    extractor.extract_all_orthos()
