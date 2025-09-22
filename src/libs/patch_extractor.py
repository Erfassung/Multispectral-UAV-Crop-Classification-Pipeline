"""
Patch Extractor Module

This module provides functionality for extracting labeled patches from multispectral 
orthoimages using vector field boundaries.
"""

import os
import glob
import re
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from GeoPatch import TrainPatch
from tqdm import tqdm
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extract labeled patches from multispectral orthoimages using vector field boundaries.
    
    This class handles the entire workflow of:
    1. Loading and encoding vector crop boundaries
    2. Creating crop masks from vector data
    3. Extracting patches using the GeoPatch library
    4. Renaming patches based on majority crop class
    """
    
    def __init__(
        self,
        vector_path: str,
        ortho_dir: str,
        mask_path: Optional[str] = None,
        output_dir: str = "data/processed/patches",
        patch_size: int = 256,
        stride: int = 128,
        channel_first: bool = True,
        min_plots_per_class: int = 3
    ):
        """
        Initialize the PatchExtractor.
        
        Args:
            vector_path: Path to GeoJSON or Shapefile with crop boundaries
            ortho_dir: Directory containing multispectral ortho-TIFFs
            mask_path: Optional pre-computed crop mask file
            output_dir: Output directory for patches
            patch_size: Size of patches in pixels
            stride: Sliding window stride in pixels
            channel_first: Whether to store patches as (C,H,W) or (H,W,C)
            min_plots_per_class: Minimum plots required per crop class
        """
        self.vector_path = vector_path
        self.ortho_dir = ortho_dir
        self.mask_path = mask_path if (mask_path and os.path.isfile(mask_path)) else None
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.stride = stride
        self.channel_first = channel_first
        self.min_plots_per_class = min_plots_per_class
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and encode vector data
        self.gdf = None
        self.crop_mapping = None
        self._load_and_encode_vector()
    
    def _load_and_encode_vector(self) -> None:
        """Load vector data and encode crop types as integer labels."""
        logger.info(f"Loading vector data from {self.vector_path}")
        gdf = gpd.read_file(self.vector_path)
        
        # Remove cross-crop mixtures
        bad = (gdf["crop"].str.contains(r"Mixture", na=False) &
               gdf["crop"].str.contains(r"-", na=False))
        if bad.any():
            logger.info(f"Dropping {bad.sum()} cross-crop mixtures")
            gdf = gdf.loc[~bad].copy()
        
        # Encode crop types as 1-based integer labels
        gdf["crop_label"] = gdf["crop"].astype("category").cat.codes + 1
        
        # Create and save mapping
        self.crop_mapping = (
            gdf[["crop_label", "crop"]]
            .drop_duplicates()
            .sort_values("crop_label")
            .reset_index(drop=True)
        )
        
        mapping_path = os.path.join(self.output_dir, "crop_label_mapping.csv")
        self.crop_mapping.to_csv(mapping_path, index=False)
        logger.info(f"Saved crop label mapping to {mapping_path}")
        
        # Filter classes with too few plots
        counts = gdf["crop_label"].value_counts()
        valid = counts[counts >= self.min_plots_per_class].index.tolist()
        dropped = sorted(set(counts.index) - set(valid))
        if dropped:
            logger.info(f"Excluding crop_labels with fewer than {self.min_plots_per_class} plots: {dropped}")
            gdf = gdf[gdf["crop_label"].isin(valid)].copy()
        
        # Convert LineStrings to Polygons if needed
        gdf["geometry"] = gdf.geometry.map(self._to_polygon)
        self.gdf = gdf
        logger.info(f"Loaded {len(gdf)} valid crop plots")
    
    def _to_polygon(self, geom) -> Polygon:
        """Convert LineString geometries to Polygons if they form closed loops."""
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            if coords[0] == coords[-1]:
                return Polygon(coords)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                pts = list(part.coords)
                if pts[0] == pts[-1]:
                    return Polygon(pts)
        return geom
    
    def rasterize_mask(self, reference_ortho: str, mask_out: str) -> None:
        """
        Create a crop mask by rasterizing vector data.
        
        Args:
            reference_ortho: Path to reference orthoimage for spatial properties
            mask_out: Output path for the rasterized mask
        """
        logger.info(f"Creating crop mask from vector data")
        
        with rasterio.open(reference_ortho) as src:
            crs, transform, width, height, meta = (
                src.crs, src.transform, src.width, src.height, src.meta.copy()
            )
        
        # Reproject vector data if needed
        if self.gdf.crs != crs:
            gdf_reproj = self.gdf.to_crs(crs)
        else:
            gdf_reproj = self.gdf
        
        # Create shapes for rasterization
        shapes = [
            (geom, int(label))
            for geom, label in zip(gdf_reproj.geometry, gdf_reproj["crop_label"])
        ]
        
        # Rasterize
        mask_arr = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        )
        
        # Save mask
        mask_meta = meta.copy()
        mask_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
        with rasterio.open(mask_out, "w", **mask_meta) as dst:
            dst.write(mask_arr, 1)
        
        self.mask_path = mask_out
        logger.info(f"Created crop mask: {mask_out}")
    
    def extract_all_orthos(self) -> None:
        """Extract patches from all orthoimages in the input directory."""
        # Create mask if not provided
        if not self.mask_path:
            ortho_files = sorted(glob.glob(os.path.join(self.ortho_dir, "*.tif")))
            if not ortho_files:
                raise ValueError(f"No .tif files found in {self.ortho_dir}")
            
            first_ortho = ortho_files[0]
            mask_out = os.path.join(self.output_dir, "crop_mask.tif")
            self.rasterize_mask(first_ortho, mask_out)
        
        # Process all orthoimages
        ortho_files = sorted(glob.glob(os.path.join(self.ortho_dir, "*.tif")))
        logger.info(f"Processing {len(ortho_files)} orthoimages")
        
        for ortho_path in tqdm(ortho_files, desc="Extracting patches"):
            self._process_single_ortho(ortho_path)
    
    def _process_single_ortho(self, ortho_path: str) -> None:
        """
        Extract patches from a single orthoimage.
        
        Args:
            ortho_path: Path to the orthoimage file
        """
        name = os.path.splitext(os.path.basename(ortho_path))[0]
        out_dir = os.path.join(self.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Use GeoPatch to extract patches
        patcher = TrainPatch(
            image=ortho_path,
            label=self.mask_path,
            patch_size=self.patch_size,
            stride=self.stride,
            channel_first=self.channel_first
        )
        patcher.data_dimension()
        patcher.patch_info()
        
        # Save patches as GeoTIFFs
        patcher.save_Geotif(folder_name=out_dir, only_label=False)
        
        # Rename patches based on majority class
        self._rename_patches_with_label(out_dir)
        
        logger.info(f"Extracted patches for {name}")
    
    def _rename_patches_with_label(self, patch_folder: str) -> None:
        """
        Rename patch files to include majority crop class.
        
        Args:
            patch_folder: Directory containing patch files
        """
        for fp in glob.glob(os.path.join(patch_folder, "*.tif")):
            with rasterio.open(fp) as src:
                lbl = src.read(1)
            
            # Find majority class (excluding background)
            u, c = np.unique(lbl[lbl > 0], return_counts=True)
            if len(u) == 0:
                continue
            
            maj = u[np.argmax(c)]
            base = os.path.splitext(os.path.basename(fp))[0]
            
            # Only add _classX if not already present
            if not re.search(r"_class\d+$", base):
                new = os.path.join(patch_folder, f"{base}_class{maj}.tif")
                os.rename(fp, new)
    
    def get_crop_mapping(self) -> dict:
        """
        Get mapping from crop labels to crop names.
        
        Returns:
            Dictionary mapping integer labels to crop names
        """
        if self.crop_mapping is None:
            return {}
        return dict(zip(self.crop_mapping["crop_label"], self.crop_mapping["crop"]))


def rename_patches_with_label(patch_folder: str) -> None:
    """
    Standalone function to rename patches with majority label.
    
    Args:
        patch_folder: Directory containing patch files to rename
    """
    extractor = PatchExtractor.__new__(PatchExtractor)  # Create instance without __init__
    extractor._rename_patches_with_label(patch_folder)