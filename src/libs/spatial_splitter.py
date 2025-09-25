"""
Spatial Splitter Module

This module provides functionality for creating spatially-aware train/validation/test
splits to prevent data leakage in geospatial machine learning.
"""

import os
import re
import ast
import pandas as pd
import rasterio
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialSplitter:
    """
    Create spatially-aware train/validation/test splits for geospatial data.
    
    This class uses a zone mask to assign patches to different splits based on
    their geographic location, preventing data leakage between training and testing.
    """
    
    def __init__(
        self,
        mapping_csv: str,
        zone_mask_path: str,
        stacked_folder: str,
        output_csv: Optional[str] = None,
        min_temporal_samples: int = 3,
        excluded_classes: List[int] = [1, 2],
        zone_split_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the SpatialSplitter.
        
        Args:
            mapping_csv: Path to stack mapping CSV file
            zone_mask_path: Path to zone mask raster file
            stacked_folder: Directory containing stacked .npy files
            output_csv: Output path for split mapping (optional)
            min_temporal_samples: Minimum temporal samples required
            excluded_classes: List of class IDs to exclude from splits
            zone_split_mapping: Custom mapping from zone IDs to split names
        """
        self.mapping_csv = mapping_csv
        self.zone_mask_path = zone_mask_path
        self.stacked_folder = stacked_folder
        self.output_csv = output_csv or os.path.join(
            os.path.dirname(mapping_csv), "stack_mapping_zone_split.csv"
        )
        self.min_temporal_samples = min_temporal_samples
        self.excluded_classes = excluded_classes
        
        # Default zone to split mapping
        self.zone_split_mapping = zone_split_mapping or {
            1: "train", 2: "train",
            3: "val",
            4: "test"
        }
        
        # Initialize data storage
        self.df = None
        self.zone_ds = None
        self.zone_arr = None
    
    def load_data(self) -> None:
        """Load mapping CSV and zone mask data."""
        logger.info(f"Loading mapping from {self.mapping_csv}")
        self.df = pd.read_csv(self.mapping_csv)
        
        # Filter by minimum temporal samples
        initial_count = len(self.df)
        self.df = self.df[
            self.df["input_files"].apply(
                lambda s: len(ast.literal_eval(s)) >= self.min_temporal_samples
            )
        ].reset_index(drop=True)
        
        filtered_count = initial_count - len(self.df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} patches with < {self.min_temporal_samples} temporal samples")
        
        # Load zone mask
        logger.info(f"Loading zone mask from {self.zone_mask_path}")
        self.zone_ds = rasterio.open(self.zone_mask_path)
        self.zone_arr = self.zone_ds.read(1)
        
        logger.info(f"Loaded {len(self.df)} patches for spatial splitting")
    
    def assign_zones_and_crops(self) -> None:
        """Assign zone IDs and crop types to each patch based on spatial location."""
        logger.info("Assigning spatial zones and crop types")
        
        zone_ids = []
        crop_types = []
        
        for files_str in tqdm(self.df["input_files"], desc="Assigning zones & crops"):
            zone_id, crop_type = self._process_patch_location(files_str)
            zone_ids.append(zone_id)
            crop_types.append(crop_type)
        
        # Add new columns
        self.df["zone_id"] = zone_ids
        self.df["crop_type"] = crop_types
        
        # Log zone distribution
        zone_counts = pd.Series(zone_ids).value_counts().sort_index()
        logger.info(f"Zone distribution: {zone_counts.to_dict()}")
    
    def _process_patch_location(self, files_str: str) -> Tuple[int, int]:
        """
        Process a single patch to determine its zone and crop type.
        
        Args:
            files_str: String representation of input file list
            
        Returns:
            Tuple of (zone_id, crop_type)
        """
        try:
            file_list = ast.literal_eval(files_str)
            if not file_list:
                return -1, -1
            
            # Use first file for spatial location
            patch_path = file_list[0].replace("\\", os.sep)
            
            # Extract crop type from filename
            crop_type = self._extract_crop_type(patch_path)
            
            # Get spatial location
            zone_id = self._get_zone_from_patch(patch_path)
            
            return zone_id, crop_type
            
        except Exception as e:
            logger.warning(f"Error processing patch location: {e}")
            return -1, -1
    
    def _extract_crop_type(self, patch_path: str) -> int:
        """Extract crop type from patch filename."""
        match = re.search(r"_class(\d+)\.tif$", patch_path)
        return int(match.group(1)) if match else -1
    
    def _get_zone_from_patch(self, patch_path: str) -> int:
        """
        Get zone ID for a patch based on its spatial location.
        
        Args:
            patch_path: Path to patch file
            
        Returns:
            Zone ID from the zone mask
        """
        try:
            with rasterio.open(patch_path) as src:
                width, height = src.width, src.height
                transform = src.transform
            
            # Compute center coordinates in world space
            center_x, center_y = width / 2, height / 2
            world_x, world_y = transform * (center_x, center_y)
            
            # Map to zone mask pixel coordinates
            row, col = self.zone_ds.index(world_x, world_y)
            
            # Check bounds and get zone value
            if 0 <= row < self.zone_arr.shape[0] and 0 <= col < self.zone_arr.shape[1]:
                return int(self.zone_arr[row, col])
            else:
                return -1
                
        except Exception as e:
            logger.warning(f"Error getting zone for {patch_path}: {e}")
            return -1
    
    def apply_filters(self) -> None:
        """Apply filtering based on excluded classes and available files."""
        initial_count = len(self.df)
        
        # Exclude specified classes
        if self.excluded_classes:
            self.df = self.df[~self.df["crop_type"].isin(self.excluded_classes)].reset_index(drop=True)
            excluded_count = initial_count - len(self.df)
            if excluded_count > 0:
                logger.info(f"Excluded {excluded_count} patches from classes {self.excluded_classes}")
        
        # Filter to only include patches with existing .npy files
        if os.path.exists(self.stacked_folder):
            stacked_files = {f for f in os.listdir(self.stacked_folder) if f.endswith(".npy")}
            before_filter = len(self.df)
            self.df = self.df[self.df["output_file"].isin(stacked_files)].reset_index(drop=True)
            missing_count = before_filter - len(self.df)
            if missing_count > 0:
                logger.info(f"Filtered out {missing_count} patches with missing .npy files")
    
    def assign_splits(self, custom_split_function: Optional[Callable[[int], str]] = None) -> None:
        """
        Assign train/validation/test splits based on zone IDs.
        
        Args:
            custom_split_function: Optional custom function to map zone IDs to split names
        """
        logger.info("Assigning spatial splits")
        
        split_function = custom_split_function or self._default_zone_to_split
        self.df["split"] = self.df["zone_id"].apply(split_function)
        
        # Remove excluded patches
        before_exclude = len(self.df)
        self.df = self.df[self.df["split"] != "exclude"].reset_index(drop=True)
        excluded_count = before_exclude - len(self.df)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} patches from invalid zones")
        
        # Log split distribution
        split_counts = self.df["split"].value_counts()
        logger.info(f"Split distribution: {split_counts.to_dict()}")
    
    def _default_zone_to_split(self, zone_id: int) -> str:
        """Default mapping from zone ID to split name."""
        return self.zone_split_mapping.get(zone_id, "exclude")
    
    def save_results(self) -> Tuple[str, str]:
        """
        Save splitting results to CSV files.
        
        Returns:
            Tuple of (full_mapping_path, stacked_split_path)
        """
        # Save full mapping with zone information
        self.df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved full spatial split mapping to {self.output_csv}")
        
        # Save simplified mapping for stacked arrays
        stacked_split_path = os.path.join(self.stacked_folder, "stacked_arrays_split.csv")
        self.df[["output_file", "split"]].to_csv(stacked_split_path, index=False)
        logger.info(f"Saved stacked array splits to {stacked_split_path}")
        
        return self.output_csv, stacked_split_path
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.zone_ds:
            self.zone_ds.close()
    
    def run(self) -> Tuple[str, str]:
        """
        Run the complete spatial splitting pipeline.
        
        Returns:
            Tuple of (full_mapping_path, stacked_split_path)
        """
        try:
            self.load_data()
            self.assign_zones_and_crops()
            self.apply_filters()
            self.assign_splits()
            return self.save_results()
        finally:
            self.cleanup()
    
    def get_split_statistics(self) -> Dict:
        """
        Get detailed statistics about the spatial splits.
        
        Returns:
            Dictionary with split statistics
        """
        if self.df is None:
            return {}
        
        stats = {
            "total_patches": len(self.df),
            "split_distribution": self.df["split"].value_counts().to_dict(),
            "class_distribution_by_split": {},
            "zone_distribution_by_split": {}
        }
        
        # Class distribution by split
        for split in self.df["split"].unique():
            split_df = self.df[self.df["split"] == split]
            stats["class_distribution_by_split"][split] = split_df["crop_type"].value_counts().to_dict()
            stats["zone_distribution_by_split"][split] = split_df["zone_id"].value_counts().to_dict()
        
        return stats
    
    def set_custom_zone_mapping(self, zone_mapping: Dict[int, str]) -> None:
        """
        Set custom zone to split mapping.
        
        Args:
            zone_mapping: Dictionary mapping zone IDs to split names
        """
        self.zone_split_mapping = zone_mapping
        logger.info(f"Updated zone mapping: {zone_mapping}")
    
    def get_patches_by_split(self, split_name: str) -> List[str]:
        """
        Get list of patch files for a specific split.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            
        Returns:
            List of output filenames for the specified split
        """
        if self.df is None:
            return []
        
        split_df = self.df[self.df["split"] == split_name]
        return split_df["output_file"].tolist()