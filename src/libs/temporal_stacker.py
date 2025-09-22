"""
Temporal Stacker Module

This module provides functionality for stacking temporal patches into 4D arrays
for time-series analysis.
"""

import os
import re
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalStacker:
    """
    Stack temporal patches into 4D arrays for time-series crop classification.
    
    This class handles:
    1. Collection of temporal patches from multiple date folders
    2. Temporal alignment and stacking into 4D arrays
    3. Metadata tracking for downstream processing
    """
    
    def __init__(
        self,
        base_dir: str = "data/processed/patches",
        output_dir: str = "data/processed/stacked",
        patch_subdir: str = "patch",
        date_pattern: str = r"\d{6}_reflectance_ortho",
        expected_bands: int = 10,
        min_temporal_samples: int = 3
    ):
        """
        Initialize the TemporalStacker.
        
        Args:
            base_dir: Base directory containing date-organized patches
            output_dir: Output directory for stacked arrays
            patch_subdir: Subdirectory name within each date folder
            date_pattern: Regex pattern to match date folders
            expected_bands: Expected number of spectral bands
            min_temporal_samples: Minimum temporal samples required for stacking
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.patch_subdir = patch_subdir
        self.date_pattern = date_pattern
        self.expected_bands = expected_bands
        self.min_temporal_samples = min_temporal_samples
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.patch_dict = {}
        self.mapping = []
    
    def collect_patches(self) -> None:
        """Collect patches from all date folders and organize by patch ID."""
        logger.info(f"Collecting patches from {self.base_dir}")
        
        # Find all date folders
        folders = self._find_date_folders()
        logger.info(f"Found {len(folders)} date folders")
        
        # Build patch dictionary: patch_id → list of {date, path}
        for folder in tqdm(folders, desc="Collecting patches"):
            date = self._extract_date_from_folder(folder)
            patch_folder = os.path.join(self.base_dir, folder, self.patch_subdir)
            
            if not os.path.isdir(patch_folder):
                logger.warning(f"Patch folder not found: {patch_folder}")
                continue
            
            for fname in os.listdir(patch_folder):
                if not fname.endswith(".tif"):
                    continue
                
                patch_id = self._extract_patch_id(fname)
                full_path = os.path.join(patch_folder, fname)
                
                self.patch_dict.setdefault(patch_id, []).append({
                    "date": date,
                    "path": full_path,
                    "filename": fname
                })
        
        logger.info(f"Collected {len(self.patch_dict)} unique patches")
    
    def _find_date_folders(self) -> List[str]:
        """Find all folders matching the date pattern."""
        if not os.path.exists(self.base_dir):
            raise ValueError(f"Base directory does not exist: {self.base_dir}")
        
        folders = [
            f for f in os.listdir(self.base_dir)
            if re.match(self.date_pattern, f) and 
               os.path.isdir(os.path.join(self.base_dir, f))
        ]
        return sorted(folders)
    
    def _extract_date_from_folder(self, folder: str) -> datetime:
        """Extract date from folder name."""
        match = re.match(r"(\d{6})_reflectance_ortho", folder)
        if not match:
            raise ValueError(f"Cannot extract date from folder: {folder}")
        return datetime.strptime(match.group(1), "%y%m%d")
    
    def _extract_patch_id(self, filename: str) -> str:
        """Extract patch ID from filename."""
        return filename.split("_")[0]
    
    def _extract_class_label(self, filename: str) -> Optional[str]:
        """Extract class label from filename."""
        match = re.search(r"class(\d+)", filename)
        return match.group(1) if match else None
    
    def stack_patches(self) -> None:
        """Stack temporal patches into 4D arrays and save as .npy files."""
        logger.info("Starting temporal stacking")
        
        processed_count = 0
        skipped_count = 0
        
        for idx, (patch_id, entries) in enumerate(
            tqdm(self.patch_dict.items(), desc="Stacking patches"), 1
        ):
            try:
                stacked_array, metadata = self._stack_single_patch(patch_id, entries)
                
                if stacked_array is None:
                    skipped_count += 1
                    continue
                
                # Save array and update mapping
                output_filename = self._save_stacked_array(
                    stacked_array, patch_id, metadata["class_label"]
                )
                
                self.mapping.append({
                    "output_file": output_filename,
                    "class": int(metadata["class_label"]) if metadata["class_label"] else None,
                    "patch_id": patch_id,
                    "dates": [d.strftime("%Y-%m-%d") for d in metadata["dates"]],
                    "input_files": metadata["input_files"],
                    "shape": list(stacked_array.shape),
                    "temporal_samples": len(metadata["dates"])
                })
                
                processed_count += 1
                
                if idx % 100 == 0:
                    logger.info(f"Processed {processed_count}/{idx} patches")
                    
            except Exception as e:
                logger.error(f"Error processing patch {patch_id}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Stacking complete: {processed_count} processed, {skipped_count} skipped")
    
    def _stack_single_patch(self, patch_id: str, entries: List[Dict]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Stack temporal data for a single patch.
        
        Args:
            patch_id: Unique patch identifier
            entries: List of temporal entries for this patch
            
        Returns:
            Tuple of (stacked_array, metadata) or (None, {}) if failed
        """
        # Sort entries by date
        entries = sorted(entries, key=lambda e: e["date"])
        
        # Filter entries with insufficient temporal samples
        if len(entries) < self.min_temporal_samples:
            return None, {}
        
        arrays, dates, files = [], [], []
        class_label = None
        
        for entry in entries:
            try:
                with rasterio.open(entry["path"]) as src:
                    data = src.read()  # (bands, H, W)
                
                # Validate band count
                if data.shape[0] != self.expected_bands:
                    logger.warning(
                        f"{entry['path']} has {data.shape[0]} bands "
                        f"(expected {self.expected_bands}), skipping"
                    )
                    continue
                
                # Convert to (H, W, bands)
                arrays.append(data.transpose(1, 2, 0))
                dates.append(entry["date"])
                files.append(entry["path"])
                
                # Extract class label from first valid file
                if class_label is None:
                    class_label = self._extract_class_label(entry["filename"])
                
            except Exception as e:
                logger.warning(f"Failed to read {entry['path']}: {e}")
                continue
        
        if not arrays:
            return None, {}
        
        # Skip background class
        if class_label == "0":
            return None, {}
        
        # Stack to (times, H, W, bands)
        arr = np.stack(arrays, axis=0)
        t, h, w, b = arr.shape
        
        # Reshape to (times*bands, H, W) for compatibility
        stacked = arr.transpose(0, 3, 1, 2).reshape(t * b, h, w)
        
        metadata = {
            "class_label": class_label,
            "dates": dates,
            "input_files": files,
            "original_shape": arr.shape
        }
        
        return stacked, metadata
    
    def _save_stacked_array(self, array: np.ndarray, patch_id: str, class_label: Optional[str]) -> str:
        """
        Save stacked array to file.
        
        Args:
            array: Stacked array to save
            patch_id: Patch identifier
            class_label: Class label for naming
            
        Returns:
            Output filename
        """
        if class_label:
            output_filename = f"{patch_id}_class{class_label}.npy"
        else:
            output_filename = f"{patch_id}.npy"
        
        output_path = os.path.join(self.output_dir, output_filename)
        np.save(output_path, array)
        
        return output_filename
    
    def save_mapping(self) -> str:
        """
        Save mapping CSV file.
        
        Returns:
            Path to saved mapping file
        """
        mapping_df = pd.DataFrame(self.mapping)
        csv_path = os.path.join(self.output_dir, "stack_mapping.csv")
        mapping_df.to_csv(csv_path, index=False)
        
        logger.info(f"Mapping saved to {csv_path}")
        return csv_path
    
    def run(self) -> str:
        """
        Run the complete temporal stacking pipeline.
        
        Returns:
            Path to mapping CSV file
        """
        self.collect_patches()
        self.stack_patches()
        return self.save_mapping()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the stacking process.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.mapping:
            return {"total_patches": 0}
        
        df = pd.DataFrame(self.mapping)
        
        stats = {
            "total_patches": len(df),
            "unique_classes": df["class"].nunique(),
            "class_distribution": df["class"].value_counts().to_dict(),
            "temporal_samples_stats": {
                "min": df["temporal_samples"].min(),
                "max": df["temporal_samples"].max(),
                "mean": df["temporal_samples"].mean(),
                "median": df["temporal_samples"].median()
            }
        }
        
        return stats