"""
Crop Classification Pipeline - Core Libraries

This module provides the core functionality for the crop classification pipeline:
- patch_extractor: Extract labeled patches from multispectral orthoimages
- temporal_stacker: Stack temporal patches into 4D arrays
- spatial_splitter: Create spatially-aware train/validation/test splits
"""

from .patch_extractor import PatchExtractor
from .temporal_stacker import TemporalStacker  
from .spatial_splitter import SpatialSplitter

__version__ = "1.0.0"
__author__ = "Crop Classification Team"

__all__ = [
    "PatchExtractor",
    "TemporalStacker", 
    "SpatialSplitter"
]