#!/usr/bin/env python3
"""
Crop Classification Pipeline - Main Script

This script orchestrates the complete crop classification preprocessing pipeline:
1. Extract labeled patches from multispectral orthoimages
2. Stack temporal patches into 4D arrays
3. Create spatially-aware train/validation/test splits

Usage:
    python main.py --help
    python main.py extract --vector data/raw/fields.geojson --ortho-dir data/raw/orthoimages/
    python main.py stack --patches-dir data/processed/patches
    python main.py split --mapping data/processed/stacked/stack_mapping.csv --zone-mask data/raw/zone_mask.tif
    python main.py pipeline --vector data/raw/fields.geojson --ortho-dir data/raw/orthoimages/ --zone-mask data/raw/zone_mask.tif
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from libs.patch_extractor import PatchExtractor
from libs.temporal_stacker import TemporalStacker
from libs.spatial_splitter import SpatialSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: str = ".") -> None:
    """Create necessary directory structure."""
    directories = [
        "data/raw/orthoimages",
        "data/raw/vectors", 
        "data/processed/patches",
        "data/processed/stacked",
        "output/models",
        "output/predictions",
        "output/visualizations"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


def extract_patches(args) -> str:
    """
    Extract labeled patches from orthoimages.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Path to output directory
    """
    logger.info("Starting patch extraction")
    
    extractor = PatchExtractor(
        vector_path=args.vector,
        ortho_dir=args.ortho_dir,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        channel_first=args.channel_first,
        min_plots_per_class=args.min_plots_per_class
    )
    
    extractor.extract_all_orthos()
    
    # Log statistics
    crop_mapping = extractor.get_crop_mapping()
    logger.info(f"Extraction complete. Found {len(crop_mapping)} crop classes:")
    for label, name in crop_mapping.items():
        logger.info(f"  Class {label}: {name}")
    
    return args.output_dir


def stack_temporal(args) -> str:
    """
    Stack temporal patches into 4D arrays.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Path to mapping CSV file
    """
    logger.info("Starting temporal stacking")
    
    stacker = TemporalStacker(
        base_dir=args.patches_dir,
        output_dir=args.output_dir,
        patch_subdir=args.patch_subdir,
        date_pattern=args.date_pattern,
        expected_bands=args.expected_bands,
        min_temporal_samples=args.min_temporal_samples
    )
    
    mapping_path = stacker.run()
    
    # Log statistics
    stats = stacker.get_statistics()
    logger.info(f"Stacking complete. Statistics:")
    logger.info(f"  Total patches: {stats['total_patches']}")
    
    # Only log additional stats if patches were processed
    if stats['total_patches'] > 0:
        logger.info(f"  Unique classes: {stats['unique_classes']}")
        logger.info(f"  Temporal samples - Min: {stats['temporal_samples_stats']['min']}, "
                   f"Max: {stats['temporal_samples_stats']['max']}, "
                   f"Mean: {stats['temporal_samples_stats']['mean']:.1f}")
    else:
        logger.warning("  No patches were processed. Check min_temporal_samples requirement.")
    
    return mapping_path


def create_spatial_splits(args) -> tuple:
    """
    Create spatially-aware train/validation/test splits.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (full_mapping_path, stacked_split_path)
    """
    logger.info("Starting spatial splitting")
    
    splitter = SpatialSplitter(
        mapping_csv=args.mapping,
        zone_mask_path=args.zone_mask,
        stacked_folder=args.stacked_folder,
        output_csv=args.output_csv,
        min_temporal_samples=args.min_temporal_samples,
        excluded_classes=args.excluded_classes
    )
    
    # Set custom zone mapping if provided
    if args.zone_mapping:
        custom_mapping = {}
        for mapping in args.zone_mapping:
            zone_id, split_name = mapping.split(':')
            custom_mapping[int(zone_id)] = split_name
        splitter.set_custom_zone_mapping(custom_mapping)
    
    full_path, stacked_path = splitter.run()
    
    # Log statistics
    stats = splitter.get_split_statistics()
    logger.info(f"Spatial splitting complete. Statistics:")
    logger.info(f"  Total patches: {stats['total_patches']}")
    for split, count in stats['split_distribution'].items():
        logger.info(f"  {split.capitalize()}: {count} patches")
    
    return full_path, stacked_path


def run_full_pipeline(args) -> None:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Starting full preprocessing pipeline")
    
    # Setup directories
    setup_directories()
    
    # Step 1: Extract patches
    logger.info("=" * 50)
    logger.info("STEP 1: Patch Extraction")
    logger.info("=" * 50)
    
    patch_args = argparse.Namespace(
        vector=args.vector,
        ortho_dir=args.ortho_dir,
        mask_path=args.mask_path,
        output_dir=args.patches_output or "data/processed/patches",
        patch_size=args.patch_size,
        stride=args.stride,
        channel_first=args.channel_first,
        min_plots_per_class=args.min_plots_per_class
    )
    patches_dir = extract_patches(patch_args)
    
    # Step 2: Stack temporal data
    logger.info("=" * 50)
    logger.info("STEP 2: Temporal Stacking")
    logger.info("=" * 50)
    
    stack_args = argparse.Namespace(
        patches_dir=patches_dir,
        output_dir=args.stacked_output or "data/processed/stacked",
        patch_subdir=args.patch_subdir,
        date_pattern=args.date_pattern,
        expected_bands=args.expected_bands,
        min_temporal_samples=args.min_temporal_samples
    )
    mapping_path = stack_temporal(stack_args)
    
    # Step 3: Create spatial splits
    logger.info("=" * 50)
    logger.info("STEP 3: Spatial Splitting")
    logger.info("=" * 50)
    
    split_args = argparse.Namespace(
        mapping=mapping_path,
        zone_mask=args.zone_mask,
        stacked_folder=stack_args.output_dir,
        output_csv=None,
        min_temporal_samples=args.min_temporal_samples,
        excluded_classes=args.excluded_classes,
        zone_mapping=args.zone_mapping
    )
    full_path, stacked_path = create_spatial_splits(split_args)
    
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Results saved to:")
    logger.info(f"  Patches: {patches_dir}")
    logger.info(f"  Stacked arrays: {stack_args.output_dir}")
    logger.info(f"  Split mapping: {stacked_path}")
    logger.info(f"  Full mapping: {full_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crop Classification Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract patches subcommand
    extract_parser = subparsers.add_parser('extract', help='Extract labeled patches from orthoimages')
    extract_parser.add_argument('--vector', required=True, help='Path to vector file (GeoJSON/Shapefile)')
    extract_parser.add_argument('--ortho-dir', required=True, help='Directory containing orthoimage TIFFs')
    extract_parser.add_argument('--mask-path', help='Pre-computed crop mask (optional)')
    extract_parser.add_argument('--output-dir', default='data/processed/patches', help='Output directory for patches')
    extract_parser.add_argument('--patch-size', type=int, default=64, help='Patch size in pixels')
    extract_parser.add_argument('--stride', type=int, default=32, help='Sliding window stride')
    extract_parser.add_argument('--channel-first', action='store_true', default=True, help='Store patches as (C,H,W)')
    extract_parser.add_argument('--min-plots-per-class', type=int, default=3, help='Minimum plots per crop class')
    
    # Stack temporal subcommand
    stack_parser = subparsers.add_parser('stack', help='Stack temporal patches into 4D arrays')
    stack_parser.add_argument('--patches-dir', default='data/processed/patches', help='Directory containing patches')
    stack_parser.add_argument('--output-dir', default='data/processed/stacked', help='Output directory for stacked arrays')
    stack_parser.add_argument('--patch-subdir', default='patch', help='Subdirectory name within date folders')
    stack_parser.add_argument('--date-pattern', default=r'\d{6}_reflectance_ortho', help='Regex pattern for date folders')
    stack_parser.add_argument('--expected-bands', type=int, default=10, help='Expected number of spectral bands')
    stack_parser.add_argument('--min-temporal-samples', type=int, default=3, help='Minimum temporal samples required')
    
    # Split spatial subcommand
    split_parser = subparsers.add_parser('split', help='Create spatial train/val/test splits')
    split_parser.add_argument('--mapping', required=True, help='Path to stack mapping CSV')
    split_parser.add_argument('--zone-mask', required=True, help='Path to zone mask raster')
    split_parser.add_argument('--stacked-folder', required=True, help='Directory containing stacked .npy files')
    split_parser.add_argument('--output-csv', help='Output path for split mapping')
    split_parser.add_argument('--min-temporal-samples', type=int, default=3, help='Minimum temporal samples required')
    split_parser.add_argument('--excluded-classes', type=int, nargs='+', default=[1, 2], help='Class IDs to exclude')
    split_parser.add_argument('--zone-mapping', nargs='+', help='Custom zone mappings (format: zone:split)')
    
    # Full pipeline subcommand
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete preprocessing pipeline')
    pipeline_parser.add_argument('--vector', required=True, help='Path to vector file (GeoJSON/Shapefile)')
    pipeline_parser.add_argument('--ortho-dir', required=True, help='Directory containing orthoimage TIFFs')
    pipeline_parser.add_argument('--zone-mask', required=True, help='Path to zone mask raster')
    pipeline_parser.add_argument('--mask-path', help='Pre-computed crop mask (optional)')
    pipeline_parser.add_argument('--patches-output', help='Output directory for patches')
    pipeline_parser.add_argument('--stacked-output', help='Output directory for stacked arrays')
    pipeline_parser.add_argument('--patch-size', type=int, default=64, help='Patch size in pixels')
    pipeline_parser.add_argument('--stride', type=int, default=32, help='Sliding window stride')
    pipeline_parser.add_argument('--channel-first', action='store_true', default=True, help='Store patches as (C,H,W)')
    pipeline_parser.add_argument('--min-plots-per-class', type=int, default=3, help='Minimum plots per crop class')
    pipeline_parser.add_argument('--patch-subdir', default='patch', help='Subdirectory name within date folders')
    pipeline_parser.add_argument('--date-pattern', default=r'\d{6}_reflectance_ortho', help='Regex pattern for date folders')
    pipeline_parser.add_argument('--expected-bands', type=int, default=10, help='Expected number of spectral bands')
    pipeline_parser.add_argument('--min-temporal-samples', type=int, default=3, help='Minimum temporal samples required')
    pipeline_parser.add_argument('--excluded-classes', type=int, nargs='+', default=[1, 2], help='Class IDs to exclude')
    pipeline_parser.add_argument('--zone-mapping', nargs='+', help='Custom zone mappings (format: zone:split)')
    
    # Setup subcommand
    setup_parser = subparsers.add_parser('setup', help='Setup directory structure')
    setup_parser.add_argument('--base-dir', default='.', help='Base directory for setup')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_patches(args)
    elif args.command == 'stack':
        stack_temporal(args)
    elif args.command == 'split':
        create_spatial_splits(args)
    elif args.command == 'pipeline':
        run_full_pipeline(args)
    elif args.command == 'setup':
        setup_directories(args.base_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()