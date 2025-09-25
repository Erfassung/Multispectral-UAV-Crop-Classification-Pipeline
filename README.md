# Crop Classification Pipeline

A comprehensive preprocessing pipeline for multispectral crop classification using temporal orthoimages and machine learning.

##  Overview

This pipeline transforms raw multispectral orthoimages and crop field vectors into machine learning-ready datasets with spatially-aware train/validation/test splits. It's designed for time-series crop classification using Random Forest and other ML algorithms.

### Key Features

- ** Patch Extraction**: Extract labeled patches from multispectral orthoimages using field boundaries
- ** Temporal Stacking**: Combine multi-date patches into 4D time-series arrays
- ** Spatial Splitting**: Create spatially-aware train/val/test splits to prevent data leakage
- ** Configurable Pipeline**: Fully configurable parameters via command line or config files
- ** Built-in Analytics**: Comprehensive statistics and validation throughout the process

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crop-classification-pipeline.git
cd crop-classification-pipeline

# Run setup script (creates venv, installs dependencies, sets up directories)
python setup.py
```

### 2. Activate Environment

```bash
# Windows
venv\\Scripts\\activate

# Linux/MacOS  
source venv/bin/activate
```

### 3. Prepare Your Data

Place your data in the following structure:
```
data/raw/
├── orthoimages/          # Multispectral TIFF files
│   ├── 230601_reflectance_ortho.tif
│   ├── 230615_reflectance_ortho.tif
│   └── ...
├── vectors/              # Crop field boundaries
│   └── crop_fields.geojson
└── zone_masks/           # Spatial zone masks for splitting
    └── spatial_zones.tif
```

### 4. Run the Pipeline

```bash
# Complete pipeline in one command
python main.py pipeline \\
    --vector data/raw/vectors/crop_fields.geojson \\
    --ortho-dir data/raw/orthoimages/ \\
    --zone-mask data/raw/zone_masks/spatial_zones.tif
```

## Detailed Usage

### Step-by-Step Execution

You can run each step independently for more control:

```bash
# Step 1: Extract patches
python main.py extract \\
    --vector data/raw/vectors/crop_fields.geojson \\
    --ortho-dir data/raw/orthoimages/ \\
    --patch-size 256 \\
    --stride 128

# Step 2: Stack temporal data
python main.py stack \\
    --patches-dir data/processed/patches \\
    --output-dir data/processed/stacked

# Step 3: Create spatial splits
python main.py split \\
    --mapping data/processed/stacked/stack_mapping.csv \\
    --zone-mask data/raw/zone_masks/spatial_zones.tif \\
    --stacked-folder data/processed/stacked
```

### Configuration Options

#### Patch Extraction
- `--patch-size`: Size of patches in pixels (default: 256)
- `--stride`: Sliding window stride (default: 128)  
- `--channel-first`: Store patches as (C,H,W) instead of (H,W,C)
- `--min-plots-per-class`: Minimum field plots required per crop class (default: 3)

#### Temporal Stacking
- `--expected-bands`: Number of spectral bands expected (default: 10)
- `--min-temporal-samples`: Minimum temporal observations required (default: 3)
- `--date-pattern`: Regex pattern for date folder matching

#### Spatial Splitting
- `--excluded-classes`: Crop class IDs to exclude (default: [1, 2])
- `--zone-mapping`: Custom zone to split mapping (format: `zone:split`)

### Custom Zone Mapping

By default, the pipeline uses:
- Zones 1,2 → train
- Zone 3 → validation  
- Zone 4 → test

Customize with:
```bash
python main.py split ... --zone-mapping 1:train 2:train 3:val 4:test 5:test
```

## Project Structure

```
crop-classification-pipeline/
├── main.py                 # Main pipeline orchestrator
├── setup.py               # Environment setup script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
├── src/
│   └── libs/             # Core library modules
│       ├── __init__.py
│       ├── patch_extractor.py     # Patch extraction functionality
│       ├── temporal_stacker.py    # Temporal stacking functionality  
│       └── spatial_splitter.py    # Spatial splitting functionality
├── data/
│   ├── raw/              # Input data
│   │   ├── orthoimages/
│   │   ├── vectors/
│   │   └── zone_masks/
│   └── processed/        # Pipeline outputs
│       ├── patches/
│       └── stacked/
├── output/               # Final results
│   ├── models/
│   ├── predictions/
│   └── visualizations/
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
└── logs/                # Log files
```

## Machine Learning Training

After preprocessing, use the included training notebook:

```bash
jupyter notebook notebooks/example_training_pipeline.ipynb
```

The notebook demonstrates:
- Feature engineering from 4D temporal data
- Random Forest training with feature selection
- PCA-based dimensionality reduction
- Comprehensive model evaluation
- Spatial error analysis and visualization

## Advanced Usage

### Using as Python Library

```python
from src.libs import PatchExtractor, TemporalStacker, SpatialSplitter

# Extract patches
extractor = PatchExtractor(
    vector_path="data/raw/vectors/fields.geojson",
    ortho_dir="data/raw/orthoimages/",
    output_dir="data/processed/patches"
)
extractor.extract_all_orthos()

# Stack temporal data
stacker = TemporalStacker(
    base_dir="data/processed/patches",
    output_dir="data/processed/stacked"
)
mapping_path = stacker.run()

# Create spatial splits
splitter = SpatialSplitter(
    mapping_csv=mapping_path,
    zone_mask_path="data/raw/zone_masks/zones.tif",
    stacked_folder="data/processed/stacked"
)
splitter.run()
```

### Custom Processing

Each module is designed to be flexible and extensible:

```python
# Custom crop filtering
extractor = PatchExtractor(...)
extractor.min_plots_per_class = 5  # Require more plots per class

# Custom temporal requirements  
stacker = TemporalStacker(...)
stacker.min_temporal_samples = 5  # Require more temporal observations

# Custom spatial zones
splitter = SpatialSplitter(...)
splitter.set_custom_zone_mapping({1: "train", 2: "val", 3: "test"})
```

## Data Requirements

### Input Data Format

#### Multispectral Orthoimages
- **Format**: GeoTIFF (.tif)
- **Naming**: `YYMMDD_reflectance_ortho.tif` (e.g., `230601_reflectance_ortho.tif`)
- **Bands**: 10 spectral bands (configurable)
- **Coordinate System**: Any projected CRS
- **Data Type**: Float32 or UInt16

#### Crop Field Vectors  
- **Format**: GeoJSON (.geojson) or Shapefile (.shp)
- **Required Fields**:
  - `crop`: Crop type name (string)
  - `plot_ID`: Unique plot identifier (optional)
- **Geometry**: Polygon features representing field boundaries
- **Coordinate System**: Any CRS (will be reprojected to match orthoimages)

#### Spatial Zone Mask
- **Format**: GeoTIFF (.tif) 
- **Values**: Integer zone IDs (1, 2, 3, 4, etc.)
- **Coordinate System**: Must match orthoimages
- **Purpose**: Define spatial regions for train/val/test splitting

### Output Data Format

#### Extracted Patches
- **Format**: GeoTIFF files organized by date
- **Naming**: `{patch_id}_class{crop_class}.tif`
- **Size**: Configurable (default 256x256 pixels)
- **Bands**: Same as input orthoimages

#### Stacked Arrays
- **Format**: NumPy binary files (.npy)
- **Shape**: `(time×bands, height, width)` 
- **Metadata**: CSV mapping file with temporal and spatial information

#### Split Assignments
- **Format**: CSV files
- **Content**: Patch assignments to train/validation/test splits
- **Spatial Awareness**: Ensures no spatial overlap between splits

## Troubleshooting

### Common Issues

**Memory Errors**: Reduce patch size or batch processing
```bash
python main.py extract --patch-size 128 --stride 64
```

**Missing Dependencies**: Reinstall requirements
```bash
pip install -r requirements.txt
```

**Date Pattern Mismatch**: Adjust regex pattern
```bash
python main.py stack --date-pattern "\\d{8}_ortho"
```

**Spatial Projection Issues**: Ensure CRS compatibility
- Zone mask must match orthoimage CRS
- Vector data will be automatically reprojected

### Performance Optimization

**Large Datasets**:
- Use larger stride values to reduce patch count
- Increase minimum temporal samples to filter sparse data
- Use smaller patch sizes to reduce memory usage

**Computational Resources**:
- The pipeline supports parallel processing where possible
- Consider running on high-memory systems for large datasets
- Use SSD storage for improved I/O performance

## Pipeline Statistics

The pipeline provides comprehensive statistics at each step:

- **Patch Extraction**: Number of patches per class, spatial distribution
- **Temporal Stacking**: Temporal sample statistics, data quality metrics  
- **Spatial Splitting**: Train/val/test distribution, class balance per split

Example output:
```
 Patch Extraction Complete:
   Total patches: 15,432
   Crop classes: 5 (Potato: 3,245, Soybean: 4,123, ...)
   
 Temporal Stacking Complete:
   Stacked patches: 12,847
   Temporal samples: min=3, max=8, mean=5.2
   
 Spatial Splitting Complete:
   Train: 8,459 patches (65.8%)
   Validation: 2,144 patches (16.7%) 
   Test: 2,244 patches (17.5%)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone for development
git clone https://github.com/your-username/crop-classification-pipeline.git
cd crop-classification-pipeline

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/crop-classification-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/crop-classification-pipeline/discussions)
- **Documentation**: See `docs/` directory for additional documentation

## Acknowledgments

- **GeoPatch Library**: For efficient patch extraction from geospatial data
- **Rasterio/GDAL**: For robust geospatial data handling
- **Scikit-learn**: For machine learning utilities and validation metrics
- **Contributors**: Thanks to all contributors who have helped improve this pipeline

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{crop_classification_pipeline,
  title={Crop Classification Pipeline: A Preprocessing Framework for Multispectral Time-Series Data},
  author={Nelson Pinheiro},
  year={2024},
  url={https://github.com/your-username/crop-classification-pipeline}
}
```

---

**🌾 Happy Crop Classifying! 🌾**
