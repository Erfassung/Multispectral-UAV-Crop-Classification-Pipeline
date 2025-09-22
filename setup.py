#!/usr/bin/env python3
"""
Setup Script for Crop Classification Pipeline

This script sets up the complete environment for the crop classification pipeline:
- Creates directory structure
- Sets up Python virtual environment  
- Installs required dependencies
- Copies example notebooks

Usage:
    python setup.py
    python setup.py --python-path /path/to/python3.9
    python setup.py --no-venv  # Skip virtual environment creation
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a system command with error handling."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def create_directory_structure():
    """Create the standard directory structure."""
    directories = [
        "data",
        "data/raw",
        "data/raw/orthoimages", 
        "data/raw/vectors",
        "data/raw/zone_masks",
        "data/processed",
        "data/processed/patches",
        "data/processed/stacked",
        "output",
        "output/models",
        "output/predictions", 
        "output/visualizations",
        "notebooks",
        "logs",
        "configs"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Create .gitkeep files for empty directories
    gitkeep_dirs = [
        "data/raw/orthoimages",
        "data/raw/vectors", 
        "data/raw/zone_masks",
        "data/processed/patches",
        "data/processed/stacked",
        "output/models",
        "output/predictions",
        "output/visualizations",
        "logs"
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()
    
    print("✅ Directory structure created successfully")

def create_virtual_environment(python_path="python3"):
    """Create and activate virtual environment."""
    venv_name = "venv"
    
    print(f"🐍 Creating virtual environment with {python_path}...")
    
    # Check if Python exists
    try:
        result = run_command([python_path, "--version"], "Checking Python version")
        print(f"   Using: {result.stdout.strip()}")
    except:
        print(f"   ❌ Python not found at {python_path}")
        print("   Please specify correct Python path with --python-path")
        sys.exit(1)
    
    # Create virtual environment
    if os.path.exists(venv_name):
        print(f"   Virtual environment '{venv_name}' already exists")
    else:
        run_command([python_path, "-m", "venv", venv_name], "Creating virtual environment")
    
    return venv_name

def get_venv_python(venv_name):
    """Get the path to Python in the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_name, "Scripts", "python.exe")
    else:  # Unix/Linux/MacOS
        return os.path.join(venv_name, "bin", "python")

def install_dependencies(venv_name=None):
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    if venv_name:
        python_cmd = get_venv_python(venv_name)
    else:
        python_cmd = sys.executable
    
    # Upgrade pip first
    run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    
    # Install from requirements.txt
    if os.path.exists("requirements.txt"):
        run_command([python_cmd, "-m", "pip", "install", "-r", "requirements.txt"], "Installing requirements")
    else:
        print("   ⚠️  requirements.txt not found, installing basic packages...")
        basic_packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "geopandas>=0.10.0",
            "rasterio>=1.2.0",
            "shapely>=1.8.0",
            "tqdm>=4.62.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "jupyter>=1.0.0",
            "joblib>=1.1.0"
        ]
        for package in basic_packages:
            run_command([python_cmd, "-m", "pip", "install", package], f"Installing {package}")
    
    print("✅ Dependencies installed successfully")

def copy_example_notebook():
    """Copy example training notebook if it exists."""
    notebook_source = "../training_notebooks/GeoPatch_RF_vs_PCA_KFold_Pipeline.ipynb"
    notebook_dest = "notebooks/example_training_pipeline.ipynb"
    
    if os.path.exists(notebook_source):
        print("📓 Copying example notebook...")
        shutil.copy2(notebook_source, notebook_dest)
        print(f"   Copied: {notebook_source} -> {notebook_dest}")
    else:
        print("   ⚠️  Example notebook not found, skipping...")

def create_config_files():
    """Create example configuration files."""
    print("⚙️  Creating configuration files...")
    
    # Create example config.json
    config_content = """{
    "patch_extraction": {
        "patch_size": 256,
        "stride": 128,
        "channel_first": true,
        "min_plots_per_class": 3
    },
    "temporal_stacking": {
        "expected_bands": 10,
        "min_temporal_samples": 3,
        "date_pattern": "\\\\d{6}_reflectance_ortho"
    },
    "spatial_splitting": {
        "excluded_classes": [1, 2],
        "zone_mapping": {
            "1": "train",
            "2": "train", 
            "3": "val",
            "4": "test"
        }
    }
}"""
    
    with open("configs/config.json", "w") as f:
        f.write(config_content)
    
    # Create example data README
    data_readme = """# Data Directory Structure

## Raw Data
- `data/raw/orthoimages/`: Place your multispectral orthoimage TIFFs here
- `data/raw/vectors/`: Place your crop field vector files (GeoJSON/Shapefile) here  
- `data/raw/zone_masks/`: Place your spatial zone mask rasters here

## Processed Data
- `data/processed/patches/`: Extracted patches (auto-generated)
- `data/processed/stacked/`: Temporal stacks (auto-generated)

## Expected File Naming
- Orthoimages: `YYMMDD_reflectance_ortho.tif` (e.g., `230601_reflectance_ortho.tif`)
- Vector files: Any name ending in `.geojson` or `.shp`
- Zone masks: Any name ending in `.tif`

## Getting Started
1. Place your data files in the appropriate raw directories
2. Run: `python main.py pipeline --vector data/raw/vectors/your_file.geojson --ortho-dir data/raw/orthoimages/ --zone-mask data/raw/zone_masks/your_mask.tif`
"""
    
    with open("data/README.md", "w") as f:
        f.write(data_readme)
    
    print("   Created: configs/config.json")
    print("   Created: data/README.md")

def create_gitignore():
    """Create .gitignore file.""" 
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Data files
data/raw/orthoimages/*.tif
data/raw/vectors/*.geojson
data/raw/vectors/*.shp
data/raw/zone_masks/*.tif
data/processed/patches/
data/processed/stacked/
output/models/*.joblib
output/predictions/*.csv
output/visualizations/*.png

# Logs
logs/*.log
pipeline.log

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("   Created: .gitignore")

def print_next_steps(venv_name=None):
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    if venv_name:
        if os.name == 'nt':  # Windows
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:  # Unix/Linux/MacOS
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print(f"\n📋 Next steps:")
        print(f"1. Activate virtual environment: {activate_cmd}")
        print(f"2. Place your data in the data/raw/ directories")
        print(f"3. Run the pipeline: python main.py --help")
    else:
        print(f"\n📋 Next steps:")
        print(f"1. Place your data in the data/raw/ directories") 
        print(f"2. Run the pipeline: python main.py --help")
    
    print(f"\n🔍 Example commands:")
    print(f"   python main.py setup")
    print(f"   python main.py pipeline --vector data/raw/vectors/fields.geojson \\")
    print(f"                            --ortho-dir data/raw/orthoimages/ \\")
    print(f"                            --zone-mask data/raw/zone_masks/zones.tif")
    
    print(f"\n📚 Documentation:")
    print(f"   - See README.md for detailed usage instructions")
    print(f"   - See data/README.md for data organization")
    print(f"   - See notebooks/ for example training pipeline")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup crop classification pipeline environment")
    parser.add_argument("--python-path", default="python3", help="Path to Python executable")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Crop Classification Pipeline")
    print("="*50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create virtual environment (optional)
    venv_name = None
    if not args.no_venv:
        venv_name = create_virtual_environment(args.python_path)
    
    # Install dependencies (optional)
    if not args.no_deps:
        install_dependencies(venv_name)
    
    # Copy example files
    copy_example_notebook()
    
    # Create configuration files
    create_config_files()
    create_gitignore()
    
    # Print next steps
    print_next_steps(venv_name)

if __name__ == "__main__":
    main()