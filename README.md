# Cutting-and-Label

pipeline for encoding vector crop labels, rasterizing them, and generating training patches.

---

## 📁 Project Directory Structure

```
cutting-and-label/
├── .venv/                        # Python virtual environment
├── .vscode/                      # VSCode settings
├── data/
│   ├── raw/                      # input orthoimagery TIFFs
│   │   ├── 230517_reflectance_ortho.tif
│   │   ├── 230526_reflectance_ortho.tif
│   │   └── …
│   └── shapefiles/
│       ├── md_FieldSHP.shp       # original shapefile
│       ├── fieldshape.geojson    # intermediate vector after changing coordinate types of md_FieldSHP.shp to EPSG:32632 - WGS 84 / UTM zone 32N
│       ├── final_shape.geojson   # encoded vector with crop_label
│       ├── crop_mask_aligned.tif # aligned mask for all orthos
│       └── …
├── patches/                      # output patches per ortho
├── patches_lab/                  # output label-only patches
├── check_bands.py                # utility script
├── create_patches.py             # generates GeoPatch tiles
├── label_encoding.py             # produces `final_shape.geojson`
├── rio_x_array.py                # rasterizes vector → mask.tif
├── testscript.py                 # scratch / experiments
├── README.md                     # this file
├── requirements.txt              # pip dependencies
└── .gitignore
```

---

## 1. Install Dependencies

Activate your virtual environment and install:

```bash
pip install -r requirements.txt
```

---

## 2. Run the Three Main Steps

1. **Encode your shapefile**  
   ```bash
   python label_encoding.py
   ```
   - Reads `data/shapefiles/md_FieldSHP.shp`  
   - Writes `data/shapefiles/final_shape.geojson` with integer `crop_label`

2. **Rasterize the GeoJSON at 3 cm**  
   ```bash
   python rio_x_array.py
   ```
   - Reads `final_shape.geojson`  
   - Writes `data/shapefiles/crop_mask_aligned.tif` at 0.03 m resolution, aligned to your orthos

3. **Create patches with GeoPatch**  
   ```bash
   python create_patches.py
   ```
   - Tiles all TIFFs in `data/raw/` into 256×256 patches (stride 128)  
   - Saves label-only patches into `patches/name_of_ortho/`

---

## 3. (Alternative) Rasterize in QGIS / with GDAL

If you prefer GDAL directly, run this **one-liner** in the OSGeo4W Shell or Windows cmd (no backslashes):

```bash
gdal_rasterize   -where "crop_label <> 8"   -a crop_label   -init 255 -a_nodata 255   -at   -tr 0.03 0.03 -tap   -te 357385.377829 5610155.846325 357558.563412 5610216.304563   -ot UInt16 -of GTiff   data/shapefiles/final_shape.geojson   data/shapefiles/final_shape_raster.tif
```

- `-where "crop_label <> 8"`: skip background  
- `-init 255 -a_nodata 255`: nodata outside 0–7  
- `-at`: ALL_TOUCHED fill  
- `-tr 0.03 0.03`: 3 cm pixels  
- `-tap`: align origin  
- `-te xmin ymin xmax ymax`: exact bounds  
- `-ot UInt16`: integer output  
- `-of GTiff`: GeoTIFF

Verify:

```bash
gdalinfo -mm -stats data/shapefiles/final_shape_raster.tif
# Minimum = 0, Maximum = 7, NoData = 255
```

In QGIS: add `final_shape_raster.tif`, set **Symbology ► Singleband pseudocolor**, classify 0–7, mark 255 as transparent.

Now you’re ready to feed the TIFF into your GeoPatch workflow!
