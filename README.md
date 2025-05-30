```markdown
# Cutting-and-Label

A reproducible pipeline for encoding vector crop labels, rasterizing them at 3 cm resolution, and generating training patches.

---

## Project Directory Structure

```


├── .venv/                          # Python virtual environment
├── .vscode/                        # VSCode settings
├── data/
│   ├── raw/                        # input orthoimagery TIFFs
│   │   ├── 230517\_reflectance\_ortho.tif
│   │   ├── 230526\_reflectance\_ortho.tif
│   │   └── …
│   └── shapefiles/
│       ├── final\_shape.geojson     # encoded vector with crop\_label
│       ├── fieldshape.geojson      # original or intermediate vectors
│       ├── crop\_mask.tif           # 3 cm mask (unaligned)
│       ├── crop\_mask\_aligned.tif   # aligned mask for all orthos
│       └── …
├── patches/                        # output patches per ortho
├── patches\_lab/                    # output label-only patches
├── check\_bands.py                  # utility script
├── create\_patches.py               # generates GeoPatch tiles
├── label\_encoding.py               # produces `final_shape.geojson`
├── rio\_x\_array.py                  # rasterizes vector → mask.tif
├── testscript.py                   # scratch / experiments
├── README.md                       # this file
├── requirements.txt                # pip dependencies
└── .gitignore

````

---

## 1. Install Dependencies

Activate your virtual environment and install:

```bash
pip install -r requirements.txt
````
---

## 2. Run the Three Main Steps

1. **Encode your shapefile**

   ```bash
   python label_encoding.py
   ```

   * Reads `data/shapefiles/md_FieldSHP.shp`
   * Creates `data/shapefiles/final_shape.geojson` with integer `crop_label`.

2. **Rasterize the GeoJSON at 3 cm**

   ```bash
   python rio_x_array.py
   ```

   * Reads `final_shape.geojson`
   * Writes `data/shapefiles/crop_mask_aligned.tif` at 0.03 m resolution, aligned to your orthos.

3. **Create patches with GeoPatch**

   ```bash
   python create_patches.py
   ```

   * Tiles all TIFFs in `data/raw/` into 256×256 patches (stride 128)
   * Saves label-only patches into `patches/`

---

## 3. (Alternative) Rasterize from GeoJSON in QGIS

If you prefer GDAL directly, open the **OSGeo4W Shell** (or Windows cmd) and run this one line (no `\`):

```bash
gdal_rasterize \
  -where "crop_label <> 8" \
  -a crop_label \
  -init 255 -a_nodata 255 \
  -at \
  -tr 0.03 0.03 -tap \
  -te 357385.377829 5610155.846325 357558.563412 5610216.304563 \
  -ot UInt16 -of GTiff \
  data/shapefiles/final_shape.geojson \
  data/shapefiles/final_shape_raster.tif
```

* `-where "crop_label <> 8"`: skip background
* `-init 255 -a_nodata 255`: nodata outside 0–7 range
* `-at`: ALL\_TOUCHED fill
* `-tr 0.03 0.03`: 3 cm pixels
* `-tap`: align origin
* `-te xmin ymin xmax ymax`: exact bounds
* `-ot UInt16`: integer output
* `-of GTiff`: GeoTIFF format

Verify with:

```bash
gdalinfo -mm -stats data/shapefiles/final_shape_raster.tif
# Minimum = 0, Maximum = 7, NoData = 255
```

Then load in QGIS:

1. **Add** `final_shape_raster.tif`
2. Set **Symbology ► Singleband pseudocolor**
3. Classify **0–7** and mark **255** as transparent

Now you can feed the TIFF into your `create_patches.py` or GeoPatch workflow.
