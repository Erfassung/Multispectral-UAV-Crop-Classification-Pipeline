https://chatgpt.com/canvas/shared/6828800d57e881919d401cb8f072910a

pip install -r requirements.txt
color space von rgb zu lab noch ändern später !!!

# Rasterize with python script rio_x_array:
run in Terminal: python rio_x_array.py





# Rasterize from .geojson with QGIS.

Rasterize to a labeled TIFF when you have geojson.
# Run one line in the OSGeo4W Shell or Windows cmd (no backslashes) find the OSGeo4W Shell over your machines search function. 
# edit the code below if you run it outside your ide for me it was like this because i tested it on my desktop:
# gdal_rasterize -where "crop_label <> 8" -a crop_label -init 255 -a_nodata 255 -at -tr 0.03 0.03 -tap -te 357385.377829 5610155.846325 357558.563412 5610216.304563 -ot UInt16 -of GTiff "C:\Users\pinhe\Desktop\final_shape.geojson" "C:\Users\pinhe\Desktop\final_shape_raster.tif" edit the code below as needed.

gdal_rasterize -where "crop_label <> 8" -a crop_label -init 255 -a_nodata 255 -at -tr 0.03 0.03 -tap -te 357385.377829 5610155.846325 357558.563412 5610216.304563 -ot UInt16 -of GTiff data/shapefiles/final_shape.geojson data/shapefiles/final_shape_raster.tif

# Explanation:
-where "crop_label <> 8": skip the background polygon

-init 255 -a_nodata 255: start at 255 (nodata), outside 0–7 range

-at (ALL_TOUCHED): fill every touched pixel

-tr 0.03 0.03: 3 cm × 3 cm pixels

-tap: align grid origin to multiples of 0.03

-te xmin ymin xmax ymax: exact bounds from your GeoJSON

-ot UInt16: integer output

-of GTiff: GeoTIFF format

# Verify the raster output:
gdalinfo -mm -stats data/shapefiles/final_shape_raster.tif
Minimum = 0, Maximum = 7
NoData = 255

# now we can load into QGIS for checking the geojson + the raster.tif and then feed it to our geopatch script
In QGIS: Add final_shape_raster.tif, set Symbology ► Singleband pseudocolor, classify 0–7, mark 255 as transparent.
