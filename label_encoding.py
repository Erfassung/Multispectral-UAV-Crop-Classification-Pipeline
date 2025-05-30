from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import pandas as pd
import os
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""le = LabelEncoder() # example from documentation
le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
list(le.classes_)
print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))"""

#check/create the input folder
os.makedirs("data/shapefiles", exist_ok=True)

#read shapefile
gdf = gpd.read_file("data/shapefiles/fieldshape.geojson") #shape file needs to be in the sub dir data/shapefiles 

#encode field and capture array
le = LabelEncoder()
labels = le.fit_transform(gdf["crop"])
print("String → Integer mapping:")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls!r} → {code}")

#print transformed array
print("\nTransformed crop_label array (first 10):")
print(labels[:10])

#assign back and print sample new column
gdf["crop_label"] = labels
print("\ngdf['crop_label'] sample values:")
print(gdf["crop_label"].head().tolist())

#inverse_transform on a sample
sample_codes = [2, 2, 1]
print("\nInverse transform of [2, 2, 1]:")
print(list(le.inverse_transform(sample_codes)))

#write geojson
gdf.to_file("data/shapefiles/final_shape.geojson", driver="GeoJSON") #the file will be written to the sub dir data/shapefiles


