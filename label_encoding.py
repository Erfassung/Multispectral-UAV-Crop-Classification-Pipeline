from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import pandas as pd
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""le = LabelEncoder() # example from documentation
le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
list(le.classes_)
print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))"""

gdf = gpd.read_file("data/shapefiles/md_FieldSHP.shp") #input
le = LabelEncoder() 
gdf["crop_label"] = le.fit_transform(gdf["crop"])
#print(list(le.inverse_transform(gdf["crop_label"]))) # check the encoding print reverse labels
gdf.to_file("data/shapefiles/md_FieldSHP.geojson", driver='GeoJSON')


