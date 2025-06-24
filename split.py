import os
import re
import ast
import pandas as pd
import rasterio
from tqdm import tqdm


mapping_csv     = "stacked_patches_npy2/stack_mapping.csv"
zone_mask_path  = "data/processed/zone_mask.tif"
output_csv      = "stacked_patches_npy2/stack_mapping_zone_split.csv"
stacked_folder  = "stacked_patches_npy2"   

#load the mapping from stacking
df = pd.read_csv(mapping_csv)

df = df[df["input_files"]
          .apply(lambda s: len(ast.literal_eval(s)) >= 3)
       ].reset_index(drop=True)
#open zone mask
zone_ds  = rasterio.open(zone_mask_path)
zone_arr = zone_ds.read(1)

zone_ids   = []
crop_types = []

#pull out the tif read center worldcord lookup zone_id extract crop_type
for files_str in tqdm(df["input_files"], desc="Assigning zones & crops"):
    rel_list = ast.literal_eval(files_str)
    if not rel_list:
        zone_ids.append(-1)
        crop_types.append(-1)
        continue

    #build the.tif path
    tif_rel   = rel_list[0]
    patch_tif = tif_rel.replace("\\", os.sep)

    #parse croptype from filename _classX.tif
    m = re.search(r"_class(\d+)\.tif$", patch_tif)
    crop_type = int(m.group(1)) if m else -1

    #read transform, size
    with rasterio.open(patch_tif) as src:
        w, h  = src.width, src.height
        tf    = src.transform

    #compute centre in world coords
    cx, cy           = w/2, h/2
    world_x, world_y = tf * (cx, cy)

    #map to mask row/col
    rz, cz = zone_ds.index(world_x, world_y)
    if 0 <= rz < zone_arr.shape[0] and 0 <= cz < zone_arr.shape[1]:
        zid = int(zone_arr[rz, cz])
    else:
        zid = -1

    zone_ids.append(zid)
    crop_types.append(crop_type)

zone_ds.close()

#append the new columns
df["zone_id"]   = zone_ids
df["crop_type"] = crop_types

#exclude class 1 and class 2
df = df[~df["crop_type"].isin([1, 2])].reset_index(drop=True)

#map zone split
def zone_to_split(z):
    if z in [1,2]: return "train"
    if z == 3:    return "val"
    if z == 4:    return "test"
    return "exclude"

df["split"] = df["zone_id"].apply(zone_to_split)

#drop any exclude
df = df[df["split"] != "exclude"].reset_index(drop=True)

#filter rows with has a .npy in stacked_folder
stacked_files = {f for f in os.listdir(stacked_folder) if f.endswith(".npy")}
df = df[df["output_file"].isin(stacked_files)].reset_index(drop=True)

#save full mapping with zone/split
df.to_csv(output_csv, index=False)
print("Saved spatially‐split mapping to:", output_csv)

#save just the output_file split for your stacked arrays
stacked_split = os.path.join(stacked_folder, "stacked_arrays_split.csv")
df[["output_file","split"]].to_csv(stacked_split, index=False)
print("Saved stacked‐array only splits to:", stacked_split)
