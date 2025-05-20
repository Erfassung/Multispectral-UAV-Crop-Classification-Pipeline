import numpy as np
import rasterio as rs
import glob
import os
import math
import tqdm
import matplotlib.pyplot as plt
from GeoPatch import TrainPatch

patches_folder = "patches"
os.makedirs(patches_folder, exist_ok=True)

for image in glob.glob("data/*.tif"):
    Ortho = image
    label1 = "data/label/shaperaster.tif"
    patch = TrainPatch(image=Ortho, label=label1, patch_size=256, stride=128, channel_first=True)
    patch.data_dimension()
    patch.patch_info()
    patch.save_Geotif(folder_name=patches_folder, only_label=True)


