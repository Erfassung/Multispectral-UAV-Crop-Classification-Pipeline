import numpy as np
import rasterio as rs
import glob
import os
import math
import tqdm
import matplotlib.pyplot as plt
from GeoPatch import TrainPatch

Ortho = "data/230517_reflectance_ortho.tif"
label1 = "data/Shapes.tif"

patch = TrainPatch(image= Ortho, label=label1, patch_size=256, stride=128, channel_first=True)

patch.data_dimension()

patch.patch_info()

patch.save_Geotif(folder_name="tif", only_label=True)

patch.visualize(folder_name='tif',patches_to_show=2,band_num=1,

fig_size=(10, 20),dpi=96)


