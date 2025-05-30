import rasterio
import numpy as np


with rasterio.open('patches/patch/79_img.tif') as src:
    band_mins = []
    band_maxs = []
    
    for i in range(1, src.count + 1): 
        band = src.read(i)
        band_mins.append(np.min(band))
        band_maxs.append(np.max(band))


for i, (bmin, bmax) in enumerate(zip(band_mins, band_maxs), 1):
    print(f'Band {i}: min = {bmin}, max = {bmax}')

