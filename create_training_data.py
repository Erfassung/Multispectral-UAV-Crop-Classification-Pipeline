from skimage import io, color
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

rgb_pic = "patches_128_64/patch/478_img.tif"
img_rgb = io.imread(rgb_pic)
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb.astype(np.float32) / 255.0
img_lab = color.rgb2lab(img_rgb)
plt.imshow(img_lab)
plt.axis('off')
plt.show()
"""exit()
def change_image_color_space(image_path):
    for image in glob.glob(image_path):
        img_rgb = io.imread(image_path)
        if img_rgb.dtype == np.uint8:
            img_rgb = img_rgb.astype(np.float32) / 255.0
        img_lab = color.rgb2lab(img_rgb)

        return img_lab"""


