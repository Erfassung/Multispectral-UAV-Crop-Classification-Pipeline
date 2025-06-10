# import os
# import glob
# from GeoPatch import TrainPatch

# patches_root = "patches"
# os.makedirs(patches_root, exist_ok=True)

# for ortho_path in glob.glob("data/raw/*.tif"):
#     stem = os.path.splitext(os.path.basename(ortho_path))[0]
#     out_dir = os.path.join(patches_root, stem)
#     os.makedirs(out_dir, exist_ok=True)
#     mask_path = "data/shapefiles/crop_mask_aligned.tif"
#     patcher = TrainPatch(image=ortho_path, label=mask_path, patch_size=256, stride=128, channel_first=True)
#     patcher.data_dimension()
#     patcher.patch_info()
#     patcher.save_Geotif(folder_name=out_dir, only_label=True)
#     print(f"Saved patches for {stem} to {out_dir}")
import os
import glob
import numpy as np
from GeoPatch import TrainPatch

patches_root = "patches"
os.makedirs(patches_root, exist_ok=True)

def extract_time_label(filename):
    # Angenommen, der Dateiname ist z.B. 230517_reflectance_ortho.tif, dann werden die ersten 6 Zeichen als Datum verwendet
    return filename[:6]

for ortho_path in glob.glob("data/raw/*.tif"):
    stem = os.path.splitext(os.path.basename(ortho_path))[0]
    out_dir = os.path.join(patches_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    mask_path = "data/shapefiles/crop_mask_aligned.tif"
    patcher = TrainPatch(image=ortho_path, label=mask_path, patch_size=256, stride=128, channel_first=True)
    patcher.data_dimension()
    patcher.patch_info()
    # Patch und Label extrahieren
    img_stack, mask_stack = patcher.save_numpy(folder_name="", only_label=True, return_stacked=True, save_stack=False)
    time_label = extract_time_label(os.path.basename(ortho_path))
    for i in range(mask_stack.shape[0]):
        patch = mask_stack[i]
        patch_filename = f"patch_{i:05d}_time{time_label}.npz"
        np.savez_compressed(os.path.join(out_dir, patch_filename), patch=patch)
    print(f"Saved patches for {stem} to {out_dir}, with time labels in filename.")
