import os
import glob
from GeoPatch import TrainPatch

patches_root = "patches"
os.makedirs(patches_root, exist_ok=True)

masks_dir = "data/shapefiles/masks_per_ortho"

for ortho_path in glob.glob("data/raw/*.tif"):
    stem = os.path.splitext(os.path.basename(ortho_path))[0]
    out_dir = os.path.join(patches_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    mask_path = "data/shapefiles/crop_mask_aligned.tif"
    patcher = TrainPatch(image=ortho_path, label=mask_path, patch_size=256, stride=128, channel_first=True)
    patcher.data_dimension()
    patcher.patch_info()
    patcher.save_Geotif(folder_name=out_dir, only_label=True)
    print(f"Saved patches for {stem} to {out_dir}")
