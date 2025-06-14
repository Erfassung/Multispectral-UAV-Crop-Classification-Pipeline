import os
import re
import rasterio
import numpy as np
from datetime import datetime
from tqdm import tqdm  # Optional, for progress bar

BASE_DIR = "patches_npy/"
PATCH_SUBDIR = "patch"
OUTPUT_DIR = "stacked_patches_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

patch_dict = {}

# Find all folders matching the ortho naming pattern
folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if re.match(r"\d{6}_reflectance_ortho", f) and os.path.isdir(os.path.join(BASE_DIR, f))
])

for folder in folders:
    match = re.match(r"(\d{6})_reflectance_ortho", folder)
    if not match:
        continue
    date_str = match.group(1)  # e.g. "230601"
    timestamp = datetime.strptime(date_str, "%y%m%d")

    patch_folder = os.path.join(BASE_DIR, folder, PATCH_SUBDIR)
    if not os.path.exists(patch_folder):
        continue

    for fname in os.listdir(patch_folder):
        if not fname.endswith(".tif"):
            continue
        patch_id = fname.split("_")[0]
        patch_dict.setdefault(patch_id, []).append({
            'date': timestamp,
            'path': os.path.join(patch_folder, fname)
        })

    for patch_idx, (patch_id, entries) in enumerate(patch_dict.items(), 1):
        arrays = []
        dates = []

        # Sort by date
        entries = sorted(entries, key=lambda x: x['date'])

        print(f"Processing patch {patch_id} ({patch_idx}/{len(patch_dict)}) with {len(entries)} time points...")

        for entry in entries:
            with rasterio.open(entry['path']) as src:
                data = src.read()  # (bands, height, width)
                if data.shape[0] != 10:
                    print(f"⚠️  Patch {entry['path']} has {data.shape[0]} bands – expected: 10")
                    continue
                data = np.transpose(data, (1, 2, 0))  # (height, width, bands)
                arrays.append(data)
                dates.append(entry['date'])

        if len(arrays) == 0:
            continue

        stacked_array = np.stack(arrays, axis=0)
        t, h, w, b = stacked_array.shape
        stacked_4d = stacked_array.transpose(0, 3, 1, 2).reshape(t * b, h, w)

        # Extract class label from the first filename in the group
        first_patch_fname = os.path.basename(entries[0]['path'])
        label_match = re.search(r'class(\d+)', first_patch_fname)
        if label_match:
            label = label_match.group(1)
            if label == "0":
                print(f"Skipping patch {patch_id} (class 0, background)")
                continue  # Skip background patches
            out_name = f"{patch_id}_class{label}.npy"
        else:
            out_name = f"{patch_id}.npy"

        out_path = os.path.join(OUTPUT_DIR, out_name)
        np.save(out_path, stacked_4d)
        print(f"✅ Saved: {out_path} – shape {stacked_4d.shape}")