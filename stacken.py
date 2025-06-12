import os
import re
import xarray as xr
import rasterio
import numpy as np
from datetime import datetime

# 🔧 Basisverzeichnisse anpassen
BASE_DIR = "patches_npy/"
PATCH_SUBDIR = "patch"
OUTPUT_DIR = "stacked_patches_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary: patch_id → [{date: ..., path: ...}, ...]
patch_dict = {}

# Alle Zeitordner sortiert einlesen (z. B. t00_230601_...)
folders = sorted([f for f in os.listdir(BASE_DIR) if f.startswith("t")])

for folder in folders:
    match = re.match(r"t\d+_(\d{6})_reflectance_ortho", folder)
    if not match:
        continue
    date_str = match.group(1)  # z. B. "230601"
    timestamp = datetime.strptime(date_str, "%y%m%d")

    patch_folder = os.path.join(BASE_DIR, folder, PATCH_SUBDIR)
    if not os.path.exists(patch_folder):
        continue

    for fname in os.listdir(patch_folder):
        if not fname.endswith(".tif"):
            continue
        patch_id = fname.split("_")[0]  # z. B. "10" aus "10_class5.tif"
        patch_dict.setdefault(patch_id, []).append({
            'date': timestamp,
            'path': os.path.join(patch_folder, fname)
        })

# Für jede Patch-ID das Zeitstapel-Array erzeugen und speichern
for patch_id, entries in patch_dict.items():
    arrays = []
    dates = []

    # Nach Zeit sortieren
    entries = sorted(entries, key=lambda x: x['date'])

    for entry in entries:
        with rasterio.open(entry['path']) as src:
            data = src.read()  # (bands, height, width)
            if data.shape[0] != 10:
                print(f"⚠️  Patch {entry['path']} hat {data.shape[0]} Bänder – erwartet: 10")
                continue
            data = np.transpose(data, (1, 2, 0))  # (height, width, bands)
            arrays.append(data)
            dates.append(entry['date'])

    if len(arrays) == 0:
        continue

    # Zeitlicher Stack: (time, height, width, bands)
    stacked_array = np.stack(arrays, axis=0)

    # Speichern als .npy-Datei
    out_path = os.path.join(OUTPUT_DIR, f"{patch_id}.npy")
    np.save(out_path, stacked_array)

    print(f"✅ Gespeichert: {out_path} – shape {stacked_array.shape}")

