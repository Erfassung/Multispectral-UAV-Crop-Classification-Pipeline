import os
import numpy as np
import rasterio
from collections import Counter

# Passe das an deinen root-Ordner an
root_dir = "patches_npy"

# Gehe alle Tagesordner durch
for day in os.listdir(root_dir):
    day_path = os.path.join(root_dir, day)
    label_dir = os.path.join(day_path, "label")
    patch_dir = os.path.join(day_path, "patch")

    if not os.path.isdir(label_dir) or not os.path.isdir(patch_dir):
        continue

    # Hole alle Label-Dateien
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith("_lbl.tif")])

    for label_file in label_files:
        idx = label_file.split("_")[0]
        img_file = f"{idx}_img.tif"

        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(patch_dir, img_file)

        if not os.path.exists(img_path):
            print(f"Bild {img_file} nicht gefunden")
            continue

        # Label laden
        with rasterio.open(label_path) as src:
            label_data = src.read(1)

        # Majority Vote bestimmen
        unique, counts = np.unique(label_data, return_counts=True)
        majority_class = unique[np.argmax(counts)]

        # Neuen Namen erzeugen
        new_img_name = f"{idx}_class{majority_class}.tif"
        new_img_path = os.path.join(patch_dir, new_img_name)

        # Datei umbenennen
        os.rename(img_path, new_img_path)
        print(f"{img_file} -> {new_img_name}")
