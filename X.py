import os
import argparse
import rasterio
import numpy as np
from PIL import Image

# Normalisiert einen Patch auf den Wertebereich 0-255
def normalize_patch(patch):
    patch_min, patch_max = patch.min(), patch.max()
    if patch_max - patch_min < 1e-6:
        # Wenn der Wertebereich zu klein ist, wird ein Null-Patch zurückgegeben
        return np.zeros_like(patch, dtype=np.uint8)
    patch_norm = (patch - patch_min) / (patch_max - patch_min)
    return (patch_norm * 255).astype(np.uint8)

# Speichert einen Patch als Bilddatei
def save_patch(patch, output_path):
    # Wenn der Patch nur eine Ebene hat, wird diese direkt gespeichert
    patch_to_save = patch[0] if patch.shape[0] == 1 else np.moveaxis(patch, 0, -1)
    Image.fromarray(patch_to_save).save(output_path)

# Extrahiert Patches aus einer GeoTIFF-Datei
def extract_patches_from_tif(image_path, output_dir, patch_height, patch_width, stride_y, stride_x):
    with rasterio.open(image_path) as src:
        # Liest das Bild als Array
        image_array = src.read()
        height, width = image_array.shape[1:]

    # Erstellt den Basisnamen für die Ausgabe
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)  # Erstellt das Ausgabeverzeichnis, falls es nicht existiert
    patch_id = 0

    # Iteriert über das Bild, um Patches zu extrahieren
    for y in range(0, height - patch_height + 1, stride_y):
        for x in range(0, width - patch_width + 1, stride_x):
            patch = image_array[:, y:y + patch_height, x:x + patch_width]
            if np.all(patch == 0):  # Überspringt Patches, die nur Nullen enthalten
                continue
            patch_uint8 = normalize_patch(patch)  # Normalisiert den Patch
            output_path = os.path.join(output_dir, f"{base_name}_patch_{patch_id:04d}.png")
            save_patch(patch_uint8, output_path)  # Speichert den Patch
            patch_id += 1
    print(f"\n✅ {patch_id} Patches wurden in {output_dir} gespeichert")

if __name__ == "__main__":
    # Argumentparser für die Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="GeoPatch Einzeldatei-Cutter")
    parser.add_argument("--image_path", type=str, required=True, help="Pfad zur Eingabe-.tif-Datei")
    parser.add_argument("--output_dir", type=str, default="patches", help="Verzeichnis zum Speichern der Ausgabe-Patches")
    parser.add_argument("--patch_height", type=int, default=256, help="Höhe der Patches")
    parser.add_argument("--patch_width", type=int, default=256, help="Breite der Patches")
    parser.add_argument("--stride_y", type=int, default=128, help="Schrittweite entlang der Y-Achse")
    parser.add_argument("--stride_x", type=int, default=128, help="Schrittweite entlang der X-Achse")
    args = parser.parse_args()

    # Führt die Patch-Extraktion aus
    extract_patches_from_tif(
        args.image_path,
        args.output_dir,
        args.patch_height,
        args.patch_width,
        args.stride_y,
        args.stride_x
    )

#     print(f"✅ Fertig! Insgesamt wurden {patch_id} Patches in {output_dir} gespeichert")