import os
import argparse
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path


def normalize(patch):
    pmin, pmax = patch.min(), patch.max()
    if pmax - pmin < 1e-6:
        return np.zeros_like(patch, dtype=np.uint8)
    return ((patch - pmin) / (pmax - pmin + 1e-6) * 255).astype(np.uint8)

def extract_patches(image_path, label_array, out_img_dir, out_lbl_dir, metadata, patch_height, patch_width, stride_y, stride_x):
    with rasterio.open(image_path) as img_src:
        img = img_src.read()
        h, w = img.shape[1:]

    base_name = Path(image_path).stem
    patch_id = 0

    for y in range(0, h - patch_height + 1, stride_y):
        for x in range(0, w - patch_width + 1, stride_x):
            img_patch = img[:, y:y+patch_height, x:x+patch_width]
            lbl_patch = label_array[y:y+patch_height, x:x+patch_width]

            if np.all(lbl_patch == 0):
                continue

            img_out = os.path.join(out_img_dir, f"{base_name}_patch_{patch_id:04d}.tif")
            lbl_out = os.path.join(out_lbl_dir, f"{base_name}_patch_{patch_id:04d}.tif")

            with rasterio.open(img_out, 'w', driver='GTiff',
                               height=patch_height, width=patch_width,
                               count=img.shape[0], dtype=np.uint8) as dst:
                dst.write(normalize(img_patch))

            with rasterio.open(lbl_out, 'w', driver='GTiff',
                               height=patch_height, width=patch_width,
                               count=1, dtype=lbl_patch.dtype) as dst:
                dst.write(lbl_patch, 1)

            metadata.append({
                'image_patch': img_out,
                'label_patch': lbl_out,
                'coords': f"x:{x}, y:{y}",
                'label_unique': np.unique(lbl_patch).tolist()
            })

            patch_id += 1

    return patch_id

def batch_process(img_dir, lbl_path, out_dir, patch_height, patch_width, stride_y, stride_x):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])

    out_img_dir = os.path.join(out_dir, 'images')
    out_lbl_dir = os.path.join(out_dir, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with rasterio.open(lbl_path) as lbl_src:
        label_array = lbl_src.read(1)

    metadata = []
    total = 0
    for idx, img_file in enumerate(img_files):
        print(f"[{idx + 1}/{len(img_files)}] 处理 {img_file} 与公共 label...")
        img_path = os.path.join(img_dir, img_file)
        count = extract_patches(img_path, label_array, out_img_dir, out_lbl_dir, metadata,
                                patch_height, patch_width, stride_y, stride_x)
        total += count

    # 保存 metadata
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    print(f"\n✅ 总共生成 patch 数量：{total}, metadata 保存在 metadata.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoPatch 多图配同一标签工具")
    parser.add_argument("--image_dir", type=str, required=True, help="图像文件夹")
    parser.add_argument("--label_path", type=str, required=True, help="单个标签图路径")
    parser.add_argument("--output_dir", type=str, default="geopatch_output", help="输出目录")
    parser.add_argument("--patch_height", type=int, default=256)
    parser.add_argument("--patch_width", type=int, default=256)
    parser.add_argument("--stride_y", type=int, default=128)
    parser.add_argument("--stride_x", type=int, default=128)

    args = parser.parse_args()

    batch_process(
        args.image_dir,
        args.label_path,
        args.output_dir,
        args.patch_height,
        args.patch_width,
        args.stride_y,
        args.stride_x
    )