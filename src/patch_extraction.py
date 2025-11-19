# src/patch_extraction.py
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_patches(
    input_dir: str,
    output_dir: str,
    patch_size: int = 224,
    stride: int = 112,
    img_suffix: str = "_IM.tif",
    mask_suffix: str = "_GT.tif",
) -> None:
    """
    Extract patches from paired images and masks.

    Files are expected to follow a pattern like:
      <basename>_IM.tif for images
      <basename>_GT.tif for masks
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in input_dir.glob(f"*{img_suffix}")])

    patch_id_global = 0

    for img_path in tqdm(img_files, desc="Extracting patches"):
        base_name = img_path.name.replace(img_suffix, "")
        mask_path = input_dir / f"{base_name}{mask_suffix}"

        if not mask_path.exists():
            print(f"[WARN] Missing mask for {img_path.name}, skipping.")
            continue

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        h, w = img.shape[:2]
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                img_patch = img[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                patch_name_img = f"{base_name}_patch_{patch_id_global:04d}_IM.tif"
                patch_name_mask = f"{base_name}_patch_{patch_id_global:04d}_GT.tif"

                Image.fromarray(img_patch).save(img_out_dir / patch_name_img)
                Image.fromarray(mask_patch).save(mask_out_dir / patch_name_mask)

                patch_id_global += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--img_suffix", type=str, default="_IM.tif")
    parser.add_argument("--mask_suffix", type=str, default="_GT.tif")
    args = parser.parse_args()

    extract_patches(
        args.input_dir,
        args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        img_suffix=args.img_suffix,
        mask_suffix=args.mask_suffix,
    )
