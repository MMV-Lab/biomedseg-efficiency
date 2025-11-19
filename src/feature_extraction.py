# src/feature_extraction.py
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTMAEModel


def load_vitmae(model_name: str = "facebook/vit-mae-base", device: str = "cuda"):
    """
    Load ViT-MAE model and image processor from HuggingFace.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTMAEModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


def extract_patch_features(
    image_dir: str,
    output_feature_path: str,
    output_patchlist_path: str,
    model_name: str = "facebook/vit-mae-base",
    device: str = "cuda",
    resize_size: int = 224,
) -> None:
    """
    Extract ViT-MAE features for all image patches in a directory.

    Saves:
      - features as a NumPy array (.npy)
      - corresponding patch paths as a text file (.txt)
    """
    image_dir = Path(image_dir)
    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
    )

    processor, model = load_vitmae(model_name, device=device)

    all_features = []
    patch_paths = []

    for img_path in tqdm(image_files, desc="Extracting ViT-MAE features"):
        image = Image.open(img_path).convert("RGB")
        if image.size != (resize_size, resize_size):
            image = image.resize((resize_size, resize_size), Image.BILINEAR)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        all_features.append(feat)
        patch_paths.append(str(img_path))

    features = np.stack(all_features, axis=0)
    output_feature_path = Path(output_feature_path)
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output_patchlist_path = Path(output_patchlist_path)
    output_patchlist_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_feature_path, features)

    with open(output_patchlist_path, "w") as f:
        for p in patch_paths:
            f.write(p + "\n")

    print(f"[OK] Saved features to {output_feature_path}")
    print(f"[OK] Saved patch list to {output_patchlist_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--feature_out", type=str, required=True)
    parser.add_argument("--patchlist_out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize_size", type=int, default=224)
    args = parser.parse_args()

    extract_patch_features(
        image_dir=args.image_dir,
        output_feature_path=args.feature_out,
        output_patchlist_path=args.patchlist_out,
        model_name=args.model_name,
        device=args.device,
        resize_size=args.resize_size,
    )
