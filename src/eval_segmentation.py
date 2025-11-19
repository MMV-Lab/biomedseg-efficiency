# src/eval_segmentation.py
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score


def compute_panoptic_quality(gt_bin: np.ndarray, pred_bin: np.ndarray) -> float:
    """
    A simplified panoptic quality proxy based on IoU and F1.

    You can replace this with a more exact PQ implementation if needed.
    """
    iou = jaccard_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    return 0.5 * (iou + f1)


def evaluate_pair(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    gt_bin = (gt_mask > 0).astype(np.uint8).flatten()
    pred_bin = (pred_mask > 0).astype(np.uint8).flatten()

    iou = jaccard_score(gt_bin, pred_bin, zero_division=0)
    dice = f1_score(gt_bin, pred_bin, zero_division=0)
    precision = precision_score(gt_bin, pred_bin, zero_division=0)
    recall = recall_score(gt_bin, pred_bin, zero_division=0)
    accuracy = accuracy_score(gt_bin, pred_bin)
    pq = compute_panoptic_quality(gt_bin, pred_bin)

    return iou, dice, precision, recall, accuracy, pq


def evaluate_folder(
    gt_dir: str,
    pred_dir: str,
    gt_suffix: str = "_GT.tif",
    pred_suffix: str = "_IM_cp_masks.tif",
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate segmentation results for a dataset.

    Ground truth and predictions are matched by base filename and suffix.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    results = []

    gt_files = sorted([f for f in gt_dir.iterdir() if f.name.endswith(gt_suffix)])

    for gt_file in tqdm(gt_files, desc="Evaluating masks"):
        base_name = gt_file.name.replace(gt_suffix, "")
        pred_file = pred_dir / f"{base_name}{pred_suffix}"

        if not pred_file.exists():
            print(f"[WARN] Missing prediction for {base_name}")
            continue

        gt_mask = np.array(Image.open(gt_file))
        pred_mask = np.array(Image.open(pred_file))

        iou, dice, precision, recall, accuracy, pq = evaluate_pair(gt_mask, pred_mask)

        results.append(
            {
                "image": base_name,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "panoptic_quality": pq,
            }
        )

    df = pd.DataFrame(results)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[OK] Saved metrics to {output_csv}")

    print(df.describe())
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_suffix", type=str, default="_GT.tif")
    parser.add_argument("--pred_suffix", type=str, default="_IM_cp_masks.tif")
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    evaluate_folder(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        gt_suffix=args.gt_suffix,
        pred_suffix=args.pred_suffix,
        output_csv=args.out_csv,
    )
