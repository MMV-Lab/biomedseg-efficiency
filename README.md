# Data Efficiency and Transfer Robustness in Biomedical Image Segmentation

Code for the paper:

> **Data Efficiency and Transfer Robustness in Biomedical Image Segmentation:  
> A Study of Redundancy and Forgetting with Cellpose**  
> Shuo Zhao, Jianxu Chen, IEEE BIBM 2025.

---

## Overview

This repository provides a data-centric pipeline for investigating:

1. **Redundancy** in training data via Dataset Quantization (DQ) and coreset selection.  
2. **Transfer robustness** and **catastrophic forgetting** when fine-tuning Cellpose across multiple microscopy domains  
   (e.g., Cyto, Histo, MultiInst).

Core functionalities:

- Patch extraction from paired images and masks  
- ViT-MAE–based patch feature extraction (`facebook/vit-mae-base`)  
- Dataset Quantization for binning and coreset construction  
- Segmentation evaluation (IoU, Dice, Precision, Recall, Accuracy, PQ proxy)  
- A comprehensive Jupyter notebook reproducing experiments from the BIBM 2025 paper  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MMV-Lab/biomedseg-efficiency.git
cd biomedseg-efficiency

pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for efficient feature extraction.

---

## Repository Structure

```text
src/
  patch_extraction.py      # image + mask patch extraction
  feature_extraction.py    # ViT-MAE patch feature extraction
  dq_selection.py          # Dataset Quantization + coreset selection
  eval_segmentation.py     # segmentation metrics evaluation

notebooks/
  BIBM2025_bin_pipeline.ipynb  # full experimental workflow (feature plots, DQ curves, transfer analysis)

requirements.txt
README.md
```

---

## Usage

### 1. Patch Extraction

```bash
python -m src.patch_extraction   --input_dir /path/to/raw_dataset   --output_dir /path/to/patch_dataset   --patch_size 224   --stride 112   --img_suffix "_IM.tif"   --mask_suffix "_GT.tif"
```

This creates `images/` and `masks/` subfolders under `/path/to/patch_dataset` containing the extracted patches.

---

### 2. Feature Extraction (ViT-MAE)

```bash
python -m src.feature_extraction   --image_dir /path/to/patch_dataset/images   --feature_out features/cyto_train_vitmae.npy   --patchlist_out features/cyto_train_patch_paths.txt   --model_name facebook/vit-mae-base   --device cuda
```

The script saves:

- `features/cyto_train_vitmae.npy`: feature matrix of shape `(N, D)`  
- `features/cyto_train_patch_paths.txt`: list of corresponding patch paths  

---

### 3. Dataset Quantization & Coreset Selection

```bash
python -m src.dq_selection   --features features/cyto_train_vitmae.npy   --patchlist features/cyto_train_patch_paths.txt   --out_indices features/cyto_dq_10pct_indices.npy   --out_patchlist_subset features/cyto_dq_10pct_patch_paths.txt   --n_bins 10   --ratio 0.1
```

The resulting patch list can be used to construct a reduced training set for Cellpose
or other segmentation models.

---

### 4. Segmentation Evaluation

```bash
python -m src.eval_segmentation   --gt_dir /path/to/gt_masks   --pred_dir /path/to/pred_masks   --gt_suffix "_GT.tif"   --pred_suffix "_IM_cp_masks.tif"   --out_csv outputs/metrics_moseg_test.csv
```

This will compute:

- IoU  
- Dice  
- Precision  
- Recall  
- Accuracy  
- a Panoptic Quality (PQ) proxy  

and save a CSV with per-image metrics plus summary statistics.

---

## Notebook

The notebook `notebooks/BIBM2025_bin_pipeline.ipynb` contains the full experimental
pipeline corresponding to the paper, including:

- ViT-MAE feature visualization (e.g., t-SNE)  
- Coreset selection curves across different DQ rates  
- Cross-domain transfer performance  
- Catastrophic forgetting and replay analysis  

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{zhao2025biomedseg,
  title     = {Data Efficiency and Transfer Robustness in Biomedical Image Segmentation: A Study of Redundancy and Forgetting with Cellpose},
  author    = {Shuo Zhao and Jianxu Chen},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2025}
}
```

---

## Contact

For questions or suggestions, please open an issue or contact:

**Shuo Zhao** — shuo.zhao@isas.de  or   zhaoshuoofcourse@gmail.com  
MMV-Lab
