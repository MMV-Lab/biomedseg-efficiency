# Data Efficiency and Transfer Robustness in Biomedical Image Segmentation

Code for the paper:

> **Data Efficiency and Transfer Robustness in Biomedical Image Segmentation:  
> A Study of Redundancy and Forgetting with Cellpose**  
> Shuo Zhao, Jianxu Chen, IEEE BIBM 2025.

---

## Overview

This repository provides a data-centric pipeline to study:

1. Redundancy in training data via Dataset Quantization (DQ) and coreset selection.
2. Transfer robustness and catastrophic forgetting when fine-tuning Cellpose across
   multiple domains (e.g., Cyto, Histo, MultiInst).

Main functionalities:

- Patch extraction from images and masks.
- ViT-MAE-based feature extraction for patches (`facebook/vit-mae-base`).
- Dataset Quantization for binning and coreset selection.
- Evaluation of segmentation performance using IoU, Dice, Precision, Recall,
  Accuracy, and a simplified Panoptic Quality proxy.
- A Jupyter notebook that reproduces and visualizes the main experiments.

---

## Installation

```bash
git clone https://github.com/yourname/biomedseg-efficiency.git
cd biomedseg-efficiency

pip install -r requirements.txt
