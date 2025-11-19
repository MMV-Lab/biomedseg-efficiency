# src/dq_selection.py
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def compute_distance_matrix(features: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Compute the pairwise squared Euclidean distance matrix in mini-batches.
    """
    n = features.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="Computing distance matrix"):
        end_i = min(i + batch_size, n)
        dist_matrix[i:end_i] = cdist(features[i:end_i], features, metric="sqeuclidean")
    return dist_matrix


def dataset_quantization_bins(
    features: np.ndarray,
    n_bins: int = 10,
    random_state: int = 42,
) -> List[np.ndarray]:
    """
    A simplified Dataset Quantization procedure that partitions data into n_bins.

    Returns a list of arrays, where each array contains indices assigned to one bin.
    """
    rng = np.random.RandomState(random_state)
    n = features.shape[0]
    all_indices = np.arange(n)

    dist_matrix = compute_distance_matrix(features)

    bin_sizes = [n // n_bins] * n_bins
    for i in range(n % n_bins):
        bin_sizes[i] += 1

    bins: List[np.ndarray] = []
    selected_indices = set()

    for bin_id in range(n_bins):
        bin_indices = []
        bin_size = bin_sizes[bin_id]

        remaining = list(set(all_indices) - selected_indices) if selected_indices else list(all_indices)

        norms = np.sum(dist_matrix[remaining], axis=1)
        first_idx = remaining[int(np.argmin(norms))]
        bin_indices.append(first_idx)
        selected_indices.add(first_idx)

        while len(bin_indices) < bin_size:
            remaining = list(set(all_indices) - selected_indices)
            if not remaining:
                break

            scores = []
            for idx in remaining:
                d_to_bin = dist_matrix[idx, bin_indices].mean()
                d_to_all = dist_matrix[idx].mean()
                gain = d_to_bin - d_to_all
                scores.append(gain)

            best_idx = remaining[int(np.argmax(scores))]
            bin_indices.append(best_idx)
            selected_indices.add(best_idx)

        bins.append(np.array(bin_indices, dtype=int))

    return bins


def select_coreset_from_bins(
    bins: List[np.ndarray],
    ratio: float,
) -> np.ndarray:
    """
    Select a fixed ratio of samples from each bin and merge them into a coreset.
    """
    indices = []
    for b in bins:
        k = max(1, int(len(b) * ratio))
        indices.extend(list(b[:k]))
    return np.unique(np.array(indices, dtype=int))


def run_dq_selection(
    feature_path: str,
    patchlist_path: str,
    out_indices_path: str,
    out_patchlist_subset_path: str,
    n_bins: int = 10,
    ratio: float = 0.1,
) -> None:
    """
    Full DQ pipeline: load features, quantize into bins, select a coreset, and save
    both indices and corresponding patch paths.
    """
    feature_path = Path(feature_path)
    patchlist_path = Path(patchlist_path)
    out_indices_path = Path(out_indices_path)
    out_patchlist_subset_path = Path(out_patchlist_subset_path)

    features = np.load(feature_path)
    bins = dataset_quantization_bins(features, n_bins=n_bins)
    coreset_indices = select_coreset_from_bins(bins, ratio=ratio)

    out_indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_indices_path, coreset_indices)

    with open(patchlist_path, "r") as f:
        all_paths = [line.strip() for line in f.readlines()]

    selected_paths = [all_paths[i] for i in coreset_indices]
    out_patchlist_subset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_patchlist_subset_path, "w") as f:
        for p in selected_paths:
            f.write(p + "\n")

    print(f"[OK] Saved coreset indices to {out_indices_path}")
    print(f"[OK] Saved coreset patch list to {out_patchlist_subset_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--patchlist", type=str, required=True)
    parser.add_argument("--out_indices", type=str, required=True)
    parser.add_argument("--out_patchlist_subset", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.1)
    args = parser.parse_args()

    run_dq_selection(
        feature_path=args.features,
        patchlist_path=args.patchlist,
        out_indices_path=args.out_indices,
        out_patchlist_subset_path=args.out_patchlist_subset,
        n_bins=args.n_bins,
        ratio=args.ratio,
    )
