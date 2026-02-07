from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def make_expanding_folds(
    T: int,
    train_start: int,
    first_train_end: int,
    val_window: int,
    n_folds: int,
) -> List[Dict[str, Tuple[int, int]]]:
    """
    Build expanding-window CV folds.

    Args:
        T: total time length
        train_start: usually 0
        first_train_end: index where first train ends (exclusive)
        val_window: number of timesteps for each validation fold
        n_folds: how many folds to generate

    Returns:
        List of dicts with keys: "train_slice", "val_slice"
    """
    folds: List[Dict[str, Tuple[int, int]]] = []

    train_end = first_train_end
    val_start = train_end
    for _ in range(n_folds):
        val_end = min(val_start + val_window, T)
        if val_end <= val_start:
            break

        folds.append(
            {
                "train_slice": (train_start, train_end),
                "val_slice": (val_start, val_end),
            }
        )

        # expand training to include the validation just used
        train_end = val_end
        val_start = val_end

    return folds


def extract_fold_data(
    temporal_node_tensor: np.ndarray,
    folds: Sequence[Dict[str, Tuple[int, int]]],
    fold_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    temporal_node_tensor: [T, N, F]
    folds: list of dicts from make_expanding_folds
    fold_idx: which fold (0-based)
    """
    if not (0 <= fold_idx < len(folds)):
        raise IndexError(f"Fold index {fold_idx} invalid.")

    tr_slice = folds[fold_idx]["train_slice"]
    va_slice = folds[fold_idx]["val_slice"]

    train_data = temporal_node_tensor[tr_slice[0] : tr_slice[1]]
    val_data = temporal_node_tensor[va_slice[0] : va_slice[1]]

    return train_data, val_data
