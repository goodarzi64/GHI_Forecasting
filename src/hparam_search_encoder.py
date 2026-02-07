from __future__ import annotations

import os
import random
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .cv_splits import extract_fold_data
from .temporal_autoencoder import pretrain_en_de_with_regularizers


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class HParamConfig:
    embed_dims: Sequence[int]
    conv_hiddens: Sequence[int]
    seeds: Sequence[int]
    folds: Sequence[int]
    window: int = 12
    use_attention: bool = True
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    early_stopping_patience: int = 5


def run_hparam_search(
    temporal_node_tensor: torch.Tensor,
    folds: Sequence[Dict[str, Tuple[int, int]]],
    mask_embed: torch.Tensor,
    mask_cloud: torch.Tensor,
    base_dir: str,
    cfg: HParamConfig,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """
    Grid-search encoder hyperparameters with expanding-window CV.
    Returns list of results dicts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(base_dir, exist_ok=True)

    results: List[Dict[str, float]] = []

    for embed_dim in cfg.embed_dims:
        for conv_hidden in cfg.conv_hiddens:
            if verbose:
                print(f"\nTesting embed_dim={embed_dim}, conv_hidden={conv_hidden}")

            fold_scores: List[float] = []

            for seed in cfg.seeds:
                set_seed(seed)
                if verbose:
                    print(f"  Seed {seed}")

                for f in cfg.folds:
                    if verbose:
                        print(f"    Fold {f + 1}")

                    train_data, val_data = extract_fold_data(
                        temporal_node_tensor, folds, f
                    )

                    ckpt_path = os.path.join(
                        base_dir,
                        f"ed{embed_dim}_ch{conv_hidden}_seed{seed}_fold{f + 1}.pt",
                    )

                    _, logs, _ = pretrain_en_de_with_regularizers(
                        train_tensor=train_data,
                        val_tensor=val_data,
                        embed_dim=embed_dim,
                        conv_hidden=conv_hidden,
                        save_path=ckpt_path,
                        window=cfg.window,
                        use_attention=cfg.use_attention,
                        batch_size=cfg.batch_size,
                        epochs=cfg.epochs,
                        lr=cfg.lr,
                        early_stopping_patience=cfg.early_stopping_patience,
                        device=device,
                        mask_embed=mask_embed,
                        mask_cloud=mask_cloud,
                        verbose=verbose,
                    )

                    best_val = float(min(logs["val_total"]))
                    fold_scores.append(best_val)

                    if verbose:
                        print(f"      best val = {best_val:.4f}")

            mean_val = mean(fold_scores)
            std_val = stdev(fold_scores) if len(fold_scores) > 1 else 0.0

            results.append(
                {
                    "embed_dim": float(embed_dim),
                    "conv_hidden": float(conv_hidden),
                    "val_mean": float(mean_val),
                    "val_std": float(std_val),
                }
            )

            if verbose:
                print(f"  mean={mean_val:.4f} ± {std_val:.4f}")

    return results


def select_best(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    return min(results, key=lambda x: x["val_mean"])  # type: ignore[return-value]
