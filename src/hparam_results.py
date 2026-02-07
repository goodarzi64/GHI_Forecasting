from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch


_PATTERN = re.compile(
    r"ed(?P<ed>\d+)_ch(?P<ch>\d+)_seed(?P<seed>\d+)_fold(?P<fold>\d+)\.pt"
)


def _load_logs(path: str) -> Dict[str, List[float]]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "logs" in ckpt:
        return ckpt["logs"]
    if isinstance(ckpt, dict):
        return ckpt  # assume logs directly
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def collect_hparam_records(
    ckpt_dir: str,
    pattern: re.Pattern[str] = _PATTERN,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if not match:
            continue

        info = match.groupdict()
        path = os.path.join(ckpt_dir, fname)
        logs = _load_logs(path)
        best_val = float(min(logs["val_total"]))

        records.append(
            {
                "embed_dim": int(info["ed"]),
                "conv_hidden": int(info["ch"]),
                "seed": int(info["seed"]),
                "fold": int(info["fold"]),
                "best_val": best_val,
            }
        )

    return pd.DataFrame(records)


def summarize_hparam_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["embed_dim", "conv_hidden"])
        .agg(
            val_mean=("best_val", "mean"),
            val_std=("best_val", "std"),
            n_runs=("best_val", "count"),
        )
        .reset_index()
        .sort_values("val_mean")
    )
    return summary
