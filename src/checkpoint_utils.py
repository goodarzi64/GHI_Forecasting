from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    Gater: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_losses: list[Any],
    test_metrics_history: list[Any],
    checkpoint_path: str | Path,
) -> None:
    """
    Save training checkpoint for the main model and Gater.

    Args:
        model: Main forecasting model
        Gater: Optional gating module (can be None)
        optimizer: Optimizer instance
        epoch: Current epoch number
        train_losses: List of training losses
        test_metrics_history: Validation/test metrics history
        checkpoint_path: Path to save checkpoint
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "test_metrics_history": test_metrics_history,
    }

    # Save Gater if provided
    if Gater is not None:
        state["gater_state"] = Gater.state_dict()

    # Save fusion weights if model has them
    if hasattr(model, "fusion_weights"):
        with torch.no_grad():
            weights = torch.softmax(model.fusion_weights, dim=-1).detach().cpu()
        state["fusion_weights"] = weights

    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    Embedor: torch.nn.Module | None,
    Gater: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[int, list[Any], list[Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if Embedor is not None and "embedor_state" in ckpt:
        Embedor.load_state_dict(ckpt["embedor_state"])

    if Gater is not None and "gater_state" in ckpt:
        Gater.load_state_dict(ckpt["gater_state"])

    # Restore fusion weights
    if "fusion_weights" in ckpt and hasattr(model, "fusion_weights"):
        model.fusion_weights.data.copy_(ckpt["fusion_weights"].to(device))

    return ckpt["epoch"], ckpt.get("train_losses", []), ckpt.get("test_metrics_history", [])
