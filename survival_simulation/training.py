"""Small training helpers shared by examples and notebooks."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch

from .model import cox_ph_loss


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda without silently selecting an unavailable GPU."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        **batch,
        "numerical": batch["numerical"].to(device),
        "categorical": {key: value.to(device) for key, value in batch["categorical"].items()},
        "padding_mask": batch["padding_mask"].to(device),
        "time": batch["time"].to(device),
        "event": batch["event"].to(device),
    }


def train_one_epoch(
    model: torch.nn.Module,
    batches: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        risk = model(batch["numerical"], batch["categorical"], batch["padding_mask"])
        loss = cox_ph_loss(risk, batch["time"], batch["event"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("Training loader produced no batches")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    batches: Iterable[dict[str, object]],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        risk = model(batch["numerical"], batch["categorical"], batch["padding_mask"])
        losses.append(float(cox_ph_loss(risk, batch["time"], batch["event"]).cpu()))
    if not losses:
        raise ValueError("Evaluation loader produced no batches")
    return float(np.mean(losses))


@torch.no_grad()
def predict_risk(
    model: torch.nn.Module,
    batches: Iterable[dict[str, object]],
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    predictions: list[np.ndarray] = []
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        risk = model(batch["numerical"], batch["categorical"], batch["padding_mask"])
        sample_ids.extend(raw_batch["sample_ids"])
        predictions.append(risk.cpu().numpy())
    if not predictions:
        return sample_ids, np.empty(0, dtype=np.float32)
    return sample_ids, np.concatenate(predictions)
