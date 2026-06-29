"""Training and evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ripgnn.analysis.figures import plot_calibration_curve, plot_probability_heatmap
from ripgnn.training.metrics import compute_binary_metrics
from ripgnn.utils.io import ensure_dir, write_json


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_roc_auc: float
    val_pr_auc: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    output_dir: str | Path,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    regression_weight: float = 0.0,
) -> tuple[nn.Module, list[EpochResult]]:
    """Train a model with BCE objective and optional regression auxiliary loss."""
    ensure_dir(output_dir)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = estimate_pos_weight(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    mse = nn.MSELoss()
    best_metric = float("-inf")
    best_state: dict[str, Any] | None = None
    history: list[EpochResult] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)
            burden_target = batch["burden_target"].to(device)
            output = model(dynamic, static, edge_index.to(device), edge_weight.to(device))
            logits = output["logits"]
            # Safety: clamp logits and skip NaN batches
            if logits.isnan().any():
                continue
            logits = logits.clamp(-10, 10)
            loss = criterion(logits, target)
            if regression_weight and "burden" in output:
                loss = loss + regression_weight * mse(output["burden"], burden_target)
            if loss.isnan() or loss.isinf():
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_eval = evaluate_model(model, val_loader, edge_index, edge_weight, device)
        result = EpochResult(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=val_eval["loss"],
            val_roc_auc=val_eval["metrics"]["roc_auc"],
            val_pr_auc=val_eval["metrics"]["pr_auc"],
        )
        history.append(result)
        metric = val_eval["metrics"]["roc_auc"]
        if np.isnan(metric):
            metric = val_eval["metrics"]["pr_auc"]
        if np.isnan(metric):
            # Both metrics NaN — use negative train loss as proxy
            metric = -float(np.mean(train_losses)) if train_losses else float("-inf")
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save(best_state, Path(output_dir) / "model.pt")
    pd.DataFrame([item.__dict__ for item in history]).to_csv(
        Path(output_dir) / "history.csv", index=False
    )
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a model and collect probability outputs."""
    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    all_probabilities = []
    all_targets = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)
            output = model(dynamic, static, edge_index.to(device), edge_weight.to(device))
            loss = criterion(output["logits"], target)
            losses.append(float(loss.detach().cpu()))
            probabilities = torch.sigmoid(output["logits"]).cpu().numpy()
            targets = target.cpu().numpy()
            all_probabilities.append(probabilities)
            all_targets.append(targets)
            for row_idx, organoid_id in enumerate(batch["organoid_id"]):
                rows.append(
                    {
                        "organoid_id": organoid_id,
                        "input_end_time": batch["input_end_time"][row_idx],
                        "target_time": batch["target_time"][row_idx],
                        "mean_probability": float(np.mean(probabilities[row_idx])),
                        "mean_target": float(np.mean(targets[row_idx])),
                    }
                )
    probabilities = np.concatenate(all_probabilities, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_binary_metrics(probabilities, targets)
    return {
        "loss": float(np.mean(losses)),
        "metrics": metrics,
        "probabilities": probabilities,
        "targets": targets,
        "rows": pd.DataFrame(rows),
    }


def save_evaluation_artifacts(
    evaluation: dict[str, Any],
    region_ids: list[str],
    output_dir: str | Path,
    prefix: str,
) -> None:
    """Persist metrics, summary predictions, and quick-look figures."""
    out_dir = ensure_dir(output_dir)
    write_json(evaluation["metrics"], out_dir / f"{prefix}_metrics.json")
    evaluation["rows"].to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    plot_calibration_curve(
        evaluation["metrics"]["calibration"]["prob_pred"],
        evaluation["metrics"]["calibration"]["prob_true"],
        out_dir / f"{prefix}_calibration.png",
    )
    plot_probability_heatmap(
        evaluation["probabilities"],
        region_ids,
        out_dir / f"{prefix}_rip_heatmap.png",
    )


def estimate_pos_weight(loader: DataLoader) -> torch.Tensor:
    """Estimate class balancing weight from the training data."""
    positives = 0.0
    negatives = 0.0
    for batch in loader:
        target = batch["target"]
        positives += float(target.sum())
        negatives += float(target.numel() - target.sum())
    if positives == 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)

