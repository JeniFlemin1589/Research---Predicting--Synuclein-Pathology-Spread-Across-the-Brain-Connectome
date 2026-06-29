"""Interpretation utilities for RIP models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ripgnn.utils.io import ensure_dir


def compute_attention_hubs(
    model: torch.nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    region_ids: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """Aggregate attention weights into a node-level hub ranking."""
    model.eval()
    aggregated: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["dynamic"].to(device),
                batch["static"].to(device),
                edge_index.to(device),
                edge_weight.to(device),
                return_attention=True,
            )
            if "attention" not in output:
                break
            for attention_edge_index, alpha in output["attention"]:
                edge_idx = attention_edge_index.cpu().numpy()
                attn = alpha.mean(dim=-1).cpu().numpy()
                node_scores = np.zeros(len(region_ids), dtype=float)
                for edge_pos in range(edge_idx.shape[1]):
                    src = edge_idx[0, edge_pos]
                    dst = edge_idx[1, edge_pos]
                    node_scores[src] += float(attn[edge_pos])
                    node_scores[dst] += float(attn[edge_pos])
                aggregated.append(node_scores)
    if not aggregated:
        return pd.DataFrame({"region_id": region_ids, "attention_score": np.zeros(len(region_ids))})
    mean_scores = np.mean(np.stack(aggregated, axis=0), axis=0)
    return (
        pd.DataFrame({"region_id": region_ids, "attention_score": mean_scores})
        .sort_values("attention_score", ascending=False)
        .reset_index(drop=True)
    )


def compute_occlusion_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    region_ids: list[str],
    device: torch.device,
    max_batches: int = 2,
) -> pd.DataFrame:
    """Rank nodes by prediction drop when their temporal features are zeroed out."""
    model.eval()
    drops = np.zeros(len(region_ids), dtype=float)
    batches_seen = 0
    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            baseline = torch.sigmoid(
                model(dynamic, static, edge_index.to(device), edge_weight.to(device))["logits"]
            ).mean().item()
            for node_idx in range(len(region_ids)):
                ablated = dynamic.clone()
                ablated[:, :, node_idx, :] = 0.0
                occluded = torch.sigmoid(
                    model(ablated, static, edge_index.to(device), edge_weight.to(device))["logits"]
                ).mean().item()
                drops[node_idx] += baseline - occluded
            batches_seen += 1
            if batches_seen >= max_batches:
                break
    if batches_seen:
        drops /= batches_seen
    return (
        pd.DataFrame({"region_id": region_ids, "occlusion_drop": drops})
        .sort_values("occlusion_drop", ascending=False)
        .reset_index(drop=True)
    )


def write_interpretation_tables(
    attention_hubs: pd.DataFrame,
    occlusion_scores: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write interpretation tables to disk."""
    out_dir = ensure_dir(output_dir)
    attention_hubs.to_csv(out_dir / "attention_hubs.csv", index=False)
    occlusion_scores.to_csv(out_dir / "occlusion_scores.csv", index=False)
