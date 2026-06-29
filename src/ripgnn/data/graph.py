"""Graph construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class RegionGraph:
    region_ids: list[str]
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    static_features: torch.Tensor


def build_region_graph(
    region_meta: pd.DataFrame,
    edges: pd.DataFrame,
    static_feature_cols: Sequence[str],
    topology_weight: float = 0.7,
    similarity_weight: float = 0.3,
) -> RegionGraph:
    """Build a weighted static graph from canonical artifacts."""
    region_ids = list(region_meta["region_id"])
    region_to_idx = {region: idx for idx, region in enumerate(region_ids)}
    features = region_meta.loc[:, list(static_feature_cols)].to_numpy(dtype=np.float32)
    similarity = _cosine_similarity(features)
    edge_rows: list[list[int]] = []
    edge_weights: list[float] = []

    for row in edges.itertuples(index=False):
        src = region_to_idx[row.src_region]
        dst = region_to_idx[row.dst_region]
        sim = float(similarity[src, dst])
        combined = topology_weight * float(row.edge_weight) + similarity_weight * sim
        edge_rows.append([src, dst])
        edge_weights.append(combined)

    edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    static_features = torch.tensor(features, dtype=torch.float32)
    return RegionGraph(
        region_ids=region_ids,
        edge_index=edge_index,
        edge_weight=edge_weight,
        static_features=static_features,
    )


def _cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for row vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    return np.clip(similarity, 0.0, 1.0)

