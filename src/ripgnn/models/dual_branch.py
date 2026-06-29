"""Dual-branch GATv2 + GRU RIP model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class DualBranchRIPModel(nn.Module):
    """Fuse spatial graph reasoning with temporal kinetic encoding."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        heads: int = 2,
        dropout: float = 0.15,
        regression_head: bool = False,
    ) -> None:
        super().__init__()
        self.regression_head_enabled = regression_head
        spatial_input_dim = dynamic_dim + static_dim
        self.gnn1 = GATv2Conv(
            spatial_input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
        )
        self.gnn2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            edge_dim=1,
        )
        self.gru = nn.GRU(dynamic_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        if regression_head:
            self.regressor = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        edge_attr = edge_weight.unsqueeze(-1)
        batch_size, history_len, num_nodes, feature_dim = dynamic.shape
        temporal_sequences = dynamic.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, history_len, feature_dim
        )
        _, hidden = self.gru(temporal_sequences)
        temporal_embedding = hidden[-1].reshape(batch_size, num_nodes, -1)

        spatial_embeddings = []
        attentions = []
        for current, static_features in zip(dynamic[:, -1], static, strict=True):
            x = torch.cat([current, static_features], dim=-1)
            h = self.activation(self.gnn1(x, edge_index, edge_attr=edge_attr))
            if return_attention:
                h2, attention = self.gnn2(
                    self.dropout(h),
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attentions.append(attention)
            else:
                h2 = self.gnn2(self.dropout(h), edge_index, edge_attr=edge_attr)
            spatial_embeddings.append(self.activation(h2))
        spatial_embedding = torch.stack(spatial_embeddings, dim=0)

        fused = torch.cat([spatial_embedding, temporal_embedding, static], dim=-1)
        fused_hidden = self.fusion(fused)
        logits = self.classifier(fused_hidden).squeeze(-1)
        output: dict[str, Any] = {"logits": logits}
        if self.regression_head_enabled:
            output["burden"] = self.regressor(fused_hidden).squeeze(-1)
        if attentions:
            output["attention"] = attentions
        return output
