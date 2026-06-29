"""Baseline models for RIP prediction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class LogisticRegressionBaseline(nn.Module):
    """Linear baseline on the current snapshot plus static priors."""

    def __init__(self, dynamic_dim: int, static_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dynamic_dim + static_dim, 1)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        current = torch.cat([dynamic[:, -1], static], dim=-1)
        logits = self.linear(current).squeeze(-1)
        return {"logits": logits}


class NodeMLPBaseline(nn.Module):
    """Node-wise MLP baseline using only current features."""

    def __init__(self, dynamic_dim: int, static_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        current = torch.cat([dynamic[:, -1], static], dim=-1)
        logits = self.network(current).squeeze(-1)
        return {"logits": logits}


class GRUOnlyBaseline(nn.Module):
    """Temporal baseline with per-node GRU encoding."""

    def __init__(self, dynamic_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(dynamic_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        batch_size, history_len, num_nodes, feature_dim = dynamic.shape
        sequences = dynamic.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, history_len, feature_dim)
        _, hidden = self.gru(sequences)
        logits = self.head(hidden[-1]).reshape(batch_size, num_nodes)
        return {"logits": logits}


class SpatialGNNBaseline(nn.Module):
    """Graph-only baseline on the latest snapshot."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 48,
        heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = dynamic_dim + static_dim
        self.gnn1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        self.gnn2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        edge_attr = edge_weight.unsqueeze(-1)
        logits_by_sample = []
        attentions = []
        for current, static_features in zip(dynamic[:, -1], static, strict=True):
            x = torch.cat([current, static_features], dim=-1)
            h = self.gnn1(x, edge_index, edge_attr=edge_attr)
            h = self.activation(h)
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
            logits_by_sample.append(self.head(self.activation(h2)).squeeze(-1))
        output: dict[str, Any] = {"logits": torch.stack(logits_by_sample, dim=0)}
        if attentions:
            output["attention"] = attentions
        return output


class LSTMOnlyBaseline(nn.Module):
    """Temporal baseline with per-node LSTM encoding (no graph)."""

    def __init__(self, dynamic_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.lstm = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        batch_size, history_len, num_nodes, feature_dim = dynamic.shape
        sequences = dynamic.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, history_len, feature_dim)
        _, (hidden, _cell) = self.lstm(sequences)
        logits = self.head(hidden[-1]).reshape(batch_size, num_nodes)
        return {"logits": logits}


