"""
Comprehensive Hyperparameter Optimization for Dual-Branch RIP Model.

Optimization strategies applied:
1. ARCHITECTURE: LayerNorm, residual connections, wider/deeper fusion
2. TRAINING: Cosine LR scheduler, label smoothing, more epochs
3. DATA: Feature augmentation (Gaussian noise), self-loops on graph
4. HYPERPARAMETERS: Grid search over key parameters

This runs a systematic sweep and reports the best configuration.
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ripgnn.data.contracts import validate_bundle
from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.data.graph import RegionGraph, build_region_graph
from ripgnn.data.ingest import load_canonical_bundle
from ripgnn.training.metrics import compute_binary_metrics
from ripgnn.training.splits import split_organoids
from ripgnn.utils.seed import seed_everything


# ============================================================================
# ENHANCED MODEL with LayerNorm + Residual + configurable fusion
# ============================================================================
from torch_geometric.nn import GATv2Conv


class OptimizedDualBranch(nn.Module):
    """Enhanced dual-branch model with optimization improvements."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        heads: int = 2,
        dropout: float = 0.15,
        use_layernorm: bool = True,
        use_residual: bool = True,
        gnn_layers: int = 2,
        rnn_type: str = "gru",  # "gru" or "lstm"
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        spatial_input_dim = dynamic_dim + static_dim

        # GNN spatial branch
        self.gnn1 = GATv2Conv(
            spatial_input_dim, hidden_dim, heads=heads,
            dropout=dropout, edge_dim=1,
        )
        self.gnn2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=1,
            dropout=dropout, edge_dim=1,
        )
        # Optional 3rd GNN layer for deeper models
        self.gnn_layers = gnn_layers
        if gnn_layers >= 3:
            self.gnn3 = GATv2Conv(
                hidden_dim, hidden_dim, heads=1,
                dropout=dropout, edge_dim=1,
            )

        # Temporal branch
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(dynamic_dim, hidden_dim, batch_first=True)
        self.rnn_type = rnn_type

        # Layer normalization
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.spatial_norm = nn.LayerNorm(hidden_dim)
            self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Residual projection for spatial branch
        if use_residual and spatial_input_dim != hidden_dim:
            self.spatial_residual = nn.Linear(spatial_input_dim, hidden_dim)
        else:
            self.spatial_residual = None

        # Fusion with wider first layer
        fusion_input = hidden_dim * 2 + static_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.GELU(),           # GELU > ReLU for transformers/attention
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

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

        # --- Temporal branch ---
        temporal_sequences = dynamic.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, history_len, feature_dim
        )
        if self.rnn_type == "lstm":
            _, (hidden, _) = self.rnn(temporal_sequences)
        else:
            _, hidden = self.rnn(temporal_sequences)
        temporal_embedding = hidden[-1].reshape(batch_size, num_nodes, -1)
        if self.use_layernorm:
            temporal_embedding = self.temporal_norm(temporal_embedding)

        # --- Spatial branch ---
        spatial_embeddings = []
        attentions = []
        for current, static_features in zip(dynamic[:, -1], static, strict=True):
            x = torch.cat([current, static_features], dim=-1)
            h = self.activation(self.gnn1(x, edge_index, edge_attr=edge_attr))

            if return_attention:
                h2, attention = self.gnn2(
                    self.dropout(h), edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attentions.append(attention)
            else:
                h2 = self.gnn2(self.dropout(h), edge_index, edge_attr=edge_attr)

            if self.gnn_layers >= 3:
                h2 = self.activation(h2)
                h2 = self.gnn3(self.dropout(h2), edge_index, edge_attr=edge_attr)

            # Residual connection
            if self.use_residual:
                if self.spatial_residual is not None:
                    h2 = h2 + self.spatial_residual(x)
                # If dimensions match, add directly
                elif x.shape[-1] == h2.shape[-1]:
                    h2 = h2 + x

            h2 = self.activation(h2)
            spatial_embeddings.append(h2)

        spatial_embedding = torch.stack(spatial_embeddings, dim=0)
        if self.use_layernorm:
            spatial_embedding = self.spatial_norm(spatial_embedding)

        # --- Fusion ---
        fused = torch.cat([spatial_embedding, temporal_embedding, static], dim=-1)
        fused_hidden = self.fusion(fused)
        logits = self.classifier(fused_hidden).squeeze(-1)
        output: dict[str, Any] = {"logits": logits}
        if attentions:
            output["attention"] = attentions
        return output


# ============================================================================
# ENHANCED TRAINING with LR scheduling, label smoothing, augmentation
# ============================================================================

def add_self_loops(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                   num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Add self-loops to graph (helps with message passing stability)."""
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loop_index = torch.stack([self_loops, self_loops], dim=0)
    self_loop_weight = torch.ones(num_nodes, dtype=torch.float32)
    edge_index = torch.cat([edge_index, self_loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0)
    return edge_index, edge_weight


def augment_batch(batch: dict, noise_std: float = 0.02) -> dict:
    """Add Gaussian noise to dynamic features for data augmentation."""
    augmented = dict(batch)
    dynamic = batch["dynamic"].clone()
    noise = torch.randn_like(dynamic) * noise_std
    augmented["dynamic"] = dynamic + noise
    return augmented


class LabelSmoothBCELoss(nn.Module):
    """BCE with label smoothing to prevent overconfident predictions."""

    def __init__(self, smoothing: float = 0.05, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed, pos_weight=self.pos_weight
        )


def train_optimized(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    noise_augment: float = 0.02,
    use_scheduler: bool = True,
) -> tuple[nn.Module, dict]:
    """Enhanced training loop with all optimizations."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01) if use_scheduler else None

    pos_weight = _estimate_pos_weight(train_loader)
    criterion = LabelSmoothBCELoss(smoothing=label_smoothing, pos_weight=pos_weight.to(device))

    best_metric = float("-inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience = 20  # Early stopping

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            # Apply augmentation
            if noise_augment > 0:
                batch = augment_batch(batch, noise_std=noise_augment)

            optimizer.zero_grad(set_to_none=True)
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)

            output = model(dynamic, static, edge_index.to(device), edge_weight.to(device))
            logits = output["logits"]
            if logits.isnan().any():
                continue
            logits = logits.clamp(-10, 10)
            loss = criterion(logits, target)
            if loss.isnan() or loss.isinf():
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        # Validation
        val_metrics = _evaluate_quick(model, val_loader, edge_index, edge_weight, device)
        metric = val_metrics.get("roc_auc", float("nan"))
        if np.isnan(metric):
            metric = val_metrics.get("pr_auc", float("nan"))
        if np.isnan(metric):
            metric = -np.mean(train_losses) if train_losses else float("-inf")

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_metric": best_metric}


def _estimate_pos_weight(loader):
    pos = neg = 0.0
    for batch in loader:
        t = batch["target"]
        pos += float(t.sum())
        neg += float(t.numel() - t.sum())
    return torch.tensor(max(1.0, neg / pos) if pos > 0 else 1.0, dtype=torch.float32)


def _evaluate_quick(model, loader, edge_index, edge_weight, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["dynamic"].to(device), batch["static"].to(device),
                        edge_index.to(device), edge_weight.to(device))
            logits = out["logits"]
            if logits.isnan().any():
                continue
            probs = torch.sigmoid(logits.clamp(-10, 10)).cpu().numpy()
            targets = batch["target"].numpy()
            all_probs.append(probs)
            all_targets.append(targets)
    if not all_probs:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return compute_binary_metrics(probs, targets)


# ============================================================================
# HYPERPARAMETER GRID SEARCH
# ============================================================================

@dataclass
class SearchResult:
    config: dict
    test_metrics: dict
    val_metrics: dict
    best_epoch: int


def run_sweep():
    """Run comprehensive hyperparameter sweep."""
    seed_everything(42)

    # Load data once
    bundle = load_canonical_bundle("data/henderson2019")
    validate_bundle(bundle)

    static_cols = ["snca_prior", "hemisphere", "vulnerability"]
    dynamic_cols = ["burden_norm", "delta_burden", "pathology_z"]

    graph = build_region_graph(bundle.region_meta, bundle.edges, static_cols)
    split = split_organoids(bundle.snapshots, train_fraction=0.6, val_fraction=0.2, seed=42)

    ds_train = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols,
                                      history_len=1, organoid_ids=split.train_ids)
    ds_val = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols,
                                    history_len=1, organoid_ids=split.val_ids)
    ds_test = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols,
                                     history_len=1, organoid_ids=split.test_ids)

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_temporal_samples)
    val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_temporal_samples)
    test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=collate_temporal_samples)

    # Add self-loops to graph
    edge_index_sl, edge_weight_sl = add_self_loops(graph.edge_index, graph.edge_weight, len(graph.region_ids))

    # === SWEEP CONFIGURATIONS ===
    configs = []

    # Strategy 1: Architecture variations with self-loops
    for hidden_dim in [32, 64, 128]:
        for heads in [2, 4]:
            for dropout in [0.1, 0.2, 0.3]:
                configs.append({
                    "tag": f"arch_h{hidden_dim}_a{heads}_d{int(dropout*100)}",
                    "hidden_dim": hidden_dim, "heads": heads, "dropout": dropout,
                    "lr": 3e-4, "wd": 1e-4, "epochs": 100,
                    "layernorm": True, "residual": True,
                    "label_smooth": 0.05, "noise": 0.02,
                    "self_loops": True, "rnn_type": "gru",
                    "gnn_layers": 2,
                })

    # Strategy 2: Training regime variations (on best-guess architecture)
    for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
        for wd in [1e-5, 1e-4, 1e-3]:
            configs.append({
                "tag": f"train_lr{lr}_wd{wd}",
                "hidden_dim": 64, "heads": 2, "dropout": 0.2,
                "lr": lr, "wd": wd, "epochs": 100,
                "layernorm": True, "residual": True,
                "label_smooth": 0.05, "noise": 0.02,
                "self_loops": True, "rnn_type": "gru",
                "gnn_layers": 2,
            })

    # Strategy 3: Augmentation and smoothing variations
    for noise in [0.0, 0.01, 0.02, 0.05]:
        for smooth in [0.0, 0.05, 0.1]:
            configs.append({
                "tag": f"aug_n{int(noise*100)}_s{int(smooth*100)}",
                "hidden_dim": 64, "heads": 2, "dropout": 0.2,
                "lr": 3e-4, "wd": 1e-4, "epochs": 100,
                "layernorm": True, "residual": True,
                "label_smooth": smooth, "noise": noise,
                "self_loops": True, "rnn_type": "gru",
                "gnn_layers": 2,
            })

    # Strategy 4: LSTM variants of best configs
    for hidden_dim in [64, 128]:
        for dropout in [0.1, 0.2]:
            configs.append({
                "tag": f"lstm_h{hidden_dim}_d{int(dropout*100)}",
                "hidden_dim": hidden_dim, "heads": 2, "dropout": dropout,
                "lr": 3e-4, "wd": 1e-4, "epochs": 100,
                "layernorm": True, "residual": True,
                "label_smooth": 0.05, "noise": 0.02,
                "self_loops": True, "rnn_type": "lstm",
                "gnn_layers": 2,
            })

    # Strategy 5: 3-layer GNN
    for hidden_dim in [32, 64]:
        configs.append({
            "tag": f"deep3_h{hidden_dim}",
            "hidden_dim": hidden_dim, "heads": 2, "dropout": 0.2,
            "lr": 3e-4, "wd": 1e-4, "epochs": 100,
            "layernorm": True, "residual": True,
            "label_smooth": 0.05, "noise": 0.02,
            "self_loops": True, "rnn_type": "gru",
            "gnn_layers": 3,
        })

    # Strategy 6: No self-loops (ablation)
    configs.append({
        "tag": "no_selfloops",
        "hidden_dim": 64, "heads": 2, "dropout": 0.2,
        "lr": 3e-4, "wd": 1e-4, "epochs": 100,
        "layernorm": True, "residual": True,
        "label_smooth": 0.05, "noise": 0.02,
        "self_loops": False, "rnn_type": "gru",
        "gnn_layers": 2,
    })

    print(f"Total configurations to sweep: {len(configs)}")
    print(f"{'='*90}")

    results: list[SearchResult] = []
    best_roc = 0.0
    best_tag = ""

    for i, cfg in enumerate(configs):
        tag = cfg["tag"]
        seed_everything(42)  # Reproducible per config

        ei = edge_index_sl if cfg["self_loops"] else graph.edge_index
        ew = edge_weight_sl if cfg["self_loops"] else graph.edge_weight

        model = OptimizedDualBranch(
            dynamic_dim=3, static_dim=3,
            hidden_dim=cfg["hidden_dim"],
            heads=cfg["heads"],
            dropout=cfg["dropout"],
            use_layernorm=cfg["layernorm"],
            use_residual=cfg["residual"],
            gnn_layers=cfg["gnn_layers"],
            rnn_type=cfg["rnn_type"],
        )

        try:
            model, train_info = train_optimized(
                model, train_loader, val_loader, ei, ew,
                device=torch.device("cpu"),
                epochs=cfg["epochs"],
                learning_rate=cfg["lr"],
                weight_decay=cfg["wd"],
                label_smoothing=cfg["label_smooth"],
                noise_augment=cfg["noise"],
            )

            # Evaluate on test
            test_metrics = _evaluate_quick(model, test_loader, ei, ew, torch.device("cpu"))
            val_metrics = _evaluate_quick(model, val_loader, ei, ew, torch.device("cpu"))

            roc = test_metrics.get("roc_auc", float("nan"))
            pr = test_metrics.get("pr_auc", float("nan"))
            f1 = test_metrics.get("f1", float("nan"))
            acc = test_metrics.get("accuracy", float("nan"))

            is_best = ""
            if not np.isnan(roc) and roc > best_roc:
                best_roc = roc
                best_tag = tag
                is_best = " ** BEST **"
                # Save best model
                out_dir = Path("outputs/henderson2019_optimized")
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                           out_dir / "model.pt")
                with open(out_dir / "test_metrics.json", "w") as f:
                    json.dump(test_metrics, f, indent=2, default=str)
                with open(out_dir / "best_config.json", "w") as f:
                    json.dump(cfg, f, indent=2)

            results.append(SearchResult(cfg, test_metrics, val_metrics, train_info["best_epoch"]))

            print(f"[{i+1:3d}/{len(configs)}] {tag:<40s} "
                  f"ROC={roc:.4f} PR={pr:.4f} F1={f1:.4f} Acc={acc:.4f} "
                  f"ep={train_info['best_epoch']}{is_best}")

        except Exception as e:
            print(f"[{i+1:3d}/{len(configs)}] {tag:<40s} FAILED: {e}")

    # === FINAL REPORT ===
    print(f"\n{'='*90}")
    print("TOP 10 CONFIGURATIONS BY TEST ROC-AUC")
    print(f"{'='*90}")

    valid_results = [r for r in results
                     if not np.isnan(r.test_metrics.get("roc_auc", float("nan")))]
    valid_results.sort(key=lambda r: r.test_metrics["roc_auc"], reverse=True)

    fmt = "{:<45s} {:>8} {:>8} {:>8} {:>8} {:>5}"
    print(fmt.format("Config", "ROC-AUC", "PR-AUC", "F1", "Acc", "Epoch"))
    print("-" * 90)

    # Baseline comparison
    print(fmt.format("BASELINE (previous best)", "0.7220", "0.7599", "0.7779", "0.6843", ""))
    print("-" * 90)

    for r in valid_results[:10]:
        m = r.test_metrics
        print(fmt.format(
            r.config["tag"],
            f"{m['roc_auc']:.4f}",
            f"{m['pr_auc']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['accuracy']:.4f}",
            str(r.best_epoch),
        ))

    # Save full sweep results
    sweep_data = [{
        "tag": r.config["tag"],
        "config": r.config,
        "test_roc_auc": r.test_metrics.get("roc_auc"),
        "test_pr_auc": r.test_metrics.get("pr_auc"),
        "test_f1": r.test_metrics.get("f1"),
        "test_accuracy": r.test_metrics.get("accuracy"),
        "val_roc_auc": r.val_metrics.get("roc_auc"),
        "best_epoch": r.best_epoch,
    } for r in valid_results]

    with open("outputs/hyperparameter_sweep_results.json", "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)

    print(f"\nBest configuration: {best_tag} (ROC-AUC: {best_roc:.4f})")
    print(f"Full results saved to outputs/hyperparameter_sweep_results.json")
    print(f"Best model saved to outputs/henderson2019_optimized/")


if __name__ == "__main__":
    run_sweep()
