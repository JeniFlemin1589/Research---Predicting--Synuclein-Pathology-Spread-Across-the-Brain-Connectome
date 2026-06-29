"""Command line interface for RIP-GNN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ripgnn.analysis.figures import plot_hub_ranking
from ripgnn.analysis.interpret import (
    compute_attention_hubs,
    compute_occlusion_scores,
    write_interpretation_tables,
)
from ripgnn.config import load_yaml, save_yaml
from ripgnn.data.contracts import validate_bundle
from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.data.graph import build_region_graph
from ripgnn.data.ingest import canonicalize_measurement_table, load_canonical_bundle
from ripgnn.data.synthetic import write_demo_bundle
from ripgnn.models.baselines import (
    GRUOnlyBaseline,
    LSTMOnlyBaseline,
    LogisticRegressionBaseline,
    NodeMLPBaseline,
    SpatialGNNBaseline,
)
from ripgnn.models.dual_branch import DualBranchRIPModel
from ripgnn.models.dual_branch_lstm import DualBranchLSTMModel
from ripgnn.resources import DATASET_REGISTRY
from ripgnn.training.engine import evaluate_model, save_evaluation_artifacts, train_model
from ripgnn.training.splits import assert_split_integrity, split_organoids
from ripgnn.utils.device import resolve_device
from ripgnn.utils.io import ensure_dir
from ripgnn.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(description="RIP-GNN command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-demo", help="Generate a synthetic demo bundle.")
    bootstrap.add_argument("--output-dir", default="data/demo")
    bootstrap.add_argument("--seed", type=int, default=7)

    audit = subparsers.add_parser("audit-data", help="Validate a canonical data bundle.")
    audit.add_argument("--data-dir", required=True)

    subparsers.add_parser("list-resources", help="Print the public resource registry.")

    canonicalize = subparsers.add_parser(
        "canonicalize", help="Convert a processed measurements table into canonical artifacts."
    )
    canonicalize.add_argument("--measurements", required=True)
    canonicalize.add_argument("--region-meta", required=True)
    canonicalize.add_argument("--edges", required=True)
    canonicalize.add_argument("--column-map", required=True, help="Path to YAML mapping for raw columns.")
    canonicalize.add_argument("--output-dir", required=True)

    train = subparsers.add_parser("train", help="Train a model from a YAML config.")
    train.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved run.")
    evaluate.add_argument("--run-dir", required=True)

    interpret = subparsers.add_parser("interpret", help="Run attention and occlusion analysis for a saved run.")
    interpret.add_argument("--run-dir", required=True)

    return parser


def build_model(
    model_name: str,
    dynamic_dim: int,
    static_dim: int,
    model_cfg: dict[str, Any],
) -> torch.nn.Module:
    """Instantiate a model from config."""
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    heads = int(model_cfg.get("heads", 2))
    dropout = float(model_cfg.get("dropout", 0.15))
    if model_name == "logistic_regression":
        return LogisticRegressionBaseline(dynamic_dim, static_dim)
    if model_name == "mlp":
        return NodeMLPBaseline(dynamic_dim, static_dim, hidden_dim=hidden_dim)
    if model_name == "gru_only":
        return GRUOnlyBaseline(dynamic_dim, hidden_dim=hidden_dim)
    if model_name == "lstm_only":
        return LSTMOnlyBaseline(dynamic_dim, hidden_dim=hidden_dim)
    if model_name == "gnn_only":
        return SpatialGNNBaseline(dynamic_dim, static_dim, hidden_dim=hidden_dim, heads=heads, dropout=dropout)
    if model_name == "dual_branch":
        return DualBranchRIPModel(
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            regression_head=bool(model_cfg.get("regression_head", False)),
        )
    if model_name == "dual_branch_lstm":
        return DualBranchLSTMModel(
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            regression_head=bool(model_cfg.get("regression_head", False)),
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_dataloaders(config: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Load canonical artifacts, split organoids, and build data loaders."""
    data_cfg = config["data"]
    bundle = load_canonical_bundle(data_cfg["artifact_dir"])
    validate_bundle(bundle)
    static_cols = data_cfg["static_features"]
    graph = build_region_graph(bundle.region_meta, bundle.edges, static_cols)
    split = split_organoids(
        bundle.snapshots,
        train_fraction=float(data_cfg["split"]["train_fraction"]),
        val_fraction=float(data_cfg["split"]["val_fraction"]),
        seed=int(config["project"]["seed"]),
    )
    assert_split_integrity(split)
    history_len = int(data_cfg["history_len"])
    dynamic_cols = data_cfg["dynamic_features"]
    train_dataset = TemporalRegionDataset(
        bundle.snapshots,
        graph=graph,
        dynamic_feature_cols=dynamic_cols,
        history_len=history_len,
        organoid_ids=split.train_ids,
    )
    val_dataset = TemporalRegionDataset(
        bundle.snapshots,
        graph=graph,
        dynamic_feature_cols=dynamic_cols,
        history_len=history_len,
        organoid_ids=split.val_ids,
    )
    test_dataset = TemporalRegionDataset(
        bundle.snapshots,
        graph=graph,
        dynamic_feature_cols=dynamic_cols,
        history_len=history_len,
        organoid_ids=split.test_ids,
    )
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_temporal_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_samples,
    )
    return bundle, graph, split, (train_loader, val_loader, test_loader)


def command_bootstrap_demo(args: argparse.Namespace) -> None:
    """Generate the demo bundle."""
    paths = write_demo_bundle(args.output_dir, seed=args.seed)
    for name, path in paths.items():
        print(f"{name}: {path}")


def command_audit_data(args: argparse.Namespace) -> None:
    """Validate a canonical bundle and print summary stats."""
    bundle = load_canonical_bundle(args.data_dir)
    print("snapshots rows:", len(bundle.snapshots))
    print("edges rows:", len(bundle.edges))
    print("region_meta rows:", len(bundle.region_meta))
    print("organoids:", bundle.snapshots["organoid_id"].nunique())
    print("regions:", bundle.snapshots["region_id"].nunique())
    print("timepoints:", bundle.snapshots["time_idx"].nunique())
    print("positive rate:", round(float(bundle.snapshots["target_positive"].mean()), 4))


def command_list_resources(args: argparse.Namespace) -> None:
    """Print the verified public resource registry."""
    for resource in DATASET_REGISTRY:
        print(f"- {resource.name}")
        print(f"  URL: {resource.url}")
        print(f"  Purpose: {resource.purpose}")
        print(f"  Notes: {resource.notes}")


def command_canonicalize(args: argparse.Namespace) -> None:
    """Canonicalize a processed measurement table."""
    mapping = load_yaml(args.column_map)
    bundle = canonicalize_measurement_table(
        measurements_path=args.measurements,
        region_meta_path=args.region_meta,
        edges_path=args.edges,
        column_map=mapping,
    )
    out_dir = ensure_dir(args.output_dir)
    bundle.snapshots.to_csv(out_dir / "snapshots.csv", index=False)
    bundle.edges.to_csv(out_dir / "edges.csv", index=False)
    bundle.region_meta.to_csv(out_dir / "region_meta.csv", index=False)
    print(f"Wrote canonical bundle to {out_dir}")


def command_train(args: argparse.Namespace) -> None:
    """Train a configured model and save run artifacts."""
    config = load_yaml(args.config)
    seed_everything(int(config["project"]["seed"]))
    device = resolve_device(config["project"].get("device"))
    bundle, graph, split, loaders = make_dataloaders(config)
    train_loader, val_loader, test_loader = loaders
    dynamic_dim = len(config["data"]["dynamic_features"])
    static_dim = len(config["data"]["static_features"])
    model = build_model(config["model"]["name"], dynamic_dim, static_dim, config["model"])
    output_dir = ensure_dir(config["project"]["output_dir"])
    save_yaml(config, output_dir / "resolved_config.yaml")
    pd.DataFrame(
        {
            "split": ["train"] * len(split.train_ids) + ["val"] * len(split.val_ids) + ["test"] * len(split.test_ids),
            "organoid_id": split.train_ids + split.val_ids + split.test_ids,
        }
    ).to_csv(output_dir / "split_manifest.csv", index=False)
    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        output_dir=output_dir,
        device=device,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        regression_weight=float(config["training"].get("regression_weight", 0.0)),
    )
    val_eval = evaluate_model(model, val_loader, graph.edge_index, graph.edge_weight, device)
    test_eval = evaluate_model(model, test_loader, graph.edge_index, graph.edge_weight, device)
    save_evaluation_artifacts(val_eval, graph.region_ids, output_dir, "val")
    save_evaluation_artifacts(test_eval, graph.region_ids, output_dir, "test")
    pd.DataFrame({"region_id": graph.region_ids}).to_csv(output_dir / "regions.csv", index=False)
    print(f"Training complete. Outputs written to {output_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    """Reload a run config, model, and evaluate again."""
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    device = resolve_device(config["project"].get("device"))
    _, graph, _, loaders = make_dataloaders(config)
    _, _, test_loader = loaders
    dynamic_dim = len(config["data"]["dynamic_features"])
    static_dim = len(config["data"]["static_features"])
    model = build_model(config["model"]["name"], dynamic_dim, static_dim, config["model"])
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    test_eval = evaluate_model(model, test_loader, graph.edge_index, graph.edge_weight, device)
    save_evaluation_artifacts(test_eval, graph.region_ids, run_dir, "test_reloaded")
    print(test_eval["metrics"])


def command_interpret(args: argparse.Namespace) -> None:
    """Generate attention and occlusion analyses for a trained run."""
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    device = resolve_device(config["project"].get("device"))
    _, graph, _, loaders = make_dataloaders(config)
    _, _, test_loader = loaders
    dynamic_dim = len(config["data"]["dynamic_features"])
    static_dim = len(config["data"]["static_features"])
    model = build_model(config["model"]["name"], dynamic_dim, static_dim, config["model"])
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    attention_hubs = compute_attention_hubs(
        model, test_loader, graph.edge_index, graph.edge_weight, graph.region_ids, device
    )
    occlusion_scores = compute_occlusion_scores(
        model, test_loader, graph.edge_index, graph.edge_weight, graph.region_ids, device
    )
    write_interpretation_tables(attention_hubs, occlusion_scores, run_dir)
    plot_hub_ranking(
        attention_hubs,
        score_col="attention_score",
        path=run_dir / "attention_hubs.png",
        title="Attention-based hub ranking",
    )
    plot_hub_ranking(
        occlusion_scores,
        score_col="occlusion_drop",
        path=run_dir / "occlusion_hubs.png",
        title="Occlusion sensitivity ranking",
    )
    print(f"Interpretation artifacts written to {run_dir}")


def main() -> None:
    """Dispatch CLI commands."""
    parser = build_parser()
    args = parser.parse_args()
    command_map = {
        "bootstrap-demo": command_bootstrap_demo,
        "audit-data": command_audit_data,
        "list-resources": command_list_resources,
        "canonicalize": command_canonicalize,
        "train": command_train,
        "evaluate": command_evaluate,
        "interpret": command_interpret,
    }
    command_map[args.command](args)


if __name__ == "__main__":
    main()
