from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.data.graph import build_region_graph
from ripgnn.data.synthetic import build_edges, build_region_meta, build_snapshots
from ripgnn.models.dual_branch import DualBranchRIPModel


def test_dual_branch_forward_pass() -> None:
    region_meta = build_region_meta()
    edges = build_edges()
    snapshots = build_snapshots()
    static_cols = [
        "hnoca_cortex_prior",
        "hnoca_striatum_prior",
        "hnoca_midbrain_prior",
        "hnoca_hindbrain_prior",
    ]
    graph = build_region_graph(region_meta, edges, static_cols)
    dataset = TemporalRegionDataset(
        snapshots=snapshots,
        graph=graph,
        dynamic_feature_cols=["burden_norm", "delta_burden", "pathology_z"],
        history_len=3,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_temporal_samples)
    batch = next(iter(loader))
    model = DualBranchRIPModel(dynamic_dim=3, static_dim=4, hidden_dim=16, heads=2, dropout=0.1)
    output = model(batch["dynamic"], batch["static"], graph.edge_index, graph.edge_weight)
    assert "logits" in output
    assert output["logits"].shape == batch["target"].shape
    assert torch.isfinite(output["logits"]).all()
