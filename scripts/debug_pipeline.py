"""Debug the pipeline to find NaN issues."""
from ripgnn.data.ingest import load_canonical_bundle
from ripgnn.data.graph import build_region_graph
from ripgnn.data.dataset import TemporalRegionDataset
from ripgnn.training.splits import split_organoids
import numpy as np
import torch

bundle = load_canonical_bundle("data/henderson2019")
graph = build_region_graph(
    bundle.region_meta, bundle.edges,
    ["snca_prior", "hemisphere", "vulnerability"]
)

split = split_organoids(bundle.snapshots, train_fraction=0.6, val_fraction=0.2, seed=42)
print(f"Train: {split.train_ids}")
print(f"Val: {split.val_ids}")
print(f"Test: {split.test_ids}")

ds_train = TemporalRegionDataset(
    bundle.snapshots, graph,
    ["burden_norm", "delta_burden", "pathology_z"],
    history_len=1, organoid_ids=split.train_ids,
)
ds_val = TemporalRegionDataset(
    bundle.snapshots, graph,
    ["burden_norm", "delta_burden", "pathology_z"],
    history_len=1, organoid_ids=split.val_ids,
)
ds_test = TemporalRegionDataset(
    bundle.snapshots, graph,
    ["burden_norm", "delta_burden", "pathology_z"],
    history_len=1, organoid_ids=split.test_ids,
)
print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

if len(ds_train) > 0:
    s = ds_train[0]
    print(f"Dynamic shape: {s.dynamic.shape}")
    print(f"Dynamic NaN: {s.dynamic.isnan().sum().item()}")
    print(f"Target mean: {s.target.mean():.4f}")
    print(f"Target NaN: {s.target.isnan().sum().item()}")

    from ripgnn.models import build_model
    m = build_model("dual_branch", 3, 3, {
        "hidden_dim": 64, "heads": 2, "dropout": 0.15, "regression_head": False,
    })
    m.eval()
    with torch.no_grad():
        out = m(
            s.dynamic.unsqueeze(0), s.static.unsqueeze(0),
            graph.edge_index, graph.edge_weight,
        )
        logits = out["logits"]
        print(f"Logits shape: {logits.shape}")
        print(f"Logits NaN: {logits.isnan().sum().item()}")
        print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
else:
    print("NO TRAIN SAMPLES!")
