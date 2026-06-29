"""Debug the NaN loss."""
import torch
import torch.nn as nn
from ripgnn.data.ingest import load_canonical_bundle
from ripgnn.data.contracts import validate_bundle
from ripgnn.data.graph import build_region_graph
from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.training.splits import split_organoids
from ripgnn.cli import build_model, load_yaml
from torch.utils.data import DataLoader

config = load_yaml("configs/henderson2019_dual_branch.yaml")
data_cfg = config["data"]
bundle = load_canonical_bundle(data_cfg["artifact_dir"])
validate_bundle(bundle)
graph = build_region_graph(bundle.region_meta, bundle.edges, data_cfg["static_features"])
split = split_organoids(bundle.snapshots, train_fraction=0.6, val_fraction=0.2, seed=42)

ds = TemporalRegionDataset(bundle.snapshots, graph, data_cfg["dynamic_features"],
                           history_len=1, organoid_ids=split.train_ids)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_temporal_samples)

model = build_model("dual_branch", 3, 3, config["model"])

batch = next(iter(loader))
model.eval()
with torch.no_grad():
    out = model(batch["dynamic"], batch["static"], graph.edge_index, graph.edge_weight)
    logits = out["logits"]
    print(f"Logits shape: {logits.shape}")
    print(f"Logits NaN: {logits.isnan().sum()}")
    print(f"Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
    print(f"Logits mean: {logits.mean():.6f}")
    print(f"Logits std: {logits.std():.6f}")

    target = batch["target"]
    print(f"Target shape: {target.shape}")
    print(f"Target sum: {target.sum()} / {target.numel()}")

    # Manual BCE check
    loss1 = nn.BCEWithLogitsLoss()(logits, target)
    print(f"BCE loss (no weight): {loss1.item()}")

    loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))(logits, target)
    print(f"BCE loss (weight=1.0): {loss2.item()}")

# Now try training mode
model.train()
out = model(batch["dynamic"], batch["static"], graph.edge_index, graph.edge_weight)
logits = out["logits"]
print(f"\nTraining mode logits NaN: {logits.isnan().sum()}")
print(f"Training mode logits range: [{logits.min():.6f}, {logits.max():.6f}]")
loss = nn.BCEWithLogitsLoss()(logits, target)
print(f"Training mode BCE loss: {loss.item()}")
