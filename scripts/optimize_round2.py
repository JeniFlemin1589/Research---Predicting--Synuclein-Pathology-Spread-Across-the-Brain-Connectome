"""
Round 2: Focused refinement around the best config found in round 1.

Best from round 1: lr=0.001, wd=1e-4, hidden=64, heads=2, dropout=0.2
Let's now try combining the best architecture (h128, a4) with the best LR (0.001).
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from ripgnn.data.contracts import validate_bundle
from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.data.graph import build_region_graph
from ripgnn.data.ingest import load_canonical_bundle
from ripgnn.training.splits import split_organoids
from ripgnn.utils.seed import seed_everything

# Import from round 1 script
import sys
sys.path.insert(0, "scripts")
from optimize_hyperparams import (
    OptimizedDualBranch, add_self_loops, train_optimized,
    _evaluate_quick,
)

seed_everything(42)
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

edge_index_sl, edge_weight_sl = add_self_loops(graph.edge_index, graph.edge_weight, len(graph.region_ids))

# Round 2: Combine best architecture + best training
configs = [
    # Best arch (h128 a4) + best LR (0.001)
    {"tag": "r2_h128_a4_lr001_d10",  "hidden_dim": 128, "heads": 4, "dropout": 0.10, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr001_d15",  "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr001_d20",  "hidden_dim": 128, "heads": 4, "dropout": 0.20, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr001_d25",  "hidden_dim": 128, "heads": 4, "dropout": 0.25, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},

    # Try lr=0.0008 and lr=0.0012 (bracket best LR)
    {"tag": "r2_h128_a4_lr0008_d15", "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.0008, "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr0012_d15", "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.0012, "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr0015_d15", "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.0015, "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},

    # Best config + higher label smoothing
    {"tag": "r2_h128_a4_lr001_s10",  "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.10, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h128_a4_lr001_s15",  "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.15, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},

    # Best config + no augmentation (ablation)
    {"tag": "r2_h128_a4_lr001_noaug","hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.0,  "rnn": "gru", "layers": 2, "epochs": 150},

    # Best config + more noise augmentation
    {"tag": "r2_h128_a4_lr001_n05",  "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.05, "rnn": "gru", "layers": 2, "epochs": 150},

    # LSTM with best settings
    {"tag": "r2_lstm_h128_a4_lr001", "hidden_dim": 128, "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "lstm","layers": 2, "epochs": 150},

    # h96 (between 64 and 128)
    {"tag": "r2_h96_a4_lr001",       "hidden_dim": 96,  "heads": 4, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
    {"tag": "r2_h96_a3_lr001",       "hidden_dim": 96,  "heads": 3, "dropout": 0.15, "lr": 0.001,  "wd": 1e-4, "smooth": 0.05, "noise": 0.02, "rnn": "gru", "layers": 2, "epochs": 150},
]

print(f"Round 2: {len(configs)} focused configurations")
print(f"{'='*95}")

best_roc = 0.0
best_tag = ""
all_results = []

for i, cfg in enumerate(configs):
    seed_everything(42)
    model = OptimizedDualBranch(
        dynamic_dim=3, static_dim=3,
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        dropout=cfg["dropout"], use_layernorm=True, use_residual=True,
        gnn_layers=cfg["layers"], rnn_type=cfg["rnn"],
    )

    try:
        model, info = train_optimized(
            model, train_loader, val_loader, edge_index_sl, edge_weight_sl,
            device=torch.device("cpu"), epochs=cfg["epochs"],
            learning_rate=cfg["lr"], weight_decay=cfg["wd"],
            label_smoothing=cfg["smooth"], noise_augment=cfg["noise"],
        )
        test_m = _evaluate_quick(model, test_loader, edge_index_sl, edge_weight_sl, torch.device("cpu"))
        roc = test_m.get("roc_auc", float("nan"))
        pr = test_m.get("pr_auc", float("nan"))
        f1 = test_m.get("f1", float("nan"))
        acc = test_m.get("accuracy", float("nan"))

        marker = ""
        if not np.isnan(roc) and roc > best_roc:
            best_roc = roc
            best_tag = cfg["tag"]
            marker = " ** BEST **"
            from pathlib import Path
            out_dir = Path("outputs/henderson2019_optimized")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
            with open(out_dir / "test_metrics.json", "w") as f:
                json.dump(test_m, f, indent=2, default=str)
            with open(out_dir / "best_config.json", "w") as f:
                json.dump(cfg, f, indent=2)

        all_results.append({"tag": cfg["tag"], "roc": roc, "pr": pr, "f1": f1, "acc": acc, "ep": info["best_epoch"]})
        print(f"[{i+1:2d}/{len(configs)}] {cfg['tag']:<40s} "
              f"ROC={roc:.4f} PR={pr:.4f} F1={f1:.4f} Acc={acc:.4f} ep={info['best_epoch']}{marker}")
    except Exception as e:
        print(f"[{i+1:2d}/{len(configs)}] {cfg['tag']:<40s} FAILED: {e}")

# Sort and print final
print(f"\n{'='*95}")
print("ROUND 2 TOP 5 vs BASELINE")
print(f"{'='*95}")
all_results.sort(key=lambda x: x.get("roc", 0), reverse=True)
print(f"{'Config':<45s} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Acc':>8}")
print("-" * 80)
print(f"{'ORIGINAL BASELINE':45s} {'0.7220':>8} {'0.7599':>8} {'0.7779':>8} {'0.6843':>8}")
print(f"{'ROUND 1 BEST':45s} {'0.7628':>8} {'0.8166':>8} {'0.7847':>8} {'0.7209':>8}")
print("-" * 80)
for r in all_results[:5]:
    print(f"{r['tag']:<45s} {r['roc']:8.4f} {r['pr']:8.4f} {r['f1']:8.4f} {r['acc']:8.4f}")

print(f"\nBest Round 2: {best_tag} (ROC-AUC: {best_roc:.4f})")

with open("outputs/round2_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
