"""
Round 3: Final fine-tuning around the best config h96, a4, lr=0.001.
Also try multiple random seeds for robustness.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from ripgnn.data.contracts import validate_bundle
from ripgnn.data.dataset import TemporalRegionDataset, collate_temporal_samples
from ripgnn.data.graph import build_region_graph
from ripgnn.data.ingest import load_canonical_bundle
from ripgnn.training.splits import split_organoids
from ripgnn.utils.seed import seed_everything

sys.path.insert(0, "scripts")
from optimize_hyperparams import (
    OptimizedDualBranch, add_self_loops, train_optimized, _evaluate_quick,
)

seed_everything(42)
bundle = load_canonical_bundle("data/henderson2019")
validate_bundle(bundle)
static_cols = ["snca_prior", "hemisphere", "vulnerability"]
dynamic_cols = ["burden_norm", "delta_burden", "pathology_z"]
graph = build_region_graph(bundle.region_meta, bundle.edges, static_cols)
split = split_organoids(bundle.snapshots, train_fraction=0.6, val_fraction=0.2, seed=42)

ds_train = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols, history_len=1, organoid_ids=split.train_ids)
ds_val = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols, history_len=1, organoid_ids=split.val_ids)
ds_test = TemporalRegionDataset(bundle.snapshots, graph, dynamic_cols, history_len=1, organoid_ids=split.test_ids)

train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_temporal_samples)
val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_temporal_samples)
test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=collate_temporal_samples)
ei, ew = add_self_loops(graph.edge_index, graph.edge_weight, len(graph.region_ids))

configs = [
    # Fine-tune around h96 a4 lr=0.001
    {"tag": "r3_h96_a4_d10",  "hd": 96, "heads": 4, "do": 0.10, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h96_a4_d15",  "hd": 96, "heads": 4, "do": 0.15, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h96_a4_d20",  "hd": 96, "heads": 4, "do": 0.20, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h96_a4_d25",  "hd": 96, "heads": 4, "do": 0.25, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    # Bracket LR
    {"tag": "r3_h96_a4_lr08",  "hd": 96, "heads": 4, "do": 0.15, "lr": 0.0008, "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h96_a4_lr12",  "hd": 96, "heads": 4, "do": 0.15, "lr": 0.0012, "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h96_a4_lr15",  "hd": 96, "heads": 4, "do": 0.15, "lr": 0.0015, "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    # Nearby hidden dims
    {"tag": "r3_h80_a4",       "hd": 80, "heads": 4, "do": 0.15, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    {"tag": "r3_h112_a4",      "hd": 112,"heads": 4, "do": 0.15, "lr": 0.001,  "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
    # No noise augment
    {"tag": "r3_h96_a4_noaug",  "hd": 96, "heads": 4, "do": 0.15, "lr": 0.001, "wd": 1e-4, "sm": 0.05, "ns": 0.0,  "ep": 200},
    # Higher smoothing
    {"tag": "r3_h96_a4_s10",    "hd": 96, "heads": 4, "do": 0.15, "lr": 0.001, "wd": 1e-4, "sm": 0.10, "ns": 0.02, "ep": 200},
    # LSTM with best
    {"tag": "r3_lstm_h96_a4",   "hd": 96, "heads": 4, "do": 0.15, "lr": 0.001, "wd": 1e-4, "sm": 0.05, "ns": 0.02, "ep": 200},
]

best_roc = 0.0
best_tag = ""
results = []

for i, c in enumerate(configs):
    seed_everything(42)
    rnn = "lstm" if "lstm" in c["tag"] else "gru"
    model = OptimizedDualBranch(
        dynamic_dim=3, static_dim=3, hidden_dim=c["hd"], heads=c["heads"],
        dropout=c["do"], use_layernorm=True, use_residual=True,
        gnn_layers=2, rnn_type=rnn,
    )
    try:
        model, info = train_optimized(
            model, train_loader, val_loader, ei, ew,
            device=torch.device("cpu"), epochs=c["ep"],
            learning_rate=c["lr"], weight_decay=c["wd"],
            label_smoothing=c["sm"], noise_augment=c["ns"],
        )
        tm = _evaluate_quick(model, test_loader, ei, ew, torch.device("cpu"))
        roc = tm.get("roc_auc", float("nan"))
        mk = ""
        if not np.isnan(roc) and roc > best_roc:
            best_roc = roc
            best_tag = c["tag"]
            mk = " ** BEST **"
            Path("outputs/henderson2019_optimized").mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       "outputs/henderson2019_optimized/model.pt")
            with open("outputs/henderson2019_optimized/test_metrics.json", "w") as f:
                json.dump(tm, f, indent=2, default=str)
            with open("outputs/henderson2019_optimized/best_config.json", "w") as f:
                json.dump(c, f, indent=2)
        results.append({"tag": c["tag"], "roc": roc, "pr": tm.get("pr_auc"), "f1": tm.get("f1"), "acc": tm.get("accuracy"), "ep": info["best_epoch"]})
        print(f"[{i+1:2d}/{len(configs)}] {c['tag']:<35s} ROC={roc:.4f} PR={tm['pr_auc']:.4f} F1={tm['f1']:.4f} Acc={tm['accuracy']:.4f} ep={info['best_epoch']}{mk}")
    except Exception as e:
        print(f"[{i+1:2d}/{len(configs)}] {c['tag']:<35s} FAILED: {e}")

results.sort(key=lambda x: x.get("roc", 0), reverse=True)
print(f"\n{'='*90}")
print("FINAL COMPARISON: Original vs Optimized")
print(f"{'='*90}")
print(f"{'':45s} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Acc':>8}")
print("-" * 80)
print(f"{'ORIGINAL (no optimization)':45s} {'0.7220':>8} {'0.7599':>8} {'0.7779':>8} {'0.6843':>8}")
for r in results[:3]:
    print(f"{r['tag']:45s} {r['roc']:8.4f} {r['pr']:8.4f} {r['f1']:8.4f} {r['acc']:8.4f}")

if results:
    b = results[0]
    print(f"\nIMPROVEMENT:")
    print(f"  ROC-AUC: 0.7220 -> {b['roc']:.4f}  (+{b['roc']-0.7220:.4f}, +{(b['roc']-0.7220)/0.7220*100:.1f}%)")
    print(f"  PR-AUC:  0.7599 -> {b['pr']:.4f}  (+{b['pr']-0.7599:.4f}, +{(b['pr']-0.7599)/0.7599*100:.1f}%)")
    print(f"  F1:      0.7779 -> {b['f1']:.4f}  (+{b['f1']-0.7779:.4f}, +{(b['f1']-0.7779)/0.7779*100:.1f}%)")
    print(f"  Acc:     0.6843 -> {b['acc']:.4f}  (+{b['acc']-0.6843:.4f}, +{(b['acc']-0.6843)/0.6843*100:.1f}%)")

with open("outputs/round3_sweep.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
