"""Run complete ablation study: all 7 models on Henderson 2019 data.

Models:
  1. logistic_regression  — No temporal, no graph (linear baseline)
  2. mlp                  — No temporal, no graph (nonlinear baseline)
  3. gru_only             — Temporal only (GRU, no graph)
  4. lstm_only            — Temporal only (LSTM, no graph)
  5. gnn_only             — Graph only (GATv2, no temporal)
  6. dual_branch          — GATv2 + GRU (our model, GRU variant)
  7. dual_branch_lstm     — GATv2 + LSTM (our model, LSTM variant - matches Fig.1)
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

PYTHON = r"C:\Program Files\Python312\python.exe"

ALL_MODELS = {
    "logistic_regression": {
        "name": "logistic_regression",
        "hidden_dim": 64, "heads": 1, "dropout": 0.0, "regression_head": False,
    },
    "mlp": {
        "name": "mlp",
        "hidden_dim": 64, "heads": 1, "dropout": 0.15, "regression_head": False,
    },
    "gru_only": {
        "name": "gru_only",
        "hidden_dim": 64, "heads": 1, "dropout": 0.15, "regression_head": False,
    },
    "lstm_only": {
        "name": "lstm_only",
        "hidden_dim": 64, "heads": 1, "dropout": 0.15, "regression_head": False,
    },
    "gnn_only": {
        "name": "gnn_only",
        "hidden_dim": 64, "heads": 2, "dropout": 0.15, "regression_head": False,
    },
    "dual_branch": {
        "name": "dual_branch",
        "hidden_dim": 64, "heads": 2, "dropout": 0.15, "regression_head": False,
    },
    "dual_branch_lstm": {
        "name": "dual_branch_lstm",
        "hidden_dim": 64, "heads": 2, "dropout": 0.15, "regression_head": False,
    },
}

results = {}

for model_name, model_cfg in ALL_MODELS.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    output_dir = f"outputs/henderson2019_{model_name}"

    # Skip if already trained (check for existing results)
    metrics_path = Path(output_dir) / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Check if results are valid (not all NaN)
        if metrics.get("roc_auc") is not None and str(metrics["roc_auc"]) != "NaN":
            print(f"  Already trained — loading existing results")
            results[model_name] = metrics
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            continue

    config = {
        "project": {
            "name": f"henderson2019_{model_name}",
            "output_dir": output_dir,
            "seed": 42,
            "device": "cpu",
        },
        "data": {
            "artifact_dir": "data/henderson2019",
            "history_len": 1,
            "dynamic_features": ["burden_norm", "delta_burden", "pathology_z"],
            "static_features": ["snca_prior", "hemisphere", "vulnerability"],
            "split": {"train_fraction": 0.6, "val_fraction": 0.2},
        },
        "model": model_cfg,
        "training": {
            "epochs": 50,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "regression_weight": 0.0,
        },
    }

    config_path = f"configs/henderson2019_{model_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    result = subprocess.run(
        [PYTHON, "-c",
         f"from ripgnn.cli import main; import sys; "
         f"sys.argv = ['ripgnn', 'train', '--config', '{config_path}']; main()"],
        capture_output=True, text=True, cwd="."
    )

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        continue

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        results[model_name] = metrics
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        print(f"  F1: {metrics.get('f1', 'N/A')}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")

# === RESULTS TABLE ===
print(f"\n{'='*75}")
print("FULL ABLATION RESULTS — Henderson et al. 2019 α-Synuclein Dataset")
print(f"{'='*75}")
print(f"{'Model':<25} {'Temporal':<10} {'Spatial':<10} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'Accuracy':>10}")
print("-" * 85)

model_info = {
    "logistic_regression": ("—",      "—"),
    "mlp":                 ("—",      "—"),
    "gru_only":            ("GRU",    "—"),
    "lstm_only":           ("LSTM",   "—"),
    "gnn_only":            ("—",      "GATv2"),
    "dual_branch":         ("GRU",    "GATv2"),
    "dual_branch_lstm":    ("LSTM",   "GATv2"),
}

for model_name in model_info:
    if model_name in results:
        m = results[model_name]
        temp, spat = model_info[model_name]
        roc = f"{m.get('roc_auc', float('nan')):.4f}" if m.get('roc_auc') else "N/A"
        pr = f"{m.get('pr_auc', float('nan')):.4f}" if m.get('pr_auc') else "N/A"
        f1 = f"{m.get('f1', float('nan')):.4f}" if m.get('f1') else "N/A"
        acc = f"{m.get('accuracy', float('nan')):.4f}" if m.get('accuracy') else "N/A"
        print(f"{model_name:<25} {temp:<10} {spat:<10} {roc:>10} {pr:>10} {f1:>10} {acc:>10}")

# Save full comparison
with open("outputs/henderson2019_full_ablation.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nFull results saved to outputs/henderson2019_full_ablation.json")
