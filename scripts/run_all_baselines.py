"""Run all baselines on Henderson 2019 data."""
import subprocess
import json
import sys
from pathlib import Path

PYTHON = r"C:\Program Files\Python312\python.exe"

BASELINES = {
    "logistic_regression": {"name": "logistic_regression", "hidden_dim": 64, "heads": 1, "dropout": 0.0, "regression_head": False},
    "mlp": {"name": "mlp", "hidden_dim": 64, "heads": 1, "dropout": 0.15, "regression_head": False},
    "gru_only": {"name": "gru_only", "hidden_dim": 64, "heads": 1, "dropout": 0.15, "regression_head": False},
    "gnn_only": {"name": "gnn_only", "hidden_dim": 64, "heads": 2, "dropout": 0.15, "regression_head": False},
}

results = {}

# Load existing dual_branch results
db_path = Path("outputs/henderson2019_dual_branch/test_metrics.json")
if db_path.exists():
    with open(db_path) as f:
        results["dual_branch"] = json.load(f)

for model_name, model_cfg in BASELINES.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    output_dir = f"outputs/henderson2019_{model_name}"

    # Create config
    import yaml
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

    # Train
    result = subprocess.run(
        [PYTHON, "-c",
         f"from ripgnn.cli import main; import sys; sys.argv = ['ripgnn', 'train', '--config', '{config_path}']; main()"],
        capture_output=True, text=True, cwd="."
    )

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        continue

    # Read results
    metrics_path = Path(output_dir) / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        results[model_name] = metrics
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        print(f"  F1: {metrics.get('f1', 'N/A')}")

# Summary table
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'Accuracy':>10}")
print("-" * 65)
for model_name in ["logistic_regression", "mlp", "gru_only", "gnn_only", "dual_branch"]:
    if model_name in results:
        m = results[model_name]
        roc = f"{m.get('roc_auc', float('nan')):.4f}" if m.get('roc_auc') else "N/A"
        pr = f"{m.get('pr_auc', float('nan')):.4f}" if m.get('pr_auc') else "N/A"
        f1 = f"{m.get('f1', float('nan')):.4f}" if m.get('f1') else "N/A"
        acc = f"{m.get('accuracy', float('nan')):.4f}" if m.get('accuracy') else "N/A"
        print(f"{model_name:<25} {roc:>10} {pr:>10} {f1:>10} {acc:>10}")

# Save summary
with open("outputs/henderson2019_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nFull results saved to outputs/henderson2019_comparison.json")
