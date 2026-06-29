"""Collect and print all ablation results from output directories."""
import json
from pathlib import Path

models = [
    "logistic_regression", "mlp", "gru_only", "lstm_only",
    "gnn_only", "dual_branch", "dual_branch_lstm",
]

results = {}
for name in models:
    p = Path(f"outputs/henderson2019_{name}/test_metrics.json")
    if p.exists():
        with open(p) as f:
            results[name] = json.load(f)

# Save combined
with open("outputs/henderson2019_full_ablation.json", "w") as f:
    json.dump(results, f, indent=2)

model_info = {
    "logistic_regression": ("--", "--"),
    "mlp":                 ("--", "--"),
    "gru_only":            ("GRU", "--"),
    "lstm_only":           ("LSTM", "--"),
    "gnn_only":            ("--", "GATv2"),
    "dual_branch":         ("GRU", "GATv2"),
    "dual_branch_lstm":    ("LSTM", "GATv2"),
}

fmt = "{:<25} {:<10} {:<10} {:>10} {:>10} {:>10} {:>10}"
print(fmt.format("Model", "Temporal", "Spatial", "ROC-AUC", "PR-AUC", "F1", "Accuracy"))
print("-" * 85)
for name in models:
    if name in results:
        m = results[name]
        t, s = model_info[name]
        print(fmt.format(
            name, t, s,
            f"{m['roc_auc']:.4f}",
            f"{m['pr_auc']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['accuracy']:.4f}",
        ))

print()
db_gru = results.get("dual_branch", {})
db_lstm = results.get("dual_branch_lstm", {})
gru_o = results.get("gru_only", {})
lstm_o = results.get("lstm_only", {})

print("KEY COMPARISONS:")
print(f"  GRU-only vs LSTM-only:    ROC-AUC {gru_o['roc_auc']:.4f} vs {lstm_o['roc_auc']:.4f}")
print(f"  Dual(GRU) vs Dual(LSTM):  ROC-AUC {db_gru['roc_auc']:.4f} vs {db_lstm['roc_auc']:.4f}")
print(f"  Dual(GRU) gain over GRU-only:  +{db_gru['roc_auc'] - gru_o['roc_auc']:.4f}")
print(f"  Dual(LSTM) gain over LSTM-only: +{db_lstm['roc_auc'] - lstm_o['roc_auc']:.4f}")
print(f"  Best overall: dual_branch (GRU) with ROC-AUC={db_gru['roc_auc']:.4f}")
