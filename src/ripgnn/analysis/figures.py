"""Figure generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_calibration_curve(prob_pred: list[float], prob_true: list[float], path: str | Path) -> None:
    """Plot a calibration curve."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_probability_heatmap(probabilities: np.ndarray, region_ids: list[str], path: str | Path) -> None:
    """Plot mean per-region RIP probabilities."""
    mean_probs = np.mean(probabilities, axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(6, len(region_ids) * 0.8), 2.4))
    sns.heatmap(mean_probs, cmap="magma", cbar=True, xticklabels=region_ids, yticklabels=["mean"], ax=ax)
    ax.set_title("Regional infiltration probability")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hub_ranking(frame: pd.DataFrame, score_col: str, path: str | Path, title: str) -> None:
    """Plot a ranked hub bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=frame, x=score_col, y="region_id", ax=ax, color="#4C78A8")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

