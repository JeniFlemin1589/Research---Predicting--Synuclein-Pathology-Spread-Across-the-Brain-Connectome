"""Preprocessing helpers for canonical RIP artifacts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def normalize_burden(
    frame: pd.DataFrame,
    burden_col: str = "burden",
    group_cols: Iterable[str] = ("organoid_id", "time_idx"),
    out_col: str = "burden_norm",
) -> pd.DataFrame:
    """Z-score burden within each organoid and timepoint."""
    result = frame.copy()
    grouped = result.groupby(list(group_cols))[burden_col]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    result[out_col] = ((result[burden_col] - means) / stds).fillna(0.0)
    return result


def add_delta_burden(
    frame: pd.DataFrame,
    value_col: str = "burden_norm",
    out_col: str = "delta_burden",
) -> pd.DataFrame:
    """Compute within-region temporal deltas."""
    result = frame.sort_values(["organoid_id", "region_id", "time_idx"]).copy()
    result[out_col] = (
        result.groupby(["organoid_id", "region_id"])[value_col].diff().fillna(0.0)
    )
    return result


def derive_pathology_z(
    frame: pd.DataFrame,
    value_col: str = "burden_norm",
    condition_col: str = "condition",
    control_value: str = "control",
    out_col: str = "pathology_z",
) -> pd.DataFrame:
    """Compute control-adjusted z-scores per region and timepoint."""
    result = frame.copy()
    controls = result[result[condition_col] == control_value]
    if controls.empty:
        raise ValueError(
            "No control rows were found. Provide explicit labels or include controls."
        )
    stats = (
        controls.groupby(["region_id", "time_idx"])[value_col]
        .agg(control_mean="mean", control_std="std")
        .reset_index()
    )
    result = result.merge(stats, on=["region_id", "time_idx"], how="left")
    std = result["control_std"].replace(0, np.nan)
    result[out_col] = ((result[value_col] - result["control_mean"]) / std).fillna(0.0)
    return result.drop(columns=["control_mean", "control_std"])


def derive_binary_targets(
    frame: pd.DataFrame,
    z_col: str = "pathology_z",
    threshold: float = 2.0,
    out_col: str = "target_positive",
) -> pd.DataFrame:
    """Create binary pathology labels from z-scores."""
    result = frame.copy()
    result[out_col] = (result[z_col] > threshold).astype(int)
    return result


def merge_region_priors(
    snapshots: pd.DataFrame,
    region_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Attach static region prior columns to the snapshot artifact."""
    return snapshots.merge(region_meta, on="region_id", how="left")

