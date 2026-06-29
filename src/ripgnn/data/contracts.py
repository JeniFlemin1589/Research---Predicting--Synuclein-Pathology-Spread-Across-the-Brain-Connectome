"""Canonical artifact contracts and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SNAPSHOT_REQUIRED_COLUMNS = {
    "organoid_id",
    "region_id",
    "time_idx",
    "condition",
    "burden",
    "burden_norm",
    "pathology_z",
    "delta_burden",
    "target_positive",
}

EDGE_REQUIRED_COLUMNS = {"src_region", "dst_region", "edge_weight", "edge_type"}
REGION_META_REQUIRED_COLUMNS = {"region_id"}


@dataclass(frozen=True)
class ArtifactBundle:
    snapshots: pd.DataFrame
    edges: pd.DataFrame
    region_meta: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_snapshots(frame: pd.DataFrame) -> None:
    """Validate the canonical snapshots artifact."""
    _require_columns(frame, SNAPSHOT_REQUIRED_COLUMNS, "snapshots")
    duplicate_mask = frame.duplicated(["organoid_id", "region_id", "time_idx"])
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(
            "snapshots contains duplicate organoid/region/time rows "
            f"({count} duplicates found)"
        )
    if frame["time_idx"].isna().any():
        raise ValueError("snapshots has missing time_idx values")
    if frame["target_positive"].isna().any():
        raise ValueError("snapshots has missing target_positive values")


def validate_edges(frame: pd.DataFrame) -> None:
    """Validate the canonical edges artifact."""
    _require_columns(frame, EDGE_REQUIRED_COLUMNS, "edges")
    if (frame["edge_weight"] < 0).any():
        raise ValueError("edges contains negative edge_weight values")


def validate_region_meta(frame: pd.DataFrame) -> None:
    """Validate the canonical region metadata artifact."""
    _require_columns(frame, REGION_META_REQUIRED_COLUMNS, "region_meta")
    if frame["region_id"].duplicated().any():
        raise ValueError("region_meta has duplicate region_id rows")


def validate_bundle(bundle: ArtifactBundle) -> None:
    """Validate all canonical artifacts and their alignment."""
    validate_snapshots(bundle.snapshots)
    validate_edges(bundle.edges)
    validate_region_meta(bundle.region_meta)
    known_regions = set(bundle.region_meta["region_id"])
    unknown_snapshot = sorted(set(bundle.snapshots["region_id"]) - known_regions)
    if unknown_snapshot:
        raise ValueError(f"snapshots contains unknown regions: {unknown_snapshot}")
    edge_regions = set(bundle.edges["src_region"]).union(bundle.edges["dst_region"])
    unknown_edge = sorted(edge_regions - known_regions)
    if unknown_edge:
        raise ValueError(f"edges contains unknown regions: {unknown_edge}")

