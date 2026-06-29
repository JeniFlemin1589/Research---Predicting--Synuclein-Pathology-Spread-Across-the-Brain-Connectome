"""Ingestion helpers for canonical public-data bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ripgnn.data.contracts import ArtifactBundle, validate_bundle
from ripgnn.data.preprocess import (
    add_delta_burden,
    derive_binary_targets,
    derive_pathology_z,
    normalize_burden,
)


def load_canonical_bundle(data_dir: str | Path) -> ArtifactBundle:
    """Load an already-canonicalized artifact bundle from a directory."""
    base = Path(data_dir)
    bundle = ArtifactBundle(
        snapshots=pd.read_csv(base / "snapshots.csv"),
        edges=pd.read_csv(base / "edges.csv"),
        region_meta=pd.read_csv(base / "region_meta.csv"),
    )
    validate_bundle(bundle)
    return bundle


def canonicalize_measurement_table(
    measurements_path: str | Path,
    region_meta_path: str | Path,
    edges_path: str | Path,
    column_map: dict[str, str],
    control_value: str = "control",
) -> ArtifactBundle:
    """Map an external processed table into canonical artifacts."""
    raw = pd.read_csv(measurements_path)
    renamed = raw.rename(columns=column_map).copy()
    required = {"organoid_id", "region_id", "time_idx", "condition", "burden"}
    missing = sorted(required.difference(renamed.columns))
    if missing:
        raise ValueError(f"mapped measurements table is missing required fields: {missing}")
    snapshots = renamed.loc[:, sorted(required)].copy()
    snapshots = normalize_burden(snapshots)
    snapshots = add_delta_burden(snapshots)
    snapshots = derive_pathology_z(snapshots, control_value=control_value)
    snapshots = derive_binary_targets(snapshots)
    bundle = ArtifactBundle(
        snapshots=snapshots,
        edges=pd.read_csv(edges_path),
        region_meta=pd.read_csv(region_meta_path),
    )
    validate_bundle(bundle)
    return bundle


def bundle_to_dir(bundle: ArtifactBundle, output_dir: str | Path) -> dict[str, Path]:
    """Persist a canonical bundle to CSV files."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "snapshots": base / "snapshots.csv",
        "edges": base / "edges.csv",
        "region_meta": base / "region_meta.csv",
    }
    bundle.snapshots.to_csv(paths["snapshots"], index=False)
    bundle.edges.to_csv(paths["edges"], index=False)
    bundle.region_meta.to_csv(paths["region_meta"], index=False)
    return paths


def manifest_for_public_resources() -> list[dict[str, Any]]:
    """Return a minimal manifest schema for downstream docs or notebooks."""
    return [
        {
            "expected_file": "snapshots.csv",
            "description": "Canonical regional time-series table for RIP training.",
        },
        {
            "expected_file": "edges.csv",
            "description": "Canonical static region graph edges.",
        },
        {
            "expected_file": "region_meta.csv",
            "description": "Canonical static HNOCA-style region priors.",
        },
    ]

