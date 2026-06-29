"""Synthetic public-data surrogate for end-to-end smoke testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ripgnn.data.preprocess import (
    add_delta_burden,
    derive_binary_targets,
    derive_pathology_z,
    normalize_burden,
)


REGIONS = [
    "cortex",
    "striatum",
    "midbrain",
    "hindbrain",
    "thalamus",
    "glia_niche",
]


def build_region_meta() -> pd.DataFrame:
    """Create a small interpretable region prior table."""
    rows = [
        ("cortex", 0.95, 0.15, 0.25, 0.10),
        ("striatum", 0.25, 0.95, 0.45, 0.25),
        ("midbrain", 0.15, 0.50, 0.95, 0.75),
        ("hindbrain", 0.10, 0.25, 0.80, 0.95),
        ("thalamus", 0.70, 0.65, 0.40, 0.20),
        ("glia_niche", 0.30, 0.30, 0.50, 0.55),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "region_id",
            "hnoca_cortex_prior",
            "hnoca_striatum_prior",
            "hnoca_midbrain_prior",
            "hnoca_hindbrain_prior",
        ],
    )


def build_edges() -> pd.DataFrame:
    """Create a static region topology."""
    rows = [
        ("midbrain", "striatum", 0.92, "anatomical"),
        ("striatum", "cortex", 0.84, "anatomical"),
        ("midbrain", "hindbrain", 0.74, "anatomical"),
        ("hindbrain", "thalamus", 0.56, "anatomical"),
        ("thalamus", "cortex", 0.64, "anatomical"),
        ("glia_niche", "midbrain", 0.62, "glial"),
        ("glia_niche", "striatum", 0.58, "glial"),
    ]
    forward = pd.DataFrame(rows, columns=["src_region", "dst_region", "edge_weight", "edge_type"])
    reverse = forward.rename(
        columns={"src_region": "dst_region", "dst_region": "src_region"}
    )
    return pd.concat([forward, reverse], ignore_index=True)


def build_snapshots(
    seed: int = 7,
    n_organoids: int = 24,
    n_timepoints: int = 6,
) -> pd.DataFrame:
    """Simulate a modest spread process with controls and treated organoids."""
    rng = np.random.default_rng(seed)
    region_index = {region: idx for idx, region in enumerate(REGIONS)}
    adjacency = np.zeros((len(REGIONS), len(REGIONS)), dtype=float)
    for row in build_edges().itertuples(index=False):
        adjacency[region_index[row.src_region], region_index[row.dst_region]] = row.edge_weight

    records: list[dict[str, object]] = []
    for organoid_idx in range(n_organoids):
        organoid_id = f"organoid_{organoid_idx:03d}"
        condition = "control" if organoid_idx % 4 == 0 else "treated"
        susceptibility = 0.20 if condition == "control" else 0.42 + 0.08 * rng.random()
        burden = np.zeros((n_timepoints, len(REGIONS)), dtype=float)
        seed_node = region_index["midbrain"] if organoid_idx % 3 else region_index["hindbrain"]
        burden[0, seed_node] = 1.0 + rng.normal(0.0, 0.08)
        burden[0, region_index["glia_niche"]] = 0.45 + 0.10 * rng.random()
        for time_idx in range(1, n_timepoints):
            propagated = adjacency.T @ burden[time_idx - 1]
            persistence = 0.55 * burden[time_idx - 1]
            noise = rng.normal(0.0, 0.05, size=len(REGIONS))
            burden[time_idx] = np.clip(
                persistence + susceptibility * propagated + noise,
                0.0,
                None,
            )
        for time_idx in range(n_timepoints):
            for region, region_pos in region_index.items():
                records.append(
                    {
                        "organoid_id": organoid_id,
                        "region_id": region,
                        "time_idx": time_idx,
                        "condition": condition,
                        "burden": float(burden[time_idx, region_pos]),
                    }
                )
    frame = pd.DataFrame(records)
    frame = normalize_burden(frame)
    frame = add_delta_burden(frame)
    frame = derive_pathology_z(frame)
    frame = derive_binary_targets(frame)
    return frame


def write_demo_bundle(output_dir: str | Path, seed: int = 7) -> dict[str, Path]:
    """Write a synthetic bundle compatible with the full training pipeline."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    snapshots_path = base / "snapshots.csv"
    edges_path = base / "edges.csv"
    region_meta_path = base / "region_meta.csv"
    build_snapshots(seed=seed).to_csv(snapshots_path, index=False)
    build_edges().to_csv(edges_path, index=False)
    build_region_meta().to_csv(region_meta_path, index=False)
    return {
        "snapshots": snapshots_path,
        "edges": edges_path,
        "region_meta": region_meta_path,
    }

