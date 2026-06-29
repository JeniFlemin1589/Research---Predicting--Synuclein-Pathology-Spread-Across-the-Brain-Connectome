"""
Ingest Henderson et al. 2019 α-synuclein pathology spread dataset.

Source: Henderson, Cornblath et al. "Spread of α-synuclein pathology through
the brain connectome is modulated by selective vulnerability and predicted
by network analysis." Nature Neuroscience 22, 1248–1257 (2019).

Data: https://github.com/ejcorn/connectome_diffusion
Replication: https://github.com/MathieuBo/PathoSpreading

This script converts the raw Henderson CSV files into the canonical
RIP-GNN bundle format (snapshots.csv, edges.csv, region_meta.csv).

The dataset contains:
  - Pathology quantification (% area staining) for 116 ipsi+contra regions
    at 1, 3, 6 months post-injection in NTG (control) and G20 (transgenic) mice
  - Allen Brain Institute structural connectome (58×58 ipsi + 58×58 contra)
  - Endogenous Snca gene expression per region (static priors)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_pathology_data(path: str | Path) -> pd.DataFrame:
    """Load and reshape the wide-format pathology data into long-format.

    Since each mouse is sacrificed at one timepoint, we create pseudo-organoid
    trajectories by pairing mice across timepoints within the same condition.
    This gives the temporal model multi-timepoint sequences to learn from.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    # Columns: Time post-injection (months), MBSC Region, then 116 region columns
    time_col = raw.columns[0]  # "Time post-injection (months)"
    condition_col = raw.columns[1]  # "MBSC Region" → actually condition (NTG/G20)
    region_cols = raw.columns[2:]

    # Map conditions
    raw[condition_col] = raw[condition_col].map({"NTG": "control", "G20": "treated"})

    # Map times to indices
    time_map = {1: 0, 3: 1, 6: 2}
    raw["time_idx"] = raw[time_col].map(time_map)

    # Add subject index per timepoint-condition group
    raw["subject_idx"] = raw.groupby([time_col, condition_col]).cumcount()

    # Create pseudo-organoids by pairing subjects across timepoints.
    # For each condition, subject_idx k at all available timepoints forms one trajectory.
    # If the number of mice per timepoint differs, we cycle through the shorter groups.
    all_rows = []
    for condition in ["control", "treated"]:
        cond_data = raw[raw[condition_col] == condition]
        time_groups = {}
        for t in sorted(cond_data["time_idx"].unique()):
            time_groups[t] = cond_data[cond_data["time_idx"] == t].reset_index(drop=True)

        # Number of pseudo-organoids = max subjects at any timepoint
        max_subjects = max(len(g) for g in time_groups.values())

        for pseudo_idx in range(max_subjects):
            organoid_id = f"{condition}_pseudo_{pseudo_idx}"
            for t, group in time_groups.items():
                # Cycle index if fewer subjects at this timepoint
                row_idx = pseudo_idx % len(group)
                row = group.iloc[row_idx]
                for region in region_cols:
                    val = row[region]
                    if pd.notna(val):
                        all_rows.append({
                            "organoid_id": organoid_id,
                            "region_id": region,
                            "time_idx": int(t),
                            "condition": condition,
                            "burden": float(val),
                        })

    long = pd.DataFrame(all_rows)

    # Normalize burden per region across all samples (min-max to [0,1])
    long["burden_norm"] = long.groupby("region_id")["burden"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
    )

    # Compute delta_burden per organoid × region across time
    long = long.sort_values(["organoid_id", "region_id", "time_idx"])
    long["delta_burden"] = long.groupby(
        ["organoid_id", "region_id"]
    )["burden_norm"].diff().fillna(0)

    # Z-score of burden
    mean_b = long["burden_norm"].mean()
    std_b = long["burden_norm"].std()
    long["pathology_z"] = (long["burden_norm"] - mean_b) / (std_b + 1e-12)

    # Binary target: region is "at risk" if burden is above median
    global_median = long["burden_norm"].median()
    long["target_positive"] = (long["burden_norm"] > global_median).astype(int)

    # Select and order canonical columns
    snapshots = long[[
        "organoid_id", "region_id", "time_idx", "condition",
        "burden", "burden_norm", "delta_burden", "pathology_z",
        "target_positive",
    ]].copy()

    return snapshots


def load_connectivity(
    ipsi_path: str | Path,
    contra_path: str | Path,
    region_cols: list[str],
) -> pd.DataFrame:
    """Load connectivity matrices and produce an edge list.

    We use the ipsilateral connectivity as primary edges (stronger, same hemisphere).
    The contralateral connectivity adds cross-hemisphere edges.
    We select the top-K edges per region to keep the graph manageable.
    """
    ipsi = pd.read_csv(ipsi_path, encoding="utf-8-sig", index_col=0)
    contra = pd.read_csv(contra_path, encoding="utf-8-sig", index_col=0)

    # The connectivity is for 58 "abstract" regions; we need to map them
    # to the 116 ipsi+contra regions in the pathology data
    # The ipsi/contra matrices are between the same 58 abstract regions.
    # In our data, regions are prefixed with i (ipsi) or c (contra).

    # Build mapping from abstract region name to region_id prefix
    abstract_regions = list(ipsi.index)

    edges = []

    # Ipsilateral edges: connect ipsi regions to ipsi, and contra to contra
    for i, src_abstract in enumerate(abstract_regions):
        for j, dst_abstract in enumerate(abstract_regions):
            w = float(ipsi.iloc[i, j])
            if w > 0 and i != j:
                # Both ipsi-side
                src_ipsi = _find_region(src_abstract, region_cols, prefix="i")
                dst_ipsi = _find_region(dst_abstract, region_cols, prefix="i")
                if src_ipsi and dst_ipsi:
                    edges.append({
                        "src_region": src_ipsi,
                        "dst_region": dst_ipsi,
                        "weight": w,
                        "type": "ipsi",
                    })
                # Both contra-side
                src_contra = _find_region(src_abstract, region_cols, prefix="c")
                dst_contra = _find_region(dst_abstract, region_cols, prefix="c")
                if src_contra and dst_contra:
                    edges.append({
                        "src_region": src_contra,
                        "dst_region": dst_contra,
                        "weight": w,
                        "type": "contra_internal",
                    })

    # Cross-hemisphere edges: connect ipsi to contra
    for i, src_abstract in enumerate(abstract_regions):
        for j, dst_abstract in enumerate(abstract_regions):
            w = float(contra.iloc[i, j])
            if w > 0:
                src_ipsi = _find_region(src_abstract, region_cols, prefix="i")
                dst_contra = _find_region(dst_abstract, region_cols, prefix="c")
                if src_ipsi and dst_contra:
                    edges.append({
                        "src_region": src_ipsi,
                        "dst_region": dst_contra,
                        "weight": w,
                        "type": "cross_hemisphere",
                    })

    edge_df = pd.DataFrame(edges)

    # Keep top edges by weight to avoid an overly dense graph
    # Log-transform weights since they span many orders of magnitude
    edge_df["log_weight"] = np.log1p(edge_df["weight"])

    # Normalize weights to [0, 1]
    max_w = edge_df["log_weight"].max()
    edge_df["weight"] = edge_df["log_weight"] / (max_w + 1e-12)

    # Filter: keep edges with weight above a threshold
    threshold = edge_df["weight"].quantile(0.5)  # top 50% of edges
    edge_df = edge_df[edge_df["weight"] >= threshold].copy()

    edge_df = edge_df.rename(columns={"weight": "edge_weight", "type": "edge_type"})
    edge_df = edge_df[["src_region", "dst_region", "edge_weight", "edge_type"]].drop_duplicates()

    return edge_df


def _find_region(abstract_name: str, region_cols: list[str], prefix: str) -> str | None:
    """Map an abstract region name to the actual region column.

    The abstract names in the connectivity matrix look like:
        'Cg (ACAd + ACAv)', 'Acb (ACB)', etc.

    The region columns in pathology data look like:
        'iCg', 'cCg', 'iAcb', 'cAcb', etc.

    We try matching by the abbreviation before the parenthesis.
    """
    # Extract short name from abstract name (before the parenthesis)
    short = abstract_name.split("(")[0].strip()
    # Remove spaces and special chars for matching
    short_clean = short.replace(" ", "").replace("-", "")

    target = prefix + short_clean
    for col in region_cols:
        col_clean = col.replace("-", "")
        if col_clean.lower() == target.lower():
            return col

    # Try partial match
    for col in region_cols:
        col_clean = col.replace("-", "")
        if col_clean.lower().startswith(target.lower()):
            return col

    return None


def load_snca_expression(path: str | Path) -> pd.DataFrame:
    """Load SNCA expression and build region_meta table."""
    snca = pd.read_csv(path, header=None, names=["region_id", "snca_expression"],
                       encoding="utf-8-sig")

    # Normalize expression to [0, 1]
    snca["snca_norm"] = (snca["snca_expression"] - snca["snca_expression"].min()) / (
        snca["snca_expression"].max() - snca["snca_expression"].min() + 1e-12
    )

    # Create region_meta with priors
    region_meta = snca[["region_id"]].copy()
    region_meta["snca_prior"] = snca["snca_norm"]
    # Add hemisphere feature
    region_meta["hemisphere"] = region_meta["region_id"].apply(
        lambda r: 0.0 if r.startswith("i") else 1.0
    )
    # Vulnerability score: proxy based on SNCA expression
    region_meta["vulnerability"] = snca["snca_norm"]

    return region_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Henderson et al. 2019 dataset")
    parser.add_argument("--raw-dir", default="data/raw/henderson2019")
    parser.add_argument("--output-dir", default="data/henderson2019")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pathology data...")
    snapshots = load_pathology_data(raw_dir / "pathology_data.csv")
    print(f"  Snapshots: {len(snapshots)} rows, "
          f"{snapshots['organoid_id'].nunique()} subjects, "
          f"{snapshots['region_id'].nunique()} regions, "
          f"{snapshots['time_idx'].nunique()} timepoints")

    region_cols = list(snapshots["region_id"].unique())

    print("Loading SNCA expression...")
    region_meta = load_snca_expression(raw_dir / "snca_expression.csv")
    # Keep only regions present in snapshots
    region_meta = region_meta[region_meta["region_id"].isin(region_cols)].copy()
    print(f"  Regions with SNCA priors: {len(region_meta)}")

    print("Loading connectivity matrices...")
    edges = load_connectivity(
        raw_dir / "connectivity_ipsi.csv",
        raw_dir / "connectivity_contra.csv",
        region_cols,
    )
    # Keep only edges between regions present in snapshots
    valid_regions = set(region_cols)
    edges = edges[
        edges["src_region"].isin(valid_regions) &
        edges["dst_region"].isin(valid_regions)
    ].copy()
    print(f"  Edges: {len(edges)} connections")

    # Ensure all snapshot regions have metadata
    missing_meta = set(region_cols) - set(region_meta["region_id"])
    if missing_meta:
        print(f"  Adding default priors for {len(missing_meta)} regions without SNCA data")
        default_rows = pd.DataFrame({
            "region_id": list(missing_meta),
            "snca_prior": 0.5,
            "hemisphere": [0.0 if r.startswith("i") else 1.0 for r in missing_meta],
            "vulnerability": 0.5,
        })
        region_meta = pd.concat([region_meta, default_rows], ignore_index=True)

    # Save canonical artifacts
    snapshots.to_csv(out_dir / "snapshots.csv", index=False)
    edges.to_csv(out_dir / "edges.csv", index=False)
    region_meta.to_csv(out_dir / "region_meta.csv", index=False)

    print(f"\nCanonical bundle written to {out_dir}/")
    print(f"  snapshots.csv: {len(snapshots)} rows")
    print(f"  edges.csv: {len(edges)} edges")
    print(f"  region_meta.csv: {len(region_meta)} regions")

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Subjects: {snapshots['organoid_id'].nunique()}")
    print(f"Regions: {snapshots['region_id'].nunique()}")
    print(f"Timepoints: {sorted(snapshots['time_idx'].unique())}")
    print(f"Conditions: {sorted(snapshots['condition'].unique())}")
    print(f"Positive rate: {snapshots['target_positive'].mean():.3f}")
    print(f"Mean burden (treated): "
          f"{snapshots[snapshots['condition']=='treated']['burden'].mean():.6f}")
    print(f"Mean burden (control): "
          f"{snapshots[snapshots['condition']=='control']['burden'].mean():.6f}")


if __name__ == "__main__":
    main()
