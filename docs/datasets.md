# Dataset Guide

This repository works with three canonical artifacts:

- `snapshots.csv`
- `edges.csv`
- `region_meta.csv`

## Verified public resources

### 1. Primary supervised benchmark

- Midbrain-hindbrain assembloid paper:
  [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12120764/)
- Data DOI stated by the paper:
  [10.17881/w3kx-xn95](https://doi.org/10.17881/w3kx-xn95)

Use this as the first supervised source for regional spread prediction.

### 2. Atlas priors

- HNOCA overview:
  [Human Neural Organoid Atlas](https://data.humancellatlas.org/hca-bio-networks/organoid/atlases/organoid-neural-v1-0)
- HNOCA minimal mapping:
  [Zenodo 15004818](https://zenodo.org/records/15004818)
- HNOCA full dataset:
  [Zenodo 12536007](https://zenodo.org/records/12536007)
- HNOCA tools:
  [devsystemslab/HNOCA-tools](https://github.com/devsystemslab/HNOCA-tools)

Start with the minimal mapping release. Only move to the full release if you need more detailed region priors.

### 3. Mechanistic enrichment

- ProteomeXchange:
  [PXD061393](https://proteomecentral.proteomexchange.org/dataset/PXD061393)
- Optional aggregation prior:
  [AggNet](https://github.com/Hill-Wenka/AggNet)

Treat these as enrichment or external validation sources.

## Canonical artifact expectations

### `snapshots.csv`

Required columns:

- `organoid_id`
- `region_id`
- `time_idx`
- `condition`
- `burden`
- `burden_norm`
- `pathology_z`
- `delta_burden`
- `target_positive`

### `edges.csv`

Required columns:

- `src_region`
- `dst_region`
- `edge_weight`
- `edge_type`

### `region_meta.csv`

Required columns:

- `region_id`

Recommended prior columns:

- `hnoca_cortex_prior`
- `hnoca_striatum_prior`
- `hnoca_midbrain_prior`
- `hnoca_hindbrain_prior`

## Recommended ingestion process

1. Download processed tables from the public benchmark DOI.
2. Build or export a region-level measurement table with one row per `organoid_id, region_id, time_idx`.
3. Create a small topology file for the benchmark graph as `edges.csv`.
4. Export region prior summaries from HNOCA as `region_meta.csv`.
5. If the measurements table is not already canonical, map it with:

```powershell
ripgnn canonicalize `
  --measurements data\\raw\\benchmark_measurements.csv `
  --region-meta data\\raw\\region_meta.csv `
  --edges data\\raw\\edges.csv `
  --column-map configs\\column_map_template.yaml `
  --output-dir data\\processed\\public_bundle
```

## Notes

- If the public benchmark provides region labels directly, use them.
- If it only provides burdens, this repository derives positivity from control-adjusted `z > 2.0`.
- If voxel masks are unavailable, treat Dice as optional and center the paper on probability metrics, calibration, and hub interpretation.

