"""Public resource registry for the RIP-GNN project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetResource:
    name: str
    url: str
    purpose: str
    notes: str


DATASET_REGISTRY: list[DatasetResource] = [
    DatasetResource(
        name="HNOCA atlas overview",
        url="https://data.humancellatlas.org/hca-bio-networks/organoid/atlases/organoid-neural-v1-0",
        purpose="Atlas overview and entry point for human neural organoid references.",
        notes="Use this first to understand region coverage and organoid classes.",
    ),
    DatasetResource(
        name="HNOCA full dataset",
        url="https://zenodo.org/records/12536007",
        purpose="Full atlas release for advanced region prior engineering.",
        notes="Large download, use only after validating that the minimal bundle is insufficient.",
    ),
    DatasetResource(
        name="HNOCA minimal mapping dataset",
        url="https://zenodo.org/records/15004818",
        purpose="Smaller atlas bundle for region-level priors and mappings.",
        notes="Recommended first atlas download for this repository.",
    ),
    DatasetResource(
        name="HNOCA-tools",
        url="https://github.com/devsystemslab/HNOCA-tools",
        purpose="Utilities for working with the Human Neural Organoid Cell Atlas.",
        notes="Useful if you later need atlas-native preprocessing.",
    ),
    DatasetResource(
        name="Midbrain-hindbrain assembloid benchmark paper",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12120764/",
        purpose="Closest public spread benchmark for supervised RIP prediction.",
        notes="The paper states that original and processed data are available under DOI 10.17881/w3kx-xn95.",
    ),
    DatasetResource(
        name="Midbrain-hindbrain assembloid DOI",
        url="https://doi.org/10.17881/w3kx-xn95",
        purpose="Primary public benchmark data release.",
        notes="Use as the first supervised source. Raw 3D masks may or may not be exposed.",
    ),
    DatasetResource(
        name="ProteomeXchange PXD061393",
        url="https://proteomecentral.proteomexchange.org/dataset/PXD061393",
        purpose="External mechanistic enrichment and validation dataset.",
        notes="Treat as enrichment or external validation, not as the primary supervised label source.",
    ),
    DatasetResource(
        name="AggNet",
        url="https://github.com/Hill-Wenka/AggNet",
        purpose="Optional aggregation-risk prior model.",
        notes="Not required for v1 training.",
    ),
    DatasetResource(
        name="PRGNN",
        url="https://github.com/Treeboy2762/PRGNN",
        purpose="Reference implementation for region-based GNN design patterns.",
        notes="Useful architectural reference for region graph reasoning.",
    ),
    DatasetResource(
        name="GSE278265",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278265",
        purpose="Optional disease-context transcriptomics.",
        notes="Use only if you need additional biological context beyond the core benchmark.",
    ),
    DatasetResource(
        name="GSE236002",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236002",
        purpose="Optional disease-context transcriptomics.",
        notes="Secondary source for contextual validation or priors.",
    ),
]

