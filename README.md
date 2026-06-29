# RIP-GNN

`ripgnn` is a hybrid research repository for predicting next-step regional infiltration probability (RIP) in public brain organoid spread datasets.

The repository implements:

- Canonical `snapshots`, `edges`, and `region_meta` artifacts
- A static region graph builder with HNOCA-style prior features
- Baselines: logistic regression, node MLP, GRU-only, GNN-only
- Main model: dual-branch `GATv2 + GRU + fusion MLP`
- Time-aware train/validation/test splitting
- Metrics, calibration, hub interpretation, and publication-ready figures
- Five starter notebooks for data audit, graph construction, model training, and paper figures

## Environment

This stack is targeted at Python 3.12 because PyTorch and PyG support is much more reliable there than on Python 3.14.

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch

CPU-only example from the official PyTorch Windows instructions:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you have a CUDA-capable NVIDIA GPU, choose the matching command from the official selector:

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)

### 3. Install PyTorch Geometric and the project

PyG documents that `pip install torch_geometric` works from PyG 2.3 onward, with optional extra wheels for faster kernels:

- [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

Install the project:

```powershell
pip install torch_geometric
pip install -e .[dev]
```

### 4. Bootstrap the demo dataset

```powershell
ripgnn bootstrap-demo --output-dir data/demo
```

### 5. Audit and train

```powershell
ripgnn audit-data --data-dir data/demo
ripgnn train --config configs/demo_dual_branch.yaml
ripgnn evaluate --run-dir outputs/demo_dual_branch
ripgnn interpret --run-dir outputs/demo_dual_branch
```

### 6. Run the notebooks

Start JupyterLab with the project-local configuration:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_jupyter.ps1
```

Then open the notebooks in this order:

- `notebooks/01_data_audit.ipynb`
- `notebooks/02_graph_build.ipynb`
- `notebooks/03_baseline_training.ipynb`
- `notebooks/04_dual_branch_training.ipynb`
- `notebooks/05_paper_figures.ipynb`

## Data layout

Canonical artifacts:

- `snapshots.csv`: one row per `organoid_id, region_id, time_idx`
- `edges.csv`: one row per `src_region, dst_region`
- `region_meta.csv`: one row per region with static priors

See [docs/datasets.md](docs/datasets.md) for public resource links and ingestion guidance.

## Repo layout

```text
configs/
docs/
notebooks/
src/ripgnn/
tests/
```

## Research workflow

1. Bootstrap or ingest a public dataset bundle into canonical CSVs.
2. Audit quality, labels, region coverage, and split integrity.
3. Train baselines.
4. Train the dual-branch GNN-kinetic model.
5. Generate RIP heatmaps, calibration curves, ablation plots, and hub rankings.
6. Assemble figures and methods notes for the paper.

## Important assumptions

- v1 is a public-data surrogate of the full wet-lab RIP workflow.
- If the benchmark dataset does not expose voxel masks, Dice stays optional and probability-focused metrics become primary.
- `PXD061393` is treated as enrichment or external validation data, not the primary supervised label source.
