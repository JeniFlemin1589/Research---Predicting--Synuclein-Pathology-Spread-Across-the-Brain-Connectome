# Publication Workflow

This repository is structured to support a focused conference-style paper.

## Suggested paper sections

1. Problem statement
2. Public-data RIP formulation
3. Canonical artifact design
4. Graph construction from region topology and HNOCA priors
5. Baselines vs. dual-branch model
6. Calibration and hub interpretation
7. Limitations and wet-lab extension path

## Figures to produce

1. Data pipeline diagram
2. Static region graph diagram
3. Baseline vs. dual-branch comparison table
4. RIP heatmap
5. Calibration plot
6. Attention hub ranking
7. Occlusion hub ranking
8. Ablation summary

## Experimental checklist

1. Audit the benchmark data before modeling.
2. Save the exact config used for each run.
3. Freeze train/val/test organoid splits.
4. Run at least one non-graph baseline and one graph baseline before the dual-branch model.
5. Report ROC-AUC, PR-AUC, Brier score, accuracy, and F1.
6. Include calibration because the target is probabilistic.
7. Add hub interpretation and biological plausibility discussion.
8. Document missing pieces such as unavailable voxel masks or partial metadata.

## Reproducibility checklist

1. Keep `resolved_config.yaml` with every run.
2. Save `split_manifest.csv`.
3. Save `history.csv`.
4. Save test-set metrics and quick-look figures.
5. Record exact dataset URLs and access dates in the paper.

