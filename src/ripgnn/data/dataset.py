"""Torch datasets for temporal region-graph prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ripgnn.data.graph import RegionGraph


@dataclass(frozen=True)
class TemporalSample:
    dynamic: torch.Tensor
    static: torch.Tensor
    target: torch.Tensor
    burden_target: torch.Tensor
    organoid_id: str
    input_end_time: int
    target_time: int


class TemporalRegionDataset(Dataset[TemporalSample]):
    """Create one sample per organoid and rolling temporal window."""

    def __init__(
        self,
        snapshots: pd.DataFrame,
        graph: RegionGraph,
        dynamic_feature_cols: Sequence[str],
        target_col: str = "target_positive",
        burden_col: str = "burden",
        history_len: int = 3,
        organoid_ids: Sequence[str] | None = None,
    ) -> None:
        self.graph = graph
        self.dynamic_feature_cols = list(dynamic_feature_cols)
        self.target_col = target_col
        self.burden_col = burden_col
        self.history_len = history_len
        self.region_ids = graph.region_ids
        self.region_to_idx = {region: idx for idx, region in enumerate(self.region_ids)}
        frame = snapshots.copy()
        if organoid_ids is not None:
            frame = frame[frame["organoid_id"].isin(organoid_ids)].copy()
        self.samples = self._build_samples(frame)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TemporalSample:
        return self.samples[idx]

    def _build_samples(self, frame: pd.DataFrame) -> list[TemporalSample]:
        samples: list[TemporalSample] = []
        for organoid_id, group in frame.groupby("organoid_id"):
            ordered = group.sort_values(["time_idx", "region_id"]).copy()
            times = sorted(ordered["time_idx"].unique().tolist())
            pivot_dynamic = {
                feature: self._pivot_feature(ordered, feature)
                for feature in self.dynamic_feature_cols
            }
            pivot_target = self._pivot_feature(ordered, self.target_col)
            pivot_burden = self._pivot_feature(ordered, self.burden_col)
            for end_idx in range(self.history_len - 1, len(times) - 1):
                history_times = times[end_idx - self.history_len + 1 : end_idx + 1]
                target_time = times[end_idx + 1]
                dynamic_array = np.stack(
                    [
                        np.stack([pivot_dynamic[feature][time] for feature in self.dynamic_feature_cols], axis=-1)
                        for time in history_times
                    ],
                    axis=0,
                )
                target_array = pivot_target[target_time]
                burden_array = pivot_burden[target_time]
                samples.append(
                    TemporalSample(
                        dynamic=torch.tensor(dynamic_array, dtype=torch.float32),
                        static=self.graph.static_features.clone(),
                        target=torch.tensor(target_array, dtype=torch.float32),
                        burden_target=torch.tensor(burden_array, dtype=torch.float32),
                        organoid_id=str(organoid_id),
                        input_end_time=int(history_times[-1]),
                        target_time=int(target_time),
                    )
                )
        return samples

    def _pivot_feature(self, frame: pd.DataFrame, feature: str) -> dict[int, np.ndarray]:
        matrices: dict[int, np.ndarray] = {}
        for time_idx, group in frame.groupby("time_idx"):
            ordered = (
                group.set_index("region_id")
                .reindex(self.region_ids)
                .reset_index()
            )
            matrices[int(time_idx)] = ordered[feature].to_numpy(dtype=np.float32)
        return matrices


def collate_temporal_samples(samples: list[TemporalSample]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Stack temporal samples into a training batch."""
    return {
        "dynamic": torch.stack([sample.dynamic for sample in samples], dim=0),
        "static": torch.stack([sample.static for sample in samples], dim=0),
        "target": torch.stack([sample.target for sample in samples], dim=0),
        "burden_target": torch.stack([sample.burden_target for sample in samples], dim=0),
        "organoid_id": [sample.organoid_id for sample in samples],
        "input_end_time": [sample.input_end_time for sample in samples],
        "target_time": [sample.target_time for sample in samples],
    }
