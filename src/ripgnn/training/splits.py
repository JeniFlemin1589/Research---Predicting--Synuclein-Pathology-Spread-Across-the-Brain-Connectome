"""Train/validation/test split helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GroupSplit:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def split_organoids(
    snapshots: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> GroupSplit:
    """Split organoids into disjoint train/validation/test groups."""
    organoid_ids = sorted(snapshots["organoid_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(organoid_ids)
    n_total = len(organoid_ids)
    n_train = max(1, int(n_total * train_fraction))
    n_val = max(1, int(n_total * val_fraction))
    train_ids = organoid_ids[:n_train]
    val_ids = organoid_ids[n_train : n_train + n_val]
    test_ids = organoid_ids[n_train + n_val :]
    if not test_ids:
        test_ids = val_ids[-1:]
        val_ids = val_ids[:-1]
    return GroupSplit(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def assert_split_integrity(split: GroupSplit) -> None:
    """Verify that no organoid id appears in multiple splits."""
    train_set = set(split.train_ids)
    val_set = set(split.val_ids)
    test_set = set(split.test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Split leakage detected between train/validation/test organoid ids")

