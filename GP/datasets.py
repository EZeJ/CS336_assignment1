from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Dataset:
    """
    Simple container for GP fitness data.

    Attributes:
        inputs: (n_samples, n_features) array of inputs.
        targets: (n_samples,) array of target outputs.
        feature_names: optional list of names for each feature.
    """

    inputs: np.ndarray
    targets: np.ndarray
    feature_names: list[str] | None = None

    def split(self, frac: float = 0.8, seed: int | None = None) -> tuple["Dataset", "Dataset"]:
        """Split into train/val with a given fraction for train."""
        rng = np.random.default_rng(seed)
        n = self.inputs.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(frac * n)
        train_idx, val_idx = idx[:cut], idx[cut:]
        return (
            Dataset(self.inputs[train_idx], self.targets[train_idx], self.feature_names),
            Dataset(self.inputs[val_idx], self.targets[val_idx], self.feature_names),
        )


def load_dataset(path: str | Path) -> Dataset:
    """
    Load a dataset from NPZ with keys: inputs (N,F), targets (N,), optional feature_names.
    This keeps format flexible while remaining simple.
    """
    data = np.load(path, allow_pickle=True)
    inputs = data["inputs"]
    targets = data["targets"]
    feature_names = None
    if "feature_names" in data:
        feature_names = data["feature_names"].tolist()
    return Dataset(inputs=inputs, targets=targets, feature_names=feature_names)
