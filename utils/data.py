"""Dataset helpers and custom Dataset classes for the pdl_ss26 course."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).parent.parent / "data"


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NumpyDataset(Dataset):
    """Thin wrapper around numpy arrays to use with DataLoader."""

    def __init__(self, X, y, transform=None):
        import numpy as np  # local import to keep torch-only usage possible

        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]
