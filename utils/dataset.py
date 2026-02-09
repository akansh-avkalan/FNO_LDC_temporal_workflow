# utils/dataset.py

import os
from typing import Tuple, List
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# -------------------------------
# Single-geometry dataset
# -------------------------------

class LDCGeometryDataset(Dataset):
    """
    Dataset for a single LDC geometry (harmonics / nurbs / skelneton).
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        indices: List[int],
        dtype: torch.dtype = torch.float32,
    ):
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"X file not found: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Y file not found: {y_path}")

        self.X = np.load(x_path)
        self.Y = np.load(y_path)

        # Handle npz format
        if isinstance(self.X, np.lib.npyio.NpzFile):
            self.X = self.X[list(self.X.keys())[0]]
        if isinstance(self.Y, np.lib.npyio.NpzFile):
            self.Y = self.Y[list(self.Y.keys())[0]]

        assert self.X.shape[0] == self.Y.shape[0], "X/Y sample count mismatch"

        self.indices = indices
        self.dtype = dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        x = torch.tensor(self.X[real_idx], dtype=self.dtype)
        y = torch.tensor(self.Y[real_idx], dtype=self.dtype)

        # Shapes are already channel-first:
        # X: (3, 128, 128), Y: (4, 128, 128)
        return x, y


# -------------------------------
# Split helper
# -------------------------------

def split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """
    Deterministic split of indices.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    return (
        indices[:train_end].tolist(),
        indices[train_end:val_end].tolist(),
        indices[val_end:].tolist(),
    )


# -------------------------------
# Combined dataset + loaders
# -------------------------------

def create_dataloaders(
    dataset_root: str,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,  # safer for Windows + Temporal
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders by
    splitting EACH geometry independently, then combining.
    """

    geometries = ["harmonics", "nurbs", "skelneton"]

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for geom in geometries:
        x_path = os.path.join(
            dataset_root, f"{geom}_lid_driven_cavity_X.npz"
        )
        y_path = os.path.join(
            dataset_root, f"{geom}_lid_driven_cavity_Y.npz"
        )

        # All geometries have 1000 samples
        train_idx, val_idx, test_idx = split_indices(
            n_samples=1000,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        train_datasets.append(
            LDCGeometryDataset(x_path, y_path, train_idx)
        )
        val_datasets.append(
            LDCGeometryDataset(x_path, y_path, val_idx)
        )
        test_datasets.append(
            LDCGeometryDataset(x_path, y_path, test_idx)
        )

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
