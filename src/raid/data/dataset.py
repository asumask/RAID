from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .file_utils import daily_filename


## work on both numpy arrays and torch tensors
def normalize(arr, mean, std):
    return (arr - mean) / std


def denormalize(arr, mean, std):
    return arr * std + mean


def build_dataloaders(cfg):
    train_dataset = DownscalingDataset(
        cfg=cfg,
        years=cfg.split.train.years,
        patch_size=getattr(cfg.train, "patch_size", None),
    )
    val_dataset = DownscalingDataset(
        cfg=cfg,
        years=cfg.split.val.years,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    ## validate on the full grid (deterministic, comparable across epochs); full grids only fit batch 1
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class DownscalingDataset(Dataset):
    """Reads the preprocessed .npy files written by scripts/preprocess.py.

    x: (len(input_vars) + len(static_inputs), H, W)  normalised 800m-on-160m inputs + static fields
    y: (len(target_vars), H, W)                       normalised 160m targets
    mask: (1, H, W)                                  1 = sea, 0 = land, for masked_mse
    Land cells are set to 0 in x and y.
    """

    def __init__(self, cfg, years, patch_size=None):
        root = Path(cfg.data.processed_dir)
        self.dir_800m = root / "800m_on_160m_grid"
        self.dir_160m = root / "160m"
        static_dir = root / "static"
        self.patch_size = patch_size

        self.input_vars = cfg.data.input_vars
        self.target_vars = cfg.data.target_vars

        self.dates = self._find_dates(years)
        if not self.dates:
            raise RuntimeError(f"no preprocessed samples in {root} for years {years}")

        stats = np.load(static_dir / "preprocessing_stats.npz")
        self.mean = {v: float(stats[f"{v}_mean"]) for v in self.input_vars + self.target_vars}
        self.std = {v: float(stats[f"{v}_std"]) for v in self.input_vars + self.target_vars}

        self.sea_mask = torch.from_numpy(np.load(static_dir / "land_mask.npy").astype(bool))
        ## (1, H, W) float, 1 = sea, 0 = land; returned with every sample for masked_mse
        self.loss_mask = self.sea_mask[None].float()
        self.static = self._load_static(static_dir, cfg.data.static_inputs)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]

        x = np.stack([self._load(self.dir_800m, v, date) for v in self.input_vars])
        y = np.stack([self._load(self.dir_160m, v, date) for v in self.target_vars])

        x = torch.cat([torch.from_numpy(x), self.static])
        y = torch.from_numpy(y)
        mask = self.loss_mask

        if self.patch_size is not None:
            p = self.patch_size
            _, H, W = y.shape
            r = torch.randint(0, H - p + 1, (1,)).item()
            c = torch.randint(0, W - p + 1, (1,)).item()
            x = x[:, r:r + p, c:c + p]
            y = y[:, r:r + p, c:c + p]
            mask = mask[:, r:r + p, c:c + p]

        return x, y, mask

    def _load(self, data_dir, var, date):
        arr = normalize(np.load(data_dir / daily_filename(var, date)), self.mean[var], self.std[var])
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)

    def _find_dates(self, years):
        ## keep dates for which every input and target file exists
        years = {str(y) for y in years}
        dates = []
        for path in sorted(self.dir_800m.glob(daily_filename(self.input_vars[0], "[0-9]" * 8))):
            date = path.name[:8]
            if date[:4] not in years:
                continue
            if all((self.dir_800m / daily_filename(v, date)).exists() for v in self.input_vars) and all(
                (self.dir_160m / daily_filename(v, date)).exists() for v in self.target_vars
            ):
                dates.append(date)
        return dates

    def _load_static(self, static_dir, names):
        ## land_mask is used as 0/1; other static fields are standardised over sea cells
        sea = self.sea_mask.numpy()
        fields = []
        for name in names:
            arr = np.load(static_dir / f"{name}.npy").astype(np.float32)
            if name != "land_mask":
                arr = (arr - arr[sea].mean()) / arr[sea].std()
                arr[~sea] = 0.0
            fields.append(arr)
        if not fields:
            return torch.empty(0, *self.sea_mask.shape)
        return torch.from_numpy(np.stack(fields))
