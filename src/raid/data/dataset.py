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



def trim_edges(mask, n):
    """Copy of a 2-D sea mask with a frame of n cells along every grid edge set to False/0.

    Used to keep the 160m model's open-boundary zone out of the loss; n = 0 returns an unchanged copy.
    """
    mask = mask.clone() if isinstance(mask, torch.Tensor) else mask.copy()
    if n > 0:
        mask[:n, :] = 0
        mask[-n:, :] = 0
        mask[:, :n] = 0
        mask[:, -n:] = 0
    return mask


def build_dataloaders(cfg):
    ## normalisation stats come from this fold's training years only; val reuses them
    ## no val years (--fold full) -> val_loader is None
    ## train.edge_pixels (default 0) drops that many cells along each grid edge from train and val loss
    edge_pixels = getattr(cfg.train, "edge_pixels", 0)
    train_dataset = DownscalingDataset(cfg=cfg, years=cfg.split.train.years,
                                       patch_size=getattr(cfg.train, "patch_size", None),
                                       edge_pixels=edge_pixels)

    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "norm_stats.npz",
             **{f"{v}_mean": m for v, (m, s) in train_dataset.stats.items()},
             **{f"{v}_std":  s for v, (m, s) in train_dataset.stats.items()})

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True)
    if not cfg.split.val.years:
        return train_loader, None
    ## validate on the full grid (deterministic, comparable across epochs); full grids only fit batch 1
    val_dataset = DownscalingDataset(cfg=cfg, years=cfg.split.val.years, stats=train_dataset.stats,
                                     edge_pixels=edge_pixels)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=True)

    return train_loader, val_loader


def compute_stats(dates, dir_800m, dir_160m, input_vars, target_vars, sea_mask):
    """Mean/std per variable over sea cells of the given (training) dates."""
    sea = sea_mask.numpy()
    stats = {}
    for v in dict.fromkeys(input_vars + target_vars):          # unique, ordered
        d = dir_160m if v in target_vars else dir_800m
        n, s, ss = 0, 0.0, 0.0
        for date in dates:
            a = np.load(d / daily_filename(v, date))[sea].astype(np.float64)
            a = a[np.isfinite(a)]
            n += a.size; s += a.sum(); ss += (a * a).sum()
        mean = s / n
        stats[v] = (mean, float(np.sqrt(ss / n - mean ** 2)))
    return stats


class DownscalingDataset(Dataset):
    """Reads the preprocessed .npy files written by scripts/preprocess.py.

    x: (len(input_vars) + len(static_inputs), H, W)  normalised 800m-on-160m inputs + static fields
    y: (len(target_vars), H, W)                       normalised 160m targets
    mask: (1, H, W)                                  1 = sea, 0 = land, for masked_mse
    Land cells are set to 0 in x and y. edge_pixels > 0 also sets a frame of that many cells
    along each grid edge to 0 in mask (inputs keep them); sea_mask itself is not trimmed.
    """

    def __init__(self, cfg, years, patch_size=None, crops_per_day=20, stats=None, edge_pixels=0):
        root = Path(cfg.data.processed_dir)
        self.dir_800m = root / "800m_on_160m_grid"
        self.dir_160m = root / "160m"
        static_dir = root / "static"
        self.patch_size = patch_size
        self.crops_per_day = crops_per_day

        self.input_vars = cfg.data.input_vars
        self.target_vars = cfg.data.target_vars

        self.dates = self._find_dates(years)
        if not self.dates:
            raise RuntimeError(f"no preprocessed samples in {root} for years {years}")

        self.sea_mask = torch.from_numpy(np.load(static_dir / "land_mask.npy").astype(bool))
        self.loss_mask = trim_edges(self.sea_mask, edge_pixels)[None].float()

        if stats is None:      # training set: compute from its own dates
            stats = compute_stats(self.dates, self.dir_800m, self.dir_160m,
                                  self.input_vars, self.target_vars, self.sea_mask)
        self.stats = stats
        self.mean = {v: m for v, (m, s) in stats.items()}
        self.std  = {v: s for v, (m, s) in stats.items()}

        self.static = self._load_static(static_dir, cfg.data.static_inputs)


    def __len__(self):
        return len(self.dates) * (self.crops_per_day if self.patch_size else 1)


    def __getitem__(self, idx):
        date = self.dates[idx % len(self.dates)]

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
        ## land_mask is used as 0/1; other static fields are standardised over sea cells.
        ## distance_to_coast is log1p'd first: the error changes over the first few km, but
        ## distances reach ~100 km, so plain standardising would squash the coastal range
        sea = self.sea_mask.numpy()
        fields = []
        for name in names:
            arr = np.load(static_dir / f"{name}.npy").astype(np.float32)
            if name == "distance_to_coast":
                arr = np.log1p(arr)
            if name != "land_mask":
                arr = (arr - arr[sea].mean()) / arr[sea].std()
                arr[~sea] = 0.0
            fields.append(arr)
        if not fields:
            return torch.empty(0, *self.sea_mask.shape)
        return torch.from_numpy(np.stack(fields))
