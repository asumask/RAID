"""Evaluate a fixed RAID temperature checkpoint against its interpolation input.

Run from an environment where the matching RAID package is installed.
See docs/validation.md for assumptions, commands and metric definitions.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import DownscalingDataset, denormalize
from raid.data.file_utils import daily_filename
from raid.models import build_model


def metrics(error):
    """MAE, RMSE and signed bias for a finite, non-empty error vector."""
    return [float(np.abs(error).mean()),
            float(np.sqrt(np.mean(error ** 2))), float(error.mean())]


def pooled(rows):
    """Pool daily errors with cell-count weights (rather than averaging RMSE)."""
    a = np.asarray([r[1:] for r in rows], dtype=np.float64)
    weights = a[:, 0]
    return [float(np.average(a[:, 1], weights=weights)),
            float(np.sqrt(np.average(a[:, 3] ** 2, weights=weights))),
            float(np.average(a[:, 5], weights=weights))], [
            float(np.average(a[:, 2], weights=weights)),
            float(np.sqrt(np.average(a[:, 4] ** 2, weights=weights))),
            float(np.average(a[:, 6], weights=weights))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--processed-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='New directory; existing outputs are never overwritten.')
    parser.add_argument('--fold', required=True, help='For example val2025 or val2024.')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    with args.config.open() as f:
        raw = yaml.safe_load(f)
    apply_fold(raw, args.fold)
    raw['data']['processed_dir'] = str(args.processed_dir.resolve())
    train = set(raw['split']['train']['years'])
    val = set(raw['split']['val']['years'])
    test = set(raw['split'].get('test', {}).get('years', []))
    if not val or train & val or (train | val) & test:
        raise ValueError('Empty validation split or overlapping train/val/test years.')
    if raw['data']['input_vars'] != ['temp'] or raw['data']['target_vars'] != ['temp']:
        raise ValueError('This evaluator supports temperature-only inputs/targets.')
    cfg = dict_to_namespace(raw)
    with np.load(args.run_dir / 'norm_stats.npz', allow_pickle=False) as s:
        stats = {'temp': (float(s['temp_mean']), float(s['temp_std']))}
    if not np.isfinite(stats['temp']).all() or stats['temp'][1] <= 0:
        raise ValueError('Invalid training normalisation statistics.')
    ds = DownscalingDataset(cfg, years=sorted(val), stats=stats)
    expected = []
    for year in sorted(val):
        days = np.arange(np.datetime64(f'{year}-01-01'),
                         np.datetime64(f'{year + 1}-01-01'), dtype='datetime64[D]')
        expected.extend(str(d).replace('-', '') for d in days)
    if ds.dates != expected:
        raise ValueError('Validation dates must cover every day of the configured years.')
    sea = ds.sea_mask.numpy().astype(bool)
    if sea.ndim != 2 or not sea.any():
        raise ValueError('Expected a non-empty 2-D sea mask.')
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.run_dir / 'best.pt',
                                    map_location='cpu', weights_only=True), strict=True)
    model.eval()
    # Create only after loading and initial checks; never reuse an existing result folder.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'status': 'running', 'model': raw['model']['name'], 'fold': args.fold,
        'train_years': sorted(train), 'validation_years': sorted(val),
        'config_path': str(args.config.resolve()),
        'run_dir': str(args.run_dir.resolve()), 'normalisation': stats,
        'device': 'cpu', 'torch_version': str(torch.__version__),
        'numpy_version': np.__version__, 'resolved_config': raw,
        'note': 'Fixed checkpoint; validation used for selection, not independent testing.'
    }
    metadata_file = args.output_dir / 'evaluation_metadata.json'
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')
    sums = [np.zeros(sea.shape, dtype=np.float64) for _ in range(4)]
    rows = []
    start = time.perf_counter()
    print(f'START: {raw["model"]["name"]}; fold {args.fold}; no training.', flush=True)
    print(f'Sea cells: {sea.sum()} | Dates: {len(ds)}', flush=True)
    with (args.output_dir / 'daily_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'n_cells', 'MAE_model_C', 'MAE_baseline_C',
                         'RMSE_model_C', 'RMSE_baseline_C', 'bias_model_C', 'bias_baseline_C'])
        for i, date in enumerate(ds.dates):
            b = np.load(ds.dir_800m / daily_filename('temp', date), allow_pickle=False)
            t = np.load(ds.dir_160m / daily_filename('temp', date), allow_pickle=False)
            if b.shape != sea.shape or t.shape != sea.shape:
                raise ValueError(f'Grid mismatch: {date}')
            if not (np.isfinite(b[sea]).all() and np.isfinite(t[sea]).all()):
                raise ValueError(f'Non-finite sea input/target: {date}')
            x, _, _ = ds[i]
            with torch.no_grad():
                output = model(x.unsqueeze(0))
                if tuple(output.shape) != (1, 1, *sea.shape):
                    raise ValueError(f'Unexpected output shape: {tuple(output.shape)}')
                p = denormalize(output[0, 0].numpy(), *stats['temp'])
            if not np.isfinite(p[sea]).all():
                raise ValueError(f'Non-finite sea prediction: {date}')
            # Preserve the original evaluation's float32 denormalisation, then
            # calculate errors and accumulations in float64 against raw targets.
            em = p[sea].astype(np.float64) - t[sea].astype(np.float64)
            eb = b[sea].astype(np.float64) - t[sea].astype(np.float64)
            mm, mb = metrics(em), metrics(eb)
            row = [date, int(sea.sum()), mm[0], mb[0], mm[1], mb[1], mm[2], mb[2]]
            writer.writerow(row)
            f.flush()
            rows.append(row)
            for acc, values in zip(sums, [em**2, eb**2, np.abs(em), np.abs(eb)]):
                acc[sea] += values
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(ds):
                print(f'Processed {i+1}/{len(ds)} | {date} | '
                      f'elapsed {(time.perf_counter()-start)/60:.1f} minutes', flush=True)
    mm, mb = pooled(rows)
    label = {'cnn': 'CNN', 'unet': 'U-Net'}.get(raw['model']['name'], raw['model']['name'])
    with (args.output_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'year', 'n_dates', 'MAE_C', 'RMSE_C',
                         'bias_prediction_minus_target_C'])
        years = ';'.join(map(str, sorted(val)))
        writer.writerow([label, years, len(ds), *mm])
        writer.writerow(['Interpolation', years, len(ds), *mb])
    def masked(a):
        return np.where(sea, a, np.nan)
    np.savez_compressed(args.output_dir / 'spatial_metrics.npz',
                        rmse_model=masked(np.sqrt(sums[0] / len(ds))),
                        rmse_baseline=masked(np.sqrt(sums[1] / len(ds))),
                        mae_model=masked(sums[2] / len(ds)),
                        mae_baseline=masked(sums[3] / len(ds)), sea_mask=sea)
    metadata['status'] = 'complete'
    metadata['elapsed_seconds'] = time.perf_counter() - start
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')
    reduction = 100 * (1 - mm[1] / mb[1]) if mb[1] > 0 else float('nan')
    print(f'DONE | {label} RMSE: {mm[1]:.6f} C | baseline: {mb[1]:.6f} C', flush=True)
    print(f'Pooled RMSE reduction: {reduction:.2f}%', flush=True)


if __name__ == '__main__':
    main()
