# Monthly validation summaries and array-edge sensitivity

Author: Yasaman Malekiyourtchi. These scripts analyse saved CNN and U-Net
evaluation outputs; they do not train, calibrate or run either network.

## Scope and input contract

This version deliberately checks the complete 2025 validation year (365 dates),
with 840,706 evaluated sea cells per day by default. It is not a general test-year
or partial-year analysis. Checkpoints were selected using 2025 validation data,
so these summaries are not an independent test of generalisation.

Both input directories must describe the same dates, target and interpolation
version. Matching numerical baselines are checked, but cannot independently
establish preprocessing provenance. The 160 m reference is model output, not
observations. Outputs give equal weight to sea cells and days, without physical
cell-area weighting. No significance or causal conclusions are inferred.

Required files in each evaluation directory:

- `daily_metrics.csv`: `date` as YYYYMMDD, `n_cells`, `MAE_model_C`,
  `MAE_baseline_C`, `RMSE_model_C`, `RMSE_baseline_C`, `bias_model_C`,
  `bias_baseline_C`.
- For boundary sensitivity, `summary.csv`: one model row and one row whose
  `method` is `Interpolation`, with columns `MAE_C` and `RMSE_C`.
- For boundary sensitivity, `spatial_metrics.npz`: arrays `rmse_model`,
  `rmse_baseline`, `mae_model`, `mae_baseline`, `sea_mask`. Error maps must
  summarise all 365 dates with the same number of valid values at each sea cell.

These are the outputs of the earlier validation evaluator. Compatibility with
other evaluation output formats is not assumed. Do not rename unrelated outputs
into this schema without checking their definitions.

## Requirements

Monthly analysis uses base R (including its standard graphics and stats packages).
Boundary sensitivity additionally uses R `reticulate` and Python NumPy. Neither
script needs PyTorch or the RAID Python package. Use an existing working NumPy
environment; the scripts do not install packages or change Python environments.

## Run from the repository root in R

Replace the three example paths with your own paths. Output directories must not
already exist. Sourcing the scripts only defines functions; it does not run them.

```r
source("scripts/monthly_validation.R")
source("scripts/boundary_sensitivity.R")

cnn_dir <- "/path/to/saved/cnn/evaluation"
unet_dir <- "/path/to/saved/unet/evaluation"

monthly <- run_monthly_validation(
  cnn_dir = cnn_dir,
  unet_dir = unet_dir,
  output_dir = "/path/to/new/monthly/output"
)

boundary <- run_boundary_sensitivity(
  cnn_dir = cnn_dir,
  unet_dir = unet_dir,
  output_dir = "/path/to/new/boundary/output"
)
```

`expected_n_cells` can be explicitly changed for another mask, but the complete
2025 date requirement remains. Changes to the primary scientific evaluation mask
require separate justification; a parameter alone does not validate a new mask.

## Monthly outputs and definitions

- `monthly_metrics_all_methods.csv`: 36 rows, 12 months by three methods; MAE,
  pooled RMSE and bias in degrees Celsius.
- `monthly_RMSE_table.csv`, `monthly_MAE_table.csv`, `monthly_bias_table.csv`:
  publication tables with unrounded values.
- `monthly_UNet_comparison.csv`: monthly RMSE/MAE reductions and counts of days
  with strictly lower U-Net RMSE than interpolation. Ties do not count.
- `annual_metrics_check.csv`: annual metrics calculated directly from daily
  results and checked against day-weighted monthly reconstruction.
- Three PNGs: monthly RMSE comparison, relative RMSE reduction and mean bias.
  All 12 month labels, titles, units and method legends are retained.
- `analysis_notes.txt` and `sessionInfo.txt`: assumptions, input directories and
  R environment information.

For a period with D days and identical cell counts each day:

```text
MAE_period = mean(daily MAE)
RMSE_period = sqrt(mean(daily RMSE squared))
bias_period = mean(daily bias)
relative reduction (%) = 100 * (1 - model metric / interpolation metric)
```

Bias is prediction minus target. Positive bias means warmer estimates. Positive
reduction means smaller error. Period RMSE is not mean daily RMSE. Monthly and
annual reductions are not means of daily percentage reductions. Annual metrics
reconstructed from months weight months by their numbers of days, squaring RMSE
before averaging and taking the square root afterwards.

## Boundary outputs and definitions

`boundary_sensitivity.csv` reports full-mask results and temporary exclusions of
1, 3, 5 and 10 cell strips along all four rectangular array edges. `edge_widths`
can be supplied explicitly and must include zero. Width zero retains the full
sea mask. The nearest array-edge distance is zero on the outermost row/column.
Array edges are not coastlines. These strips do not diagnose physical boundary
conditions or declare any cell invalid.

For retained cells, pooled annual RMSE is `sqrt(mean(per-cell annual RMSE^2))`;
pooled MAE is the mean per-cell annual MAE. This requires equal temporal coverage
at every cell. The excluded strip's share of total squared error is the sum of
its per-cell RMSE squared divided by that sum over the full sea mask, times 100.
The common number of days cancels in this ratio.

All methods use identical retained cells. Full-mask RMSE/MAE must match saved
summaries within 1e-6 degrees Celsius. No daily/monthly subset results, bias maps,
or predictions can be recovered from these annual RMSE/MAE maps alone.

The scripts save assumptions and R session information. Existing output paths
are rejected. Input data and previous evaluation outputs are never modified.

## Validation status

The original functions ran successfully on the 2025 server outputs, reproducing
annual metrics, monthly tables and the boundary-sensitivity results. This version
consolidates the final plots and makes input/output paths explicit. It must be
rerun on those saved server outputs before claiming end-to-end verification of
this refactored version. The preparation environment does not have R or the
original daily/spatial input files, so no local R execution is claimed.

Results remain preliminary pending boundary-value review and interpolation
provenance confirmation. Temporary edge exclusions do not replace the primary
full-mask evaluation.

## Repository contents

Commit these scripts and this guide on the working branch. Do not commit model
weights, original .npy/.npz data, Python environments, or entire timestamped result
folders. After checking the new run, selected PNGs and small CSV tables can be
attached to the merge request as review evidence. The Confluence page remains
the presentation location for the figures and interpretations.
