# Surface-temperature validation

These additions evaluate the project's saved CNN and U-Net against the processed
800 m input on the 160 m grid. They do not train or calibrate either network.
The existing project scripts and model definitions are not replaced.

## Files

- `scripts/evaluate_validation.py`: CPU evaluation of one selected checkpoint.
- `scripts/plot_validation.R`: comparisons and four PNG figures from saved results.

Use the RAID source revision, architecture/configuration, processed inputs and
training normalisation statistics that belong to the checkpoint. Weight-shape
compatibility alone does not prove matching preprocessing or training provenance.
The saved run does not by itself establish which interpolation implementation
produced the inputs; confirm that separately before calling it bicubic.

## Environment

Use the Python environment in which the matching RAID package, NumPy, PyTorch
and PyYAML are installed. The plotting script needs R, reticulate, and NumPy in
reticulate's Python environment. That Python environment can differ from the
one used for inference. No additional Python packages are introduced.

## Evaluate a new validation run

Run from the repository root. Replace the example paths with local paths:

```bash
python scripts/evaluate_validation.py \
  --config configs/step1a_cnn.yaml \
  --fold val2025 \
  --run-dir /path/to/cnn/val2025 \
  --processed-dir /path/to/data/processed \
  --output-dir /path/to/new/cnn_validation_results
```

For U-Net use `configs/step1b_unet.yaml` and its corresponding checkpoint folder.
The fold identifies validation years and checks disjoint train/val/test years.
This script intentionally does not evaluate the test split. It requires complete
calendar-year coverage, including leap days, and refuses an existing output folder.
If interrupted, keep that folder as an incomplete run and use a new output path.
The metadata status becomes `complete` only after all outputs are written.

Outputs: `daily_metrics.csv`, `summary.csv`, `spatial_metrics.npz`, and
`evaluation_metadata.json`. Progress is printed to standard output; capture it
with your usual job runner if a log is needed. Raw predictions are not saved.
Evaluation is CPU-only, with two threads by default; `--threads` changes this.

## Plot existing results: no inference required

In RStudio, open `scripts/plot_validation.R` and click Source. Then run:

```r
plot_validation(
  cnn_dir = "/path/to/existing/cnn_validation_results",
  unet_dir = "/path/to/existing/unet_validation_results",
  output_dir = "/path/to/new/comparison_figures"
)
```

Alternatively:

```bash
Rscript scripts/plot_validation.R CNN_RESULTS UNET_RESULTS NEW_OUTPUT_DIR
```

The plotter accepts the original evaluation outputs. It checks full-year dates,
cell counts, shared masks and matching baseline metrics before comparison.
Two daily plots, two spatial maps, summary CSVs and figure notes are written.
Both maps have the same full symmetric colour scale and light-grey land mask.
Changing the display scale does not change metrics. Shared limits can make small
CNN differences look pale; do not infer improvement size from colour saturation
in maps made using separate scales.

## Definitions

All errors use prediction minus raw 160 m model target, in degrees Celsius,
over the same static sea mask. Non-finite sea values stop evaluation; suspect
boundary cells are not silently removed.

- MAE: mean absolute error over evaluated cell-days.
- Pooled RMSE: square root of mean squared error over evaluated cell-days.
  Daily squared RMSE is weighted by the daily cell count; counts are fixed here.
- Bias: mean signed error. Positive means warmer predictions than the target.
- Daily relative RMSE reduction: `100 * (1 - model_RMSE / baseline_RMSE)`.
- Pooled reduction: the same ratio using pooled RMSE, not mean daily reductions.
- Per-cell annual RMSE: root mean squared daily error at each cell.
- Map: per-cell model RMSE minus per-cell interpolation RMSE. Negative is blue;
  positive is orange. The fraction below zero counts cells, not weighted area.

The 160 m target is model output, not an observation-based truth. Validation
2025 selected the checkpoints, so reevaluation of 2025 is not independent testing.
Boundary-cell questions and interpolation provenance remain unresolved.
Descriptive differences do not establish statistical significance or causes.

## Existing 2025 results to reproduce

| Method | MAE (°C) | Pooled RMSE (°C) | Bias (°C) |
|---|---:|---:|---:|
| Interpolation | 0.2621069 | 0.4006312 | -0.002769692 |
| CNN | 0.2776116 | 0.4104096 | -0.007820878 |
| U-Net | 0.2614095 | 0.3734098 | +0.009077978 |

CNN: lower daily RMSE on 87/365 days; lower annual cell RMSE at 20.24% of cells.
U-Net: lower daily RMSE on 293/365 days; lower than CNN on 326/365 days;
lower annual cell RMSE at 69.47% of cells. Pooled reductions: CNN -2.44%,
U-Net +6.79%. Rounding and environment differences should be considered when
comparing reproduced values.

## Verification status and contribution

The Python module compiles and its pure metric aggregation is checked using
hand-calculated cases. This cleaned version has not been run against the server's
RAID installation/checkpoints. The R plotting script, including the corrected date comparison, was run successfully
by the user on the server against the existing 2025 CNN and U-Net outputs. It
reproduced the reported summary metrics, day counts and cell percentages. The
generated figures still require visual inspection before replacing website figures.
No new full-year inference is needed to check the plotting script.

Add these files on a separate branch and review the diff before a merge request
(GitLab) or pull request (GitHub). Do not commit data, model weights, virtual
environments, or generated result directories. Keep existing `.gitignore` rules;
add exclusions only for the actual output location if needed.
