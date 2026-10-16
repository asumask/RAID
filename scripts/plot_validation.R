# Compare saved temperature validation results without running either network.
# Rscript scripts/plot_validation.R CNN_DIR UNET_DIR NEW_OUTPUT_DIR
# Requires reticulate with NumPy available in its Python environment.

plot_validation <- function(cnn_dir, unet_dir, output_dir) {
  if (!requireNamespace('reticulate', quietly = TRUE)) stop('Install reticulate first.')
  if (dir.exists(output_dir) || file.exists(output_dir)) stop('Choose a new output directory.')
  np <- reticulate::import('numpy', convert = TRUE)
  read_run <- function(folder) {
    d <- read.csv(file.path(folder, 'daily_metrics.csv'), stringsAsFactors = FALSE)
    required <- c('date', 'n_cells', 'MAE_model_C', 'MAE_baseline_C',
                  'RMSE_model_C', 'RMSE_baseline_C', 'bias_model_C', 'bias_baseline_C')
    if (!all(required %in% names(d))) stop('Missing daily metric columns.')
    d$Date <- as.Date(as.character(d$date), '%Y%m%d')
    d <- d[order(d$Date), ]
    stopifnot(!anyNA(d$Date), !anyDuplicated(d$Date),
              all(is.finite(as.matrix(d[required[-1]]))),
              all(d$n_cells > 0), all(d$RMSE_baseline_C > 0))
    year <- unique(format(d$Date, '%Y'))
    if (length(year) != 1L) stop('Plot one complete validation year at a time.')
    expected <- seq(as.Date(paste0(year, '-01-01')),
                    as.Date(paste0(year, '-12-31')), by = 'day')
    stopifnot(length(d$Date) == length(expected), all(d$Date == expected), length(unique(d$n_cells)) == 1L)
    z <- np$load(file.path(folder, 'spatial_metrics.npz'), allow_pickle = FALSE)
    maps <- tryCatch(list(model = z[['rmse_model']], baseline = z[['rmse_baseline']],
                          sea = z[['sea_mask']] == 1), finally = z$close())
    stopifnot(is.matrix(maps$model), !anyNA(maps$sea),
              identical(dim(maps$model), dim(maps$baseline)),
              identical(dim(maps$model), dim(maps$sea)),
              sum(maps$sea) == d$n_cells[1],
              all(is.finite(maps$model[maps$sea])),
              all(is.finite(maps$baseline[maps$sea])))
    list(daily = d, maps = maps, year = year)
  }
  cnn <- read_run(cnn_dir)
  unet <- read_run(unet_dir)
  stopifnot(identical(cnn$daily$Date, unet$daily$Date),
            identical(cnn$daily$n_cells, unet$daily$n_cells),
            identical(cnn$maps$sea, unet$maps$sea))
  for (v in c('MAE_baseline_C', 'RMSE_baseline_C', 'bias_baseline_C')) {
    if (max(abs(cnn$daily[[v]] - unet$daily[[v]])) > 1e-6) stop('Daily baselines differ.')
  }
  sea <- cnn$maps$sea
  if (max(abs(cnn$maps$baseline[sea] - unet$maps$baseline[sea])) > 1e-6)
    stop('Spatial baselines differ.')
  # Confirm spatial and daily outputs describe the same pooled errors.
  for (run in list(cnn, unet)) {
    stopifnot(abs(sqrt(mean(run$maps$model[sea]^2)) -
                    sqrt(mean(run$daily$RMSE_model_C^2))) < 1e-6)
  }
  dir.create(output_dir, recursive = TRUE)
  year <- cnn$year
  d <- cnn$daily
  u <- unet$daily
  root_mean_square <- function(x) sqrt(mean(x^2))
  rmse <- c(root_mean_square(d$RMSE_baseline_C), root_mean_square(d$RMSE_model_C),
            root_mean_square(u$RMSE_model_C))
  summary <- data.frame(method = c('Interpolation', 'CNN', 'U-Net'), year = year,
                        n_dates = nrow(d),
                        MAE_C = c(mean(d$MAE_baseline_C), mean(d$MAE_model_C), mean(u$MAE_model_C)),
                        RMSE_C = rmse,
                        bias_prediction_minus_target_C = c(mean(d$bias_baseline_C),
                          mean(d$bias_model_C), mean(u$bias_model_C)))
  summary$RMSE_reduction_percent <- 100 * (1 - rmse / rmse[1])
  summary$days_with_lower_RMSE <- c(NA, sum(d$RMSE_model_C < d$RMSE_baseline_C),
                                   sum(u$RMSE_model_C < u$RMSE_baseline_C))
  delta_cnn <- cnn$maps$model - cnn$maps$baseline
  delta_unet <- unet$maps$model - unet$maps$baseline
  summary$cells_with_lower_RMSE_percent <- c(NA, 100 * mean(delta_cnn[sea] < 0),
                                            100 * mean(delta_unet[sea] < 0))
  write.csv(summary, file.path(output_dir, 'comparison_summary.csv'), row.names = FALSE)
  write.csv(data.frame(year = year, n_dates = nrow(d),
    days_UNet_lower_RMSE_than_CNN = sum(u$RMSE_model_C < d$RMSE_model_C)),
    file.path(output_dir, 'paired_daily_comparison.csv'), row.names = FALSE)
  colours <- c('#D55E00', '#0072B2', '#009E73')
  ticks <- seq(as.Date(paste0(year, '-01-01')), as.Date(paste0(year, '-12-01')), by = 'month')
  months <- c('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
  subtitle <- paste('Validation:', year, '| Reference: 160 m model | Fixed checkpoints')
  save_png <- function(name, draw) {
    png(file.path(output_dir, name), width = 2600, height = 1600, res = 200, bg = 'white')
    on.exit(dev.off(), add = TRUE)
    draw()
  }
  save_png(paste0('CNN_UNet_interpolation_daily_RMSE_val', year, '.png'), function() {
    par(mar = c(5, 5, 5, 2))
    plot(d$Date, d$RMSE_baseline_C, type = 'l', col = colours[1], lwd = 2,
      ylim = range(d$RMSE_baseline_C, d$RMSE_model_C, u$RMSE_model_C), xaxt = 'n',
      xlab = paste('Date in', year), ylab = 'Surface-temperature RMSE (°C)',
      main = 'Daily surface-temperature RMSE: interpolation, CNN and U-Net', cex.main = .95)
    lines(d$Date, d$RMSE_model_C, col = colours[2], lwd = 2)
    lines(u$Date, u$RMSE_model_C, col = colours[3], lwd = 2)
    axis(1, at = ticks, labels = months)
    mtext(subtitle, side = 3, line = .5, cex = .8)
    legend('topright', sprintf('%s | pooled RMSE = %.3f °C', summary$method, rmse),
      col = colours, lty = 1, lwd = 2, bty = 'n', cex = .8)
  })
  reductions <- cbind(100 * (1 - d$RMSE_model_C/d$RMSE_baseline_C),
                      100 * (1 - u$RMSE_model_C/u$RMSE_baseline_C))
  save_png(paste0('CNN_UNet_relative_RMSE_reduction_val', year, '.png'), function() {
    par(mar = c(5, 5, 5, 2))
    plot(d$Date, reductions[,1], type = 'l', col = colours[2], lwd = 2,
      ylim = range(0, reductions), xaxt = 'n', xlab = paste('Date in', year),
      ylab = 'Relative RMSE reduction (%)', main = 'Daily relative RMSE reduction against interpolation')
    lines(d$Date, reductions[,2], col = colours[3], lwd = 2)
    abline(h = 0, col = '#333333', lty = 2)
    axis(1, at = ticks, labels = months)
    mtext(subtitle, side = 3, line = .5, cex = .8)
    legend('bottomright', sprintf('%s | pooled reduction = %+.2f%%',
      summary$method[2:3], summary$RMSE_reduction_percent[2:3]),
      col = colours[2:3], lwd = 2, lty = 1, bty = 'n', cex = .8)
  })
  # Both maps use the full combined range: no clipping, identical colour meaning.
  limit <- max(abs(c(delta_cnn[sea], delta_unet[sea])))
  if (limit == 0) limit <- 1e-6
  palette <- colorRampPalette(c('#0072B2', '#FFFFFF', '#D55E00'))(201)
  breaks <- seq(-limit, limit, length.out = 202)
  for (method in c('CNN', 'UNet')) {
    delta <- if (method == 'CNN') delta_cnn else delta_unet
    delta[!sea] <- NA_real_
    label <- if (method == 'CNN') 'CNN' else 'U-Net'
    save_png(paste0(method, '_minus_interpolation_RMSE_map_val', year, '.png'), function() {
      layout(matrix(c(1,2), nrow = 1), widths = c(6,1))
      par(mar = c(5,5,5,1))
      plot(NA, xlim = c(.5,ncol(delta)+.5), ylim = c(.5,nrow(delta)+.5), asp = 1,
        xaxs = 'i', yaxs = 'i', xlab = 'Column index', ylab = 'Row index',
        main = paste('Surface-temperature RMSE difference:', label, 'minus interpolation'), cex.main = .95)
      rect(.5, .5, ncol(delta)+.5, nrow(delta)+.5, col = '#E8E8E8', border = NA)
      image(seq_len(ncol(delta)), seq_len(nrow(delta)), t(delta), col = palette,
            breaks = breaks, useRaster = TRUE, add = TRUE)
      box()
      mtext(paste('Validation:', year, '| Per-cell RMSE over', nrow(d), 'days | Shared colour scale'),
            side = 3, line = .5, cex = .75)
      par(mar = c(5,1,5,5))
      values <- seq(-limit, limit, length.out = 201)
      image(c(0,1), values, matrix(rep(values, each = 2), nrow = 2),
            col = palette, breaks = breaks, xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')
      axis(4, las = 1, cex.axis = .8)
      mtext('RMSE difference (°C)', side = 4, line = 3.5)
      box()
    })
  }
  writeLines(c('Preliminary validation of selected, fixed checkpoints.',
    'Boundary-cell review and input interpolation provenance remain unresolved.',
    'Daily reduction = 100 * (1 - daily model RMSE / daily baseline RMSE).',
    'Pooled reduction uses pooled RMSE, not the mean daily percentage.',
    'Maps use identical full-range colour limits; grey denotes outside the mask.',
    paste('Shared symmetric colour limit (°C):', format(limit, digits = 10)),
    'Grid-cell percentages are not area-weighted and do not establish significance.'),
    file.path(output_dir, 'figure_notes.txt'))
  print(summary)
  cat('\nFigures saved in:\n', normalizePath(output_dir), '\n')
  invisible(summary)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 3L) {
    plot_validation(args[1], args[2], args[3])
  } else if (!interactive()) {
    stop('Usage: Rscript scripts/plot_validation.R CNN_DIR UNET_DIR NEW_OUTPUT_DIR')
  } else {
    message('Loaded. Call plot_validation(cnn_dir, unet_dir, new_output_dir).')
  }
}
