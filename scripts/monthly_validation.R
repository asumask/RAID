# Monthly summaries of fixed-checkpoint RAID evaluation outputs for 2025.
# Source this file, then call run_monthly_validation(cnn_dir, unet_dir, output_dir).
# No code runs automatically when sourced. Requires base R only.

run_monthly_validation <-
function(cnn_dir, unet_dir, output_dir, expected_n_cells = 840706L) {
  
  cnn_file <- file.path(cnn_dir, "daily_metrics.csv")
  unet_file <- file.path(unet_dir, "daily_metrics.csv")
  if (file.exists(output_dir)) stop("Choose a new output directory.")
  stopifnot(file.exists(cnn_file), file.exists(unet_file))
  
  read_daily <- function(path) {
    
    d <- read.csv(path, stringsAsFactors = FALSE)
    
    required <- c(
      "date", "n_cells",
      "MAE_model_C", "MAE_baseline_C",
      "RMSE_model_C", "RMSE_baseline_C",
      "bias_model_C", "bias_baseline_C"
    )
    
    stopifnot(all(required %in% names(d)))
    
    d$Date <- as.Date(
      as.character(d$date),
      format = "%Y%m%d"
    )
    
    d <- d[order(d$Date), ]
    
    expected <- seq(
      as.Date("2025-01-01"),
      as.Date("2025-12-31"),
      by = "day"
    )
    
    stopifnot(
      nrow(d) == length(expected),
      !anyNA(d$Date),
      !anyDuplicated(d$Date),
      all(d$Date == expected),
      all(d$n_cells == expected_n_cells)
    )
    
    numeric_columns <- setdiff(required, "date")
    
    stopifnot(
      all(vapply(
        d[numeric_columns],
        function(x) all(is.finite(x)),
        logical(1)
      )),
      all(d$RMSE_model_C >= 0),
      all(d$MAE_model_C >= 0),
      all(d$RMSE_baseline_C > 0),
      all(d$MAE_baseline_C > 0)
    )
    
    d
  }
  
  cnn <- read_daily(cnn_file)
  unet <- read_daily(unet_file)
  
  stopifnot(all(cnn$Date == unet$Date))
  
  for (column in c(
    "MAE_baseline_C",
    "RMSE_baseline_C",
    "bias_baseline_C"
  )) {
    stopifnot(
      max(abs(cnn[[column]] - unet[[column]])) < 1e-6
    )
  }
  
  month_number <- as.integer(format(cnn$Date, "%m"))
  
  # محاسبهٔ MAE، RMSE و bias برای هر ماه و هر روش
  rows <- list()
  
  for (m in 1:12) {
    
    selected <- month_number == m
    
    for (method in c("Interpolation", "CNN", "U-Net")) {
      
      if (method == "Interpolation") {
        mae <- cnn$MAE_baseline_C[selected]
        rmse <- cnn$RMSE_baseline_C[selected]
        bias <- cnn$bias_baseline_C[selected]
        
      } else if (method == "CNN") {
        mae <- cnn$MAE_model_C[selected]
        rmse <- cnn$RMSE_model_C[selected]
        bias <- cnn$bias_model_C[selected]
        
      } else {
        mae <- unet$MAE_model_C[selected]
        rmse <- unet$RMSE_model_C[selected]
        bias <- unet$bias_model_C[selected]
      }
      
      rows[[length(rows) + 1L]] <- data.frame(
        month_number = m,
        month = month.abb[m],
        method = method,
        n_days = sum(selected),
        MAE_C = mean(mae),
        RMSE_C = sqrt(mean(rmse^2)),
        bias_C = mean(bias)
      )
    }
  }
  
  monthly <- do.call(rbind, rows)
  
  extract <- function(method, metric) {
    z <- monthly[monthly$method == method, ]
    z <- z[order(z$month_number), ]
    z[[metric]]
  }
  
  rmse <- cbind(
    Interpolation = extract("Interpolation", "RMSE_C"),
    CNN = extract("CNN", "RMSE_C"),
    `U-Net` = extract("U-Net", "RMSE_C")
  )
  
  mae <- cbind(
    Interpolation = extract("Interpolation", "MAE_C"),
    CNN = extract("CNN", "MAE_C"),
    `U-Net` = extract("U-Net", "MAE_C")
  )
  
  bias <- cbind(
    Interpolation = extract("Interpolation", "bias_C"),
    CNN = extract("CNN", "bias_C"),
    `U-Net` = extract("U-Net", "bias_C")
  )
  
  reduction <- cbind(
    CNN = 100 * (1 - rmse[, "CNN"] / rmse[, "Interpolation"]),
    `U-Net` = 100 * (
      1 - rmse[, "U-Net"] / rmse[, "Interpolation"]
    )
  )
  
  n_days <- tabulate(month_number, nbins = 12)
  
  rmse_table <- data.frame(
    Month = month.abb,
    Days = n_days,
    Interpolation_RMSE_C = rmse[, "Interpolation"],
    CNN_RMSE_C = rmse[, "CNN"],
    UNet_RMSE_C = rmse[, "U-Net"],
    CNN_RMSE_reduction_percent = reduction[, "CNN"],
    UNet_RMSE_reduction_percent = reduction[, "U-Net"],
    row.names = NULL
  )
  
  mae_table <- data.frame(
    Month = month.abb,
    Interpolation_MAE_C = mae[, "Interpolation"],
    CNN_MAE_C = mae[, "CNN"],
    UNet_MAE_C = mae[, "U-Net"],
    UNet_MAE_reduction_percent = 100 * (
      1 - mae[, "U-Net"] / mae[, "Interpolation"]
    ),
    row.names = NULL
  )
  
  bias_table <- data.frame(
    Month = month.abb,
    Interpolation_bias_C = bias[, "Interpolation"],
    CNN_bias_C = bias[, "CNN"],
    UNet_bias_C = bias[, "U-Net"],
    row.names = NULL
  )
  
  out <- output_dir
  if (!dir.create(out, recursive = TRUE)) stop("Cannot create output directory.")
  write.csv(
    monthly,
    file.path(out, "monthly_metrics_all_methods.csv"),
    row.names = FALSE
  )
  
  write.csv(
    rmse_table,
    file.path(out, "monthly_RMSE_table.csv"),
    row.names = FALSE
  )
  
  write.csv(
    mae_table,
    file.path(out, "monthly_MAE_table.csv"),
    row.names = FALSE
  )
  
  write.csv(
    bias_table,
    file.path(out, "monthly_bias_table.csv"),
    row.names = FALSE
  )
  
  # One final plot version, with all twelve month labels.
  plot_monthly_validation(out, expected_n_cells)

  comparison <- data.frame(
    Month = month.abb, Days = n_days,
    UNet_RMSE_reduction_percent = reduction[, "U-Net"],
    UNet_MAE_reduction_percent = mae_table$UNet_MAE_reduction_percent,
    UNet_days_with_lower_RMSE = vapply(1:12, function(m) {
      sel <- month_number == m
      sum(unet$RMSE_model_C[sel] < unet$RMSE_baseline_C[sel])
    }, integer(1))
  )
  write.csv(comparison, file.path(out, "monthly_UNet_comparison.csv"), row.names = FALSE)

  # Reconstruct annual metrics using numbers of days as weights.
  annual <- do.call(rbind, lapply(c("Interpolation", "CNN", "U-Net"), function(method) {
    z <- monthly[monthly$method == method, ]
    if (method == "Interpolation") {
      daily_mae <- cnn$MAE_baseline_C
      daily_rmse <- cnn$RMSE_baseline_C
      daily_bias <- cnn$bias_baseline_C
    } else {
      d <- if (method == "CNN") cnn else unet
      daily_mae <- d$MAE_model_C
      daily_rmse <- d$RMSE_model_C
      daily_bias <- d$bias_model_C
    }
    reconstructed <- c(weighted.mean(z$MAE_C, z$n_days),
      sqrt(weighted.mean(z$RMSE_C^2, z$n_days)),
      weighted.mean(z$bias_C, z$n_days))
    direct <- c(mean(daily_mae), sqrt(mean(daily_rmse^2)), mean(daily_bias))
    stopifnot(max(abs(reconstructed - direct)) < 1e-10)
    data.frame(method = method, year = 2025L, n_dates = nrow(cnn),
      MAE_C = direct[1], RMSE_C = direct[2], bias_prediction_minus_target_C = direct[3])
  }))
  base_rmse <- annual$RMSE_C[annual$method == "Interpolation"]
  annual$RMSE_reduction_percent <- 100 * (1 - annual$RMSE_C / base_rmse)
  write.csv(annual, file.path(out, "annual_metrics_check.csv"), row.names = FALSE)
  writeLines(c(
    "2025 validation only; no model inference, training or calibration.",
    "Same complete dates, sea-cell counts and baselines checked across methods.",
    "MAE and bias are means of daily metrics; pooled RMSE is sqrt(mean(daily RMSE^2)).",
    "Annual metrics reconstructed from months weighted by day counts and checked.",
    "Positive reduction means lower error than interpolation; no significance inference.",
    paste("CNN input:", normalizePath(cnn_dir)),
    paste("U-Net input:", normalizePath(unet_dir))
  ), file.path(out, "analysis_notes.txt"))
  writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
  cat("\nMonthly RMSE table:\n")
  print(rmse_table, row.names = FALSE, digits = 5)
  
  cat("\nMonthly MAE table:\n")
  print(mae_table, row.names = FALSE, digits = 5)
  
  cat("\nMonthly bias table:\n")
  print(bias_table, row.names = FALSE, digits = 5)
  
  cat("\nFiles saved in:\n", out, "\n")
  
  invisible(list(
    output_dir = out,
    RMSE = rmse_table,
    MAE = mae_table,
    bias = bias_table,
    comparison = comparison,
    annual = annual
  ))
}
plot_monthly_validation <- function(input_dir, n_cells) {
  r <- read.csv(
    file.path(input_dir, "monthly_RMSE_table.csv"),
    stringsAsFactors = FALSE
  )
  
  b <- read.csv(
    file.path(input_dir, "monthly_bias_table.csv"),
    stringsAsFactors = FALSE
  )
  
  stopifnot(
    nrow(r) == 12,
    nrow(b) == 12,
    all(r$Month == month.abb),
    all(b$Month == month.abb)
  )
  
  rmse <- cbind(
    Interpolation = r$Interpolation_RMSE_C,
    CNN = r$CNN_RMSE_C,
    `U-Net` = r$UNet_RMSE_C
  )
  
  reduction <- cbind(
    CNN = r$CNN_RMSE_reduction_percent,
    `U-Net` = r$UNet_RMSE_reduction_percent
  )
  
  bias <- cbind(
    Interpolation = b$Interpolation_bias_C,
    CNN = b$CNN_bias_C,
    `U-Net` = b$UNet_bias_C
  )
  
  stopifnot(
    all(is.finite(rmse)),
    all(is.finite(reduction)),
    all(is.finite(bias))
  )
  
  out <- input_dir
  draw <- function(
    values, filename, title, ylabel,
    note, colours, line_types, symbols,
    zero = FALSE
  ) {
    
    png(
      file.path(out, filename),
      width = 2200,
      height = 1600,
      res = 200,
      pointsize = 16,
      bg = "white"
    )
    
    tryCatch({
      
      par(
        mar = c(7.0, 5.2, 6.8, 1.2),
        mgp = c(3.0, 0.8, 0),
        family = "sans"
      )
      
      limits <- range(values, if (zero) 0)
      padding <- max(diff(limits) * 0.12, 0.005)
      
      matplot(
        1:12, values,
        type = "o",
        col = colours,
        lty = line_types,
        pch = symbols,
        lwd = 2.5,
        cex = 0.9,
        xlim = c(0.6, 12.4),
        ylim = limits + c(-padding, padding),
        xaxt = "n",
        xlab = "",
        ylab = ylabel,
        cex.axis = 1.05,
        cex.lab = 1.1,
        las = 1,
        bty = "l"
      )
      
      if (zero) {
        abline(
          h = 0,
          col = "#333333",
          lty = 3,
          lwd = 1.3
        )
      }
      
      # نام هر ماه جداگانه نوشته می‌شود:
      # R دیگر اجازه ندارد بعضی نام‌ها را حذف کند.
      axis(
        side = 1,
        at = 1:12,
        labels = FALSE
      )
      
      for (m in 1:12) {
        mtext(
          month.abb[m],
          side = 1,
          at = m,
          line = 1.0,
          cex = 1.0
        )
      }
      
      mtext(
        "Month in 2025",
        side = 1,
        line = 3.0,
        cex = 1.05
      )
      
      mtext(
        title,
        side = 3,
        line = 4.8,
        font = 2,
        cex = 1.15
      )
      
      mtext(
        sprintf("Validation: 2025 | Reference: 160 m model | %s sea cells",
          format(n_cells, big.mark = ",", scientific = FALSE)),
        side = 3,
        line = 3.3,
        cex = 0.9
      )
      
      legend(
        "top",
        inset = c(0, -0.13),
        legend = colnames(values),
        col = colours,
        lty = line_types,
        pch = symbols,
        lwd = 2.5,
        horiz = TRUE,
        bty = "n",
        xpd = NA,
        cex = 1
      )
      
      mtext(
        note,
        side = 1,
        line = 5.0,
        cex = 0.9
      )
      
      mtext(
        "Preliminary: boundary-cell review and interpolation provenance pending.",
        side = 1,
        line = 6.1,
        cex = 0.8
      )
      
    }, finally = {
      dev.off()
    })
  }
  
  draw(
    values = rmse,
    filename = "monthly_RMSE_comparison_val2025.png",
    title = "Monthly pooled surface-temperature RMSE",
    ylabel = "Pooled RMSE (°C)",
    note = "Monthly RMSE = square root of the mean squared daily RMSE.",
    colours = c("#D55E00", "#0072B2", "#009E73"),
    line_types = c(1, 2, 1),
    symbols = c(16, 17, 15)
  )
  
  draw(
    values = reduction,
    filename = "monthly_RMSE_reduction_val2025.png",
    title = "Monthly RMSE reduction relative to interpolation",
    ylabel = "Relative RMSE reduction (%)",
    note = "Positive: lower RMSE than interpolation. Negative: higher RMSE.",
    colours = c("#0072B2", "#009E73"),
    line_types = c(2, 1),
    symbols = c(17, 15),
    zero = TRUE
  )
  
  draw(
    values = bias,
    filename = "monthly_bias_comparison_val2025.png",
    title = "Monthly mean surface-temperature bias",
    ylabel = "Mean bias (°C)",
    note = "Bias = prediction minus target. Positive: warmer. Negative: cooler.",
    colours = c("#D55E00", "#0072B2", "#009E73"),
    line_types = c(1, 2, 1),
    symbols = c(16, 17, 15),
    zero = TRUE
  )
  
  cat("\nTables read from:\n", input_dir, "\n")
  cat("\nNew figures saved in:\n", out, "\n")
  
  print(list.files(out))
  
  invisible(out)
}
