# Array-edge sensitivity of fixed-checkpoint RAID evaluation outputs for 2025.
# Source this file, then call run_boundary_sensitivity(cnn_dir, unet_dir, output_dir).
# Requires reticulate with NumPy; does not import torch or raid.

run_boundary_sensitivity <-
function(cnn_dir, unet_dir, output_dir,
         edge_widths = c(0L, 1L, 3L, 5L, 10L),
         expected_n_cells = 840706L) {
  
  if (file.exists(output_dir)) stop("Choose a new output directory.")
  stopifnot(length(edge_widths) > 0, all(is.finite(edge_widths)),
    all(edge_widths >= 0), all(edge_widths == floor(edge_widths)), 0 %in% edge_widths)
  edge_widths <- sort(unique(edge_widths))
  # Confirm both annual maps represent the same complete validation period.
  expected_dates <- format(seq(as.Date("2025-01-01"), as.Date("2025-12-31"), by = "day"), "%Y%m%d")
  for (folder in c(cnn_dir, unet_dir)) {
    daily <- read.csv(file.path(folder, "daily_metrics.csv"), stringsAsFactors = FALSE)
    dates <- as.character(daily$date)
    stopifnot(length(dates) == 365L, !anyDuplicated(dates),
      all(sort(dates) == expected_dates), all(daily$n_cells == expected_n_cells))
  }
  cnn_file <- file.path(cnn_dir, "spatial_metrics.npz")
  unet_file <- file.path(unet_dir, "spatial_metrics.npz")
  
  stopifnot(
    file.exists(cnn_file),
    file.exists(unet_file)
  )
  
  np <- reticulate::import("numpy", convert = TRUE)
  
  read_maps <- function(path) {
    
    z <- np$load(path, allow_pickle = FALSE)
    on.exit(z$close())
    
    needed <- c(
      "rmse_model", "rmse_baseline",
      "mae_model", "mae_baseline",
      "sea_mask"
    )
    
    stopifnot(all(needed %in% z$files))
    
    maps <- setNames(
      lapply(needed, function(name) z[[name]]),
      needed
    )
    
    shape <- dim(maps$sea_mask)
    
    stopifnot(
      length(shape) == 2,
      all(vapply(
        maps,
        function(x) identical(dim(x), shape),
        logical(1)
      )),
      all(is.finite(maps$sea_mask)),
      all(maps$sea_mask %in% c(0, 1))
    )
    
    sea <- maps$sea_mask == 1
    
    for (name in setdiff(needed, "sea_mask")) {
      stopifnot(
        all(is.finite(maps[[name]][sea])),
        all(maps[[name]][sea] >= 0)
      )
    }
    
    maps$sea_mask <- sea
    maps
  }
  
  cnn <- read_maps(cnn_file)
  unet <- read_maps(unet_file)
  
  stopifnot(
    identical(dim(cnn$sea_mask), dim(unet$sea_mask)),
    all(cnn$sea_mask == unet$sea_mask)
  )
  
  sea <- cnn$sea_mask
  
  stopifnot(sum(sea) == expected_n_cells)
  
  # هر دو مدل باید با همان baseline مقایسه شده باشند
  stopifnot(
    max(abs(
      cnn$rmse_baseline[sea] -
        unet$rmse_baseline[sea]
    )) < 1e-6,
    
    max(abs(
      cnn$mae_baseline[sea] -
        unet$mae_baseline[sea]
    )) < 1e-6
  )
  
  H <- nrow(sea)
  W <- ncol(sea)
  
  # فاصله برحسب تعداد سلول تا نزدیک‌ترین لبهٔ آرایه
  # سلول روی خود لبه فاصلهٔ صفر دارد.
  rr <- row(sea)
  cc <- col(sea)
  
  edge_distance <- pmin(
    rr - 1L,
    H - rr,
    cc - 1L,
    W - cc
  )
  
  full_n <- sum(sea)
  
  # سهم نوار کنارگذاشته‌شده از مجموع خطای مربعی سالانه
  # چون تعداد روزها برای همهٔ سلول‌ها یکسان است،
  # ضریب تعداد روزها در این نسبت حذف می‌شود.
  squared_error_share <- function(rmse_map, excluded) {
    
    denominator <- sum(rmse_map[sea]^2)
    
    if (denominator == 0) {
      return(NA_real_)
    }
    
    100 * sum(rmse_map[excluded]^2) / denominator
  }
  
  rows <- lapply(edge_widths, function(width) {
    
    keep <- sea & edge_distance >= width
    excluded <- sea & !keep
    
    stopifnot(sum(keep) > 0)
    
    # RMSE تجمیعی از RMSE سالانهٔ هر سلول
    rmse_base <- sqrt(mean(cnn$rmse_baseline[keep]^2))
    rmse_cnn <- sqrt(mean(cnn$rmse_model[keep]^2))
    rmse_unet <- sqrt(mean(unet$rmse_model[keep]^2))
    
    # MAE تجمیعی از MAE سالانهٔ هر سلول
    mae_base <- mean(cnn$mae_baseline[keep])
    mae_cnn <- mean(cnn$mae_model[keep])
    mae_unet <- mean(unet$mae_model[keep])
    
    stopifnot(rmse_base > 0, mae_base > 0)
    
    data.frame(
      edge_width_cells = width,
      retained_sea_cells = sum(keep),
      excluded_sea_cells = sum(excluded),
      excluded_cells_percent =
        100 * sum(excluded) / full_n,
      
      RMSE_interpolation_C = rmse_base,
      RMSE_CNN_C = rmse_cnn,
      RMSE_UNet_C = rmse_unet,
      
      CNN_RMSE_reduction_percent =
        100 * (1 - rmse_cnn / rmse_base),
      
      UNet_RMSE_reduction_percent =
        100 * (1 - rmse_unet / rmse_base),
      
      MAE_interpolation_C = mae_base,
      MAE_CNN_C = mae_cnn,
      MAE_UNet_C = mae_unet,
      
      CNN_MAE_reduction_percent =
        100 * (1 - mae_cnn / mae_base),
      
      UNet_MAE_reduction_percent =
        100 * (1 - mae_unet / mae_base),
      
      excluded_squared_error_interpolation_percent =
        squared_error_share(cnn$rmse_baseline, excluded),
      
      excluded_squared_error_CNN_percent =
        squared_error_share(cnn$rmse_model, excluded),
      
      excluded_squared_error_UNet_percent =
        squared_error_share(unet$rmse_model, excluded)
    )
  })
  
  answer <- do.call(rbind, rows)
  
  # نتیجهٔ بدون کنارگذاشتن لبه‌ها باید با جدول سالانه یکی باشد.
  for (spec in list(
    list(folder = cnn_dir, rmse = answer$RMSE_CNN_C[1],
         mae = answer$MAE_CNN_C[1]),
    list(folder = unet_dir, rmse = answer$RMSE_UNet_C[1],
         mae = answer$MAE_UNet_C[1])
  )) {
    
    summary <- read.csv(
      file.path(spec$folder, "summary.csv"),
      stringsAsFactors = FALSE
    )
    
    model_row <- summary[
      summary$method != "Interpolation", ,
      drop = FALSE
    ]
    
    baseline_row <- summary[
      summary$method == "Interpolation", ,
      drop = FALSE
    ]
    
    stopifnot(
      nrow(model_row) == 1,
      nrow(baseline_row) == 1,
      
      abs(spec$rmse - model_row$RMSE_C) < 1e-6,
      abs(spec$mae - model_row$MAE_C) < 1e-6,
      
      abs(
        answer$RMSE_interpolation_C[1] -
          baseline_row$RMSE_C
      ) < 1e-6,
      
      abs(
        answer$MAE_interpolation_C[1] -
          baseline_row$MAE_C
      ) < 1e-6
    )
  }
  
  out <- output_dir
  if (!dir.create(out, recursive = TRUE)) stop("Cannot create output directory.")
  write.csv(
    answer,
    file.path(out, "boundary_sensitivity.csv"),
    row.names = FALSE
  )
  
  writeLines(
    c(
      "Diagnostic sensitivity analysis; no model inference or training.",
      paste("CNN input:", normalizePath(cnn_dir)),
      paste("U-Net input:", normalizePath(unet_dir)),
      "All metrics use the existing 365-day validation maps for 2025.",
      "Widths refer to strips along all four rectangular array edges.",
      "Array-edge distance is not distance to coast.",
      "The same retained sea cells are used for all three methods.",
      "Each retained cell and day has equal weight.",
      "Excluded cells are not declared invalid by this analysis.",
      "These scenarios do not replace the primary evaluation mask.",
      "Daily or monthly subset performance cannot be recovered from annual maps.",
      "Full-mask RMSE and MAE were checked against saved annual summaries."
    ),
    file.path(out, "analysis_notes.txt")
  )
  
  cat(
    "\nChecks passed: matching masks and baselines;",
    "full-mask metrics match saved summaries.\n"
  )
  
  cat("\n--- Boundary sensitivity: RMSE ---\n")
  
  print(
    answer[c(
      "edge_width_cells",
      "retained_sea_cells",
      "excluded_cells_percent",
      "RMSE_interpolation_C",
      "RMSE_CNN_C",
      "RMSE_UNet_C",
      "CNN_RMSE_reduction_percent",
      "UNet_RMSE_reduction_percent"
    )],
    row.names = FALSE,
    digits = 6
  )
  
  cat("\n--- Boundary sensitivity: MAE reductions ---\n")
  
  print(
    answer[c(
      "edge_width_cells",
      "CNN_MAE_reduction_percent",
      "UNet_MAE_reduction_percent"
    )],
    row.names = FALSE,
    digits = 6
  )
  
  cat("\n--- Excluded strip: share of total squared error (%) ---\n")
  
  print(
    answer[c(
      "edge_width_cells",
      "excluded_cells_percent",
      "excluded_squared_error_interpolation_percent",
      "excluded_squared_error_CNN_percent",
      "excluded_squared_error_UNet_percent"
    )],
    row.names = FALSE,
    digits = 6
  )
  
  writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))

  cat("\nResults saved in:\n", out, "\n")
  
  invisible(list(
    results = answer,
    output_dir = out
  ))
}
