# Curated package: never execute against the original project directory.
mpn_run_guard <- Sys.getenv("MPN_RUN_ROOT")
if (!nzchar(mpn_run_guard) || !file.exists(file.path(mpn_run_guard, ".mpn_reproduction_run")))
  stop("Use run_pipeline.py with a new marked run directory.")
if (normalizePath(Sys.getenv("MPN_PROJECT_ROOT"), mustWork=TRUE) != normalizePath(mpn_run_guard, mustWork=TRUE))
  stop("Project and run roots must match.")
mpn_args_guard <- commandArgs(trailingOnly=TRUE)
if (length(mpn_args_guard) && normalizePath(mpn_args_guard[1],mustWork=TRUE) != normalizePath(mpn_run_guard,mustWork=TRUE))
  stop("Argument root differs from protected run root.")
#!/usr/bin/env Rscript

# Targeted reviewer-requested analyses for the SEER-MPN manuscript.
# This script is intentionally limited to:
#   1. comparing the included cohort with the 2,565 patients excluded after
#      restriction to index diagnoses in 2004-2023; and
#   2. refitting the existing interval-specific baseline Cox models with White
#      as the race reference category.
# It reads existing patient-level derived files but writes aggregate results only.

options(stringsAsFactors = FALSE, warn = 1)

SEED <- 20260723L
set.seed(SEED)

PROJECT_ROOT <- normalizePath(
  Sys.getenv("MPN_PROJECT_ROOT", unset = getwd()),
  mustWork = TRUE
)
INPUT_MASTER <- file.path(PROJECT_ROOT, "MPN", "patient_master_v0.4_2000_2023.csv")
INPUT_ML <- file.path(PROJECT_ROOT, "ML", "data", "ml_competing_risk_v0.4_2004_2023.csv")
OUTPUT_DIR <- file.path(
  PROJECT_ROOT, "_codex_workspace", "25_income_reanalysis_20261001", "exclusion_outputs"
)
dir.create(file.path(OUTPUT_DIR, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "logs"), recursive = TRUE, showWarnings = FALSE)

required_packages <- c(
  "data.table", "dplyr", "stringr", "survival", "broom", "ggplot2",
  "scales", "cowplot"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Missing required packages: ", paste(missing_packages, collapse = ", "))
}
if (!file.exists(INPUT_MASTER)) stop("Input not found: ", INPUT_MASTER)
if (!file.exists(INPUT_ML)) stop("Input not found: ", INPUT_ML)
suppressPackageStartupMessages(library(survival))

write_csv <- function(x, filename) {
  data.table::fwrite(
    data.table::as.data.table(x),
    file.path(OUTPUT_DIR, "tables", filename),
    na = ""
  )
}

normalise_text <- function(x) {
  stringr::str_to_lower(stringr::str_squish(as.character(x)))
}

parse_income <- function(x) {
  source(file.path(PROJECT_ROOT, "_codex_workspace", "21_sequence_income_integrity_20261001", "income_recode_fix.R"), local = TRUE)
  parse_income(x)
}

recode_urban <- function(x) {
  z <- normalise_text(x)
  dplyr::case_when(
    stringr::str_detect(z, "nonmetropolitan") ~ "Nonmetropolitan",
    stringr::str_detect(z, "metropolitan") ~ "Metropolitan",
    TRUE ~ "Unknown"
  )
}

recode_chemo <- function(x) {
  z <- normalise_text(x)
  dplyr::case_when(
    z %in% c("yes", "1", "given") ~ "Yes",
    stringr::str_detect(z, "no\\s*/?\\s*(unknown|unk)") ~ "No/Unknown",
    stringr::str_detect(z, "none\\s*/?\\s*(unknown|unk)") ~ "No/Unknown",
    z %in% c("no", "0", "none") ~ "No",
    TRUE ~ "Unknown"
  )
}

fmt_p <- function(p) {
  ifelse(
    is.na(p), NA_character_,
    ifelse(p < 1e-4, "<0.0001", format.pval(p, digits = 4, eps = 1e-4))
  )
}

tidy_cox <- function(fit, model, interval = "All follow-up") {
  broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) |>
    dplyr::transmute(
      model = model,
      interval = interval,
      term = term,
      HR = estimate,
      conf_low = conf.low,
      conf_high = conf.high,
      p_value = p.value,
      HR_CI = sprintf("%.3f (%.3f-%.3f)", HR, conf_low, conf_high),
      p_label = vapply(p_value, fmt_p, character(1))
    )
}

ph_table <- function(fit, model) {
  z <- survival::cox.zph(fit, transform = "km", terms = TRUE, singledf = FALSE)
  tab <- as.data.frame(z$table, check.names = FALSE)
  data.frame(
    model = model,
    term = rownames(tab),
    chisq = tab[["chisq"]],
    df = tab[["df"]],
    p_value = tab[["p"]],
    p_label = vapply(tab[["p"]], fmt_p, character(1)),
    row.names = NULL,
    check.names = FALSE
  )
}

piecewise_cox <- function(data, time_col, event_col, predictors, model_label) {
  keep <- unique(c(time_col, event_col, predictors))
  d <- data[stats::complete.cases(data[, keep, drop = FALSE]), keep, drop = FALSE]
  d$analysis_id <- seq_len(nrow(d))
  names(d)[names(d) == time_col] <- "followup_time"
  names(d)[names(d) == event_col] <- "interval_event"

  split_data <- survival::survSplit(
    Surv(followup_time, interval_event) ~ .,
    data = d,
    cut = c(12, 60),
    start = "tstart",
    end = "followup_time",
    event = "interval_event",
    episode = "interval_id",
    id = "split_row_id"
  )
  split_data$interval_label <- factor(
    split_data$interval_id,
    levels = 1:3,
    labels = c("0-12 months", ">12-60 months", ">60 months")
  )

  estimates <- list()
  counts <- list()
  for (band in levels(split_data$interval_label)) {
    dd <- droplevels(split_data[split_data$interval_label == band, , drop = FALSE])
    counts[[band]] <- data.frame(
      model = model_label,
      interval = band,
      risk_intervals = nrow(dd),
      events = sum(dd$interval_event == 1L, na.rm = TRUE)
    )
    fit <- survival::coxph(
      stats::reformulate(
        predictors,
        response = "Surv(tstart, followup_time, interval_event)"
      ),
      data = dd,
      ties = "efron",
      model = FALSE,
      x = FALSE,
      y = FALSE
    )
    estimates[[band]] <- tidy_cox(fit, model_label, band)
  }
  list(
    estimates = dplyr::bind_rows(estimates),
    counts = dplyr::bind_rows(counts)
  )
}

format_median_iqr <- function(x) {
  q <- stats::quantile(x, c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE)
  sprintf("%.1f [%.1f, %.1f]", q[2], q[1], q[3])
}

format_n_pct <- function(x, level) {
  den <- sum(!is.na(x))
  n <- sum(as.character(x) == level, na.rm = TRUE)
  if (!den) return("0 (NA)")
  sprintf("%s (%.1f)", format(n, big.mark = ","), 100 * n / den)
}

continuous_smd <- function(x1, x0) {
  (mean(x1, na.rm = TRUE) - mean(x0, na.rm = TRUE)) /
    sqrt((stats::var(x1, na.rm = TRUE) + stats::var(x0, na.rm = TRUE)) / 2)
}

binary_smd <- function(x1, x0, level) {
  p1 <- mean(as.character(x1) == level, na.rm = TRUE)
  p0 <- mean(as.character(x0) == level, na.rm = TRUE)
  denom <- sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
  if (!is.finite(denom) || denom == 0) return(NA_real_)
  (p1 - p0) / denom
}

master <- data.table::fread(
  INPUT_MASTER,
  encoding = "UTF-8",
  na.strings = c("", "NA"),
  colClasses = list(character = "patient_id"),
  data.table = FALSE
)

required <- c(
  "patient_id", "index_year", "age_numeric", "baseline_malignancy_at_zero",
  "time_months", "event_type", "index_mpn_subtype", "index_Sex",
  "index_Race recode (W, B, AI, API)", "index_survival_months_numeric",
  "index_Vital status recode (study cutoff used)",
  "index_Rural-Urban Continuum Code",
  "index_Median household income inflation adj to 2024",
  "index_Chemotherapy recode (yes, no/unk) (2004+)"
)
missing <- setdiff(required, names(master))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
if (anyDuplicated(master$patient_id)) stop("Duplicate patient identifiers in master file.")

eligible_era <- master |>
  dplyr::filter(index_year >= 2004L, index_year <= 2023L) |>
  dplyr::mutate(
    inclusion_group = dplyr::case_when(
      baseline_malignancy_at_zero == 1L ~ "Excluded: synchronous malignant tumour at month 0",
      is.na(time_months) | time_months < 1 ~ "Excluded: unavailable, invalid, or <1-month time",
      TRUE ~ "Included"
    ),
    inclusion_group = factor(
      inclusion_group,
      levels = c(
        "Included",
        "Excluded: synchronous malignant tumour at month 0",
        "Excluded: unavailable, invalid, or <1-month time"
      )
    ),
    excluded_binary = factor(
      ifelse(inclusion_group == "Included", "Included", "Excluded"),
      levels = c("Included", "Excluded")
    ),
    sex = factor(index_Sex),
    race = factor(`index_Race recode (W, B, AI, API)`),
    mpn_subtype = factor(index_mpn_subtype, levels = c("PV", "ET", "PMF")),
    diagnosis_era = factor(
      dplyr::case_when(
        index_year <= 2008 ~ "2004-2008",
        index_year <= 2013 ~ "2009-2013",
        index_year <= 2018 ~ "2014-2018",
        TRUE ~ "2019-2023"
      ),
      levels = c("2004-2008", "2009-2013", "2014-2018", "2019-2023")
    ),
    urban_grp = factor(
      recode_urban(`index_Rural-Urban Continuum Code`),
      levels = c("Metropolitan", "Nonmetropolitan", "Unknown")
    ),
    income_lower = parse_income(`index_Median household income inflation adj to 2024`),
    income_grp = factor(
      dplyr::case_when(
        is.na(income_lower) ~ "Unknown",
        income_lower < 60000 ~ "<$60,000",
        income_lower < 75000 ~ "$60,000-$74,999",
        income_lower < 90000 ~ "$75,000-$89,999",
        TRUE ~ ">=$90,000"
      ),
      levels = c("<$60,000", "$60,000-$74,999", "$75,000-$89,999", ">=$90,000", "Unknown")
    ),
    chemo_seer = factor(
      recode_chemo(`index_Chemotherapy recode (yes, no/unk) (2004+)`),
      levels = c("No/Unknown", "No", "Unknown", "Yes")
    )
  )

group_counts <- eligible_era |>
  dplyr::count(inclusion_group, name = "n") |>
  dplyr::mutate(percent_of_2004_2023 = 100 * n / sum(n))
write_csv(group_counts, "excluded_vs_included_counts.csv")

expected_counts <- c(
  "Included" = 85680L,
  "Excluded: synchronous malignant tumour at month 0" = 431L,
  "Excluded: unavailable, invalid, or <1-month time" = 2134L
)
observed_counts <- stats::setNames(group_counts$n, as.character(group_counts$inclusion_group))
if (!identical(as.integer(observed_counts[names(expected_counts)]), as.integer(expected_counts))) {
  stop("Cohort classification did not reproduce the expected 85,680/431/2,134 counts.")
}

included <- eligible_era[eligible_era$excluded_binary == "Included", , drop = FALSE]
excluded <- eligible_era[eligible_era$excluded_binary == "Excluded", , drop = FALSE]
sync <- eligible_era[
  eligible_era$inclusion_group == "Excluded: synchronous malignant tumour at month 0", , drop = FALSE
]
time_bad <- eligible_era[
  eligible_era$inclusion_group == "Excluded: unavailable, invalid, or <1-month time", , drop = FALSE
]

comparison_rows <- list(data.frame(
  characteristic = "Age, years",
  category = "Median [IQR]",
  included = format_median_iqr(included$age_numeric),
  excluded_all = format_median_iqr(excluded$age_numeric),
  excluded_month0 = format_median_iqr(sync$age_numeric),
  excluded_time_requirement = format_median_iqr(time_bad$age_numeric),
  smd_included_vs_excluded = continuous_smd(included$age_numeric, excluded$age_numeric)
))

categorical_specs <- list(
  "Sex" = "sex",
  "Race" = "race",
  "MPN subtype" = "mpn_subtype",
  "Diagnosis era" = "diagnosis_era",
  "Rural-urban residence" = "urban_grp",
  "Area-level median household income" = "income_grp",
  "Registry-recorded initial chemotherapy" = "chemo_seer"
)

for (label in names(categorical_specs)) {
  variable <- categorical_specs[[label]]
  levels_to_report <- levels(droplevels(eligible_era[[variable]]))
  for (level in levels_to_report) {
    comparison_rows[[length(comparison_rows) + 1L]] <- data.frame(
      characteristic = label,
      category = level,
      included = format_n_pct(included[[variable]], level),
      excluded_all = format_n_pct(excluded[[variable]], level),
      excluded_month0 = format_n_pct(sync[[variable]], level),
      excluded_time_requirement = format_n_pct(time_bad[[variable]], level),
      smd_included_vs_excluded = binary_smd(included[[variable]], excluded[[variable]], level)
    )
  }
}
comparison_table <- dplyr::bind_rows(comparison_rows) |>
  dplyr::mutate(abs_smd = abs(smd_included_vs_excluded))
write_csv(comparison_table, "excluded_vs_included_baseline.csv")

# Quantify follow-up support for the latest prespecified prediction horizon.
ml <- data.table::fread(INPUT_ML, data.table = FALSE)
test <- ml[ml$data_split == "test", , drop = FALSE]
if (nrow(test) != 25705L) stop("Locked test cohort did not reproduce n=25,705.")
censoring_fit <- survival::survfit(
  survival::Surv(time_months, as.integer(event_type == 0L)) ~ 1,
  data = test
)
censoring_60 <- summary(censoring_fit, times = 60, extend = TRUE)
horizon_support <- data.frame(
  evaluation_sample = "Locked internal test set",
  horizon_months = 60,
  n_total = nrow(test),
  n_with_observed_time_at_least_horizon = sum(test$time_months >= 60, na.rm = TRUE),
  n_censored_before_horizon = sum(test$event_type == 0L & test$time_months < 60, na.rm = TRUE),
  km_estimated_probability_not_censored_by_horizon = unname(censoring_60$surv[1]),
  method_note = "Reverse event indicator: censoring treated as the event for Kaplan-Meier estimation of G(t)."
)
write_csv(horizon_support, "locked_test_60_month_followup_support.csv")

# This rerun is limited to the exclusion comparison. Corrected baseline and
# time-varying models are fitted by rerun_income_corrected_traditional.R.
stopifnot(sum(included$income_grp == "Unknown") == 83L,
          sum(included$income_grp == "<$60,000") == 11569L)
write_csv(data.frame(status = "PASS", seed = SEED, included = nrow(included),
                     excluded = nrow(excluded), income_unknown = 83L,
                     input_master_md5 = unname(tools::md5sum(INPUT_MASTER))),
          "income_exclusion_run_manifest.csv")
writeLines(capture.output(sessionInfo()), file.path(OUTPUT_DIR, "logs", "sessionInfo.txt"))
message("Income-corrected exclusion comparison completed successfully.")

