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

# Corrected traditional descriptive, KM, and regression analyses for the
# SEER-MPN v0.4 cohort.
#
# This is a standalone script. It does not source or overwrite the existing
# project R scripts or outputs. It deliberately:
#   1. fixes Metropolitan/Nonmetropolitan classification;
#   2. excludes total tumour count from baseline-risk and KM analyses because
#      it contains follow-up information;
#   3. does not construct KM curves using ever-observed SPM as a baseline group;
#      the SPM-OS association is instead evaluated with a time-dependent Cox model;
#   4. reports proportional-hazards diagnostics and prespecified interval-specific
#      Cox estimates (0-12, >12-60, and >60 months);
#   5. preserves the SEER chemotherapy field as Yes versus No/Unknown. A genuine
#      three-level Yes/No/Unknown analysis is run only if the supplied input has
#      separately identifiable No and Unknown values;
#   6. replaces the invalid chemotherapy chi-square analysis with an OS log-rank
#      test and a competing-risk Gray test for non-haematologic SPM;
#   7. writes aggregate outputs, checksums, a run manifest, and sessionInfo().
#   8. treats accumulated tumour count as a post-index time-varying exposure,
#      with landmark KM and time-dependent Cox analyses rather than a biased
#      KM grouping based on the final end-of-follow-up tumour count;
#   9. generates a horizontal bar chart of first non-haematologic SPM types.
#
# No packages are installed by this script. No patient-level output is written.

options(stringsAsFactors = FALSE, warn = 1)

SEED <- 20260723L
set.seed(SEED)

command_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", command_full, value = TRUE)
SCRIPT_FILE <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)
} else {
  NA_character_
}

PROJECT_ROOT <- normalizePath(
  Sys.getenv("MPN_PROJECT_ROOT", unset = getwd()),
  mustWork = FALSE
)

first_existing <- function(paths) {
  paths <- unique(paths[nzchar(paths)])
  hit <- paths[file.exists(paths)]
  if (length(hit)) normalizePath(hit[1], mustWork = TRUE) else NA_character_
}

INDEX_CSV <- Sys.getenv("SEER_MPN_INDEX_CSV", unset = "")
if (!nzchar(INDEX_CSV)) {
  INDEX_CSV <- first_existing(c(
    file.path(PROJECT_ROOT, "MPN", "SEER_MPN_raw_index_records_v0.4_85680.csv"),
    file.path(PROJECT_ROOT, "SEER_MPN_raw_index_records_v0.4_85680.csv")
  ))
}

V04_MASTER_CSV <- Sys.getenv("SEER_MPN_V04_MASTER_CSV", unset = "")
if (!nzchar(V04_MASTER_CSV)) {
  V04_MASTER_CSV <- first_existing(c(
    file.path(PROJECT_ROOT, "MPN", "patient_master_v0.4_2000_2023.csv"),
    file.path(PROJECT_ROOT, "ML", "data", "patient_master_v0.4_2000_2023.csv"),
    file.path(PROJECT_ROOT, "patient_master_v0.4_2000_2023.csv")
  ))
}

RAW_TUMOR_TSV <- Sys.getenv("SEER_MPN_RAW_TUMOR_TSV", unset = "")
if (!nzchar(RAW_TUMOR_TSV)) {
  RAW_TUMOR_TSV <- first_existing(c(
    file.path(PROJECT_ROOT, "SEER_MPN_first_primary_person_selection_2000_2023.txt"),
    file.path(PROJECT_ROOT, "MPN", "SEER_MPN_first_primary_person_selection_2000_2023.txt")
  ))
}

OUTPUT_DIR <- Sys.getenv(
  "CORRECTED_R_OUTPUT_DIR",
  unset = file.path(
    PROJECT_ROOT,
    "_codex_workspace",
    "25_income_reanalysis_20261001",
    "traditional_outputs"
  )
)
OUTPUT_DIR <- normalizePath(OUTPUT_DIR, mustWork = FALSE)
allowed_output_root <- normalizePath(file.path(PROJECT_ROOT, "_codex_workspace", "25_income_reanalysis_20261001"), winslash = "/", mustWork = TRUE)
resolved_output <- normalizePath(OUTPUT_DIR, winslash = "/", mustWork = FALSE)
if (!startsWith(tolower(resolved_output), paste0(tolower(allowed_output_root), "/"))) {
  stop("Income rerun outputs must stay under the dedicated new analysis directory.")
}

for (subdir in c("tables", "figures", "qc", "logs")) {
  dir.create(file.path(OUTPUT_DIR, subdir), recursive = TRUE, showWarnings = FALSE)
}

LOG_FILE <- file.path(OUTPUT_DIR, "logs", "corrected_analysis.log")
log_connection <- file(LOG_FILE, open = "wt", encoding = "UTF-8")
sink(log_connection, type = "output", split = TRUE)
sink(log_connection, type = "message")

close_log <- function() {
  try(sink(type = "message"), silent = TRUE)
  try(sink(type = "output"), silent = TRUE)
  try(close(log_connection), silent = TRUE)
}

required_packages <- c(
  "data.table", "dplyr", "stringr", "survival", "broom",
  "ggplot2", "survminer", "cowplot", "tableone", "cmprsk", "scales"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

fmt_p <- function(p) {
  ifelse(
    is.na(p),
    NA_character_,
    ifelse(p < 1e-4, "<0.0001", format.pval(p, digits = 4, eps = 1e-4))
  )
}

write_csv <- function(x, filename) {
  data.table::fwrite(
    data.table::as.data.table(x),
    file.path(OUTPUT_DIR, "tables", filename),
    na = ""
  )
}

write_qc <- function(x, filename) {
  data.table::fwrite(
    data.table::as.data.table(x),
    file.path(OUTPUT_DIR, "qc", filename),
    na = ""
  )
}

normalise_text <- function(x) {
  stringr::str_squish(stringr::str_to_lower(as.character(x)))
}

parse_income <- function(x) {
  z <- trimws(as.character(x))
  income_map <- c(
    "< $40,000" = 40000, "$40,000 - $44,999" = 40000,
    "$45,000 - $49,999" = 45000, "$50,000 - $54,999" = 50000,
    "$55,000 - $59,999" = 55000, "$60,000 - $64,999" = 60000,
    "$65,000 - $69,999" = 65000, "$70,000 - $74,999" = 70000,
    "$75,000 - $79,999" = 75000, "$80,000 - $84,999" = 80000,
    "$85,000 - $89,999" = 85000, "$90,000 - $94,999" = 90000,
    "$95,000 - $99,999" = 95000, "$100,000 - $109,999" = 100000,
    "$110,000 - $119,999" = 110000, "$120,000+" = 120000
  )
  unknown <- is.na(z) | z == "" | z == "Unknown/missing/no match/Not 1990-2024"
  unexpected <- !unknown & !(z %in% names(income_map))
  if (any(unexpected)) stop("Unmapped SEER income labels: ", paste(unique(z[unexpected]), collapse = "; "))
  values <- unname(income_map[z])
  values[unknown] <- NA_real_
  values
}

recode_urban <- function(x) {
  z <- normalise_text(x)
  # Nonmetropolitan must be tested first because it contains "metropolitan".
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
    z %in% c("no", "0", "none") ~ "No",
    z %in% c("unknown", "unk", "missing", "not collected", "") | is.na(z) ~ "Unknown",
    stringr::str_detect(z, "no\\s*/?\\s*(unknown|unk)") ~ "No/Unknown",
    stringr::str_detect(z, "none\\s*/?\\s*(unknown|unk)") ~ "No/Unknown",
    TRUE ~ "Unknown"
  )
}

tumor_count_group_primary <- function(x) {
  factor(
    dplyr::case_when(
      x <= 1 ~ "1",
      x == 2 ~ "2",
      x >= 3 ~ "3+",
      TRUE ~ NA_character_
    ),
    levels = c("1", "2", "3+")
  )
}

tumor_count_group_exploratory <- function(x) {
  factor(
    dplyr::case_when(
      x <= 1 ~ "1",
      x == 2 ~ "2",
      x == 3 ~ "3",
      x >= 4 ~ "4+",
      TRUE ~ NA_character_
    ),
    levels = c("1", "2", "3", "4+")
  )
}

site_group <- function(x) {
  code <- stringr::str_extract(as.character(x), "^C[0-9]{2}")
  dplyr::case_when(
    code == "C61" ~ "Prostate",
    code %in% c("C33", "C34") ~ "Lung and bronchus",
    code == "C50" ~ "Breast",
    code %in% c("C18", "C19", "C20") ~ "Colorectal",
    code %in% c("C64", "C65") ~ "Kidney/renal pelvis",
    code == "C67" ~ "Urinary bladder",
    code == "C44" ~ "Skin",
    code %in% c("C54", "C55") ~ "Uterine corpus",
    code == "C73" ~ "Thyroid",
    code == "C25" ~ "Pancreas",
    code == "C22" ~ "Liver/intrahepatic bile duct",
    code == "C16" ~ "Stomach",
    code == "C15" ~ "Esophagus",
    code == "C56" ~ "Ovary",
    code == "C53" ~ "Cervix",
    code %in% c(
      "C00", "C01", "C02", "C03", "C04", "C05", "C06",
      "C09", "C10", "C11", "C12", "C13", "C14"
    ) ~ "Oral cavity/pharynx",
    code == "C80" ~ "Unknown primary",
    TRUE ~ "Other non-haematologic"
  )
}

sanitize_display_labels <- function(x) {
  x <- gsub("^<\\s*", "Under ", x)
  x <- gsub("^>=\\s*", "At least ", x)
  x <- gsub("^>\\s*", "Over ", x)
  x
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
    ph_flag_p_lt_0_05 = !is.na(tab[["p"]]) & tab[["p"]] < 0.05,
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

  results <- list()
  counts <- list()
  for (band in levels(split_data$interval_label)) {
    dd <- droplevels(split_data[split_data$interval_label == band, , drop = FALSE])
    event_n <- sum(dd$interval_event == 1L, na.rm = TRUE)
    counts[[band]] <- data.frame(
      model = model_label,
      interval = band,
      risk_intervals = nrow(dd),
      events = event_n
    )
    if (event_n < 10L) next
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
    results[[band]] <- tidy_cox(fit, model_label, band)
  }
  list(
    estimates = dplyr::bind_rows(results),
    counts = dplyr::bind_rows(counts)
  )
}

piecewise_time_dependent_cox <- function(
  data,
  predictors,
  exposure = "spm_td",
  model_label = "OS Cox with time-dependent SPM"
) {
  split_data <- survival::survSplit(
    Surv(tstart, tstop, death) ~ .,
    data = data,
    cut = c(12, 60),
    start = "tstart",
    end = "tstop",
    event = "death",
    episode = "interval_id"
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
    death_n <- sum(dd$death == 1L, na.rm = TRUE)
    exposure_values <- as.character(dd[[exposure]])
    reference_value <- if (is.factor(dd[[exposure]])) {
      levels(dd[[exposure]])[1]
    } else {
      sort(unique(exposure_values))[1]
    }
    nonreference <- exposure_values != reference_value
    counts[[band]] <- data.frame(
      model = model_label,
      interval = band,
      patients = dplyr::n_distinct(dd$analysis_id),
      risk_intervals = nrow(dd),
      deaths = death_n,
      nonreference_intervals = sum(nonreference, na.rm = TRUE),
      deaths_in_nonreference_state = sum(dd$death == 1L & nonreference, na.rm = TRUE)
    )
    if (death_n < 10L || length(unique(exposure_values)) < 2L) next
    fit <- survival::coxph(
      stats::reformulate(
        c(exposure, predictors),
        response = "Surv(tstart, tstop, death)"
      ),
      data = dd,
      ties = "efron",
      cluster = analysis_id,
      model = FALSE,
      x = FALSE,
      y = FALSE
    )
    interval_estimates <- tidy_cox(
      fit,
      model_label,
      band
    )
    interval_estimates$state_patients <- NA_integer_
    interval_estimates$state_deaths <- NA_integer_
    interval_estimates$estimability_note <- NA_character_
    if (is.factor(dd[[exposure]])) {
      exposure_levels <- levels(dd[[exposure]])
      for (exposure_level in exposure_levels[-1L]) {
        level_rows <- as.character(dd[[exposure]]) == exposure_level
        level_term <- paste0(exposure, exposure_level)
        estimate_rows <- interval_estimates$term == level_term
        level_patients <- dplyr::n_distinct(dd$analysis_id[level_rows])
        level_deaths <- sum(dd$death[level_rows] == 1L, na.rm = TRUE)
        interval_estimates$state_patients[estimate_rows] <- level_patients
        interval_estimates$state_deaths[estimate_rows] <- level_deaths
        if (level_patients < 10L || level_deaths < 5L) {
          numeric_result_columns <- intersect(
            c("estimate", "std.error", "statistic", "p_value", "HR", "conf_low", "conf_high"),
            names(interval_estimates)
          )
          interval_estimates[estimate_rows, numeric_result_columns] <- NA_real_
          interval_estimates$p_label[estimate_rows] <- "NOT ESTIMABLE (sparse state)"
          interval_estimates$estimability_note[estimate_rows] <- paste0(
            "Suppressed: ", level_patients, " patients and ", level_deaths,
            " deaths in this time-dependent state within the interval."
          )
        }
      }
    }
    estimates[[band]] <- interval_estimates
  }
  list(
    estimates = dplyr::bind_rows(estimates),
    counts = dplyr::bind_rows(counts)
  )
}

save_km <- function(
  data,
  variable,
  label,
  file_stub,
  x_max = 120,
  time_col = "os_months",
  event_col = "death_any",
  x_label = "Months since index MPN diagnosis",
  title_prefix = "Overall survival by",
  caption_text = NULL
) {
  keep <- c(time_col, event_col, variable)
  d <- data[stats::complete.cases(data[, keep, drop = FALSE]), keep, drop = FALSE]
  d[[variable]] <- droplevels(factor(d[[variable]]))
  if (nlevels(d[[variable]]) < 2L) return(NULL)

  formula <- stats::reformulate(
    variable,
    response = paste0("Surv(", time_col, ", ", event_col, ")")
  )
  # surv_fit stores the evaluated formula in the fitted object; this avoids
  # ggsurvplot risk-table failures caused by a call that only stores the symbol
  # `formula` rather than the formula itself.
  fit <- survminer::surv_fit(formula, data = d)
  lr <- survival::survdiff(formula, data = d)
  p_overall <- stats::pchisq(lr$chisq, df = length(lr$n) - 1L, lower.tail = FALSE)

  global <- data.frame(
    variable = variable,
    label = label,
    n = nrow(d),
    groups = nlevels(d[[variable]]),
    events = sum(d[[event_col]] == 1L),
    chisq = unname(lr$chisq),
    df = length(lr$n) - 1L,
    p_value = p_overall,
    p_label = fmt_p(p_overall)
  )

  med <- summary(fit)$table
  if (is.null(dim(med))) {
    med <- matrix(med, nrow = 1L, dimnames = list(levels(d[[variable]]), names(med)))
  }
  median_table <- data.frame(
    variable = variable,
    group = sub("^[^=]+=", "", rownames(med)),
    n = med[, "records"],
    events = med[, "events"],
    median_months = med[, "median"],
    lower_95 = med[, "0.95LCL"],
    upper_95 = med[, "0.95UCL"],
    row.names = NULL
  )

  pairwise <- data.frame()
  if (nlevels(d[[variable]]) >= 3L) {
    pw <- survminer::pairwise_survdiff(
      formula,
      data = d,
      p.adjust.method = "holm"
    )$p.value
    pairwise <- as.data.frame(as.table(pw))
    names(pairwise) <- c("group_1", "group_2", "p_holm")
    pairwise <- pairwise[!is.na(pairwise$p_holm), , drop = FALSE]
    pairwise$variable <- variable
    pairwise$p_label <- vapply(pairwise$p_holm, fmt_p, character(1))
    pairwise <- pairwise[, c("variable", "group_1", "group_2", "p_holm", "p_label")]
  }

  g <- survminer::ggsurvplot(
    fit,
    data = d,
    fun = "pct",
    xlim = c(0, x_max),
    break.time.by = 24,
    pval = paste0("Overall log-rank P ", fmt_p(p_overall)),
    pval.size = 3.6,
    conf.int = FALSE,
    risk.table = TRUE,
    risk.table.height = 0.27,
    risk.table.title = "Number at risk",
    legend.title = label,
    legend.labs = sanitize_display_labels(levels(d[[variable]])),
    title = paste(title_prefix, label),
    xlab = x_label,
    ylab = "Overall survival (%)",
    ggtheme = ggplot2::theme_bw(base_size = 11),
    tables.theme = survminer::theme_cleantable()
  )
  if (is.null(caption_text)) {
    caption_text <- paste0(
      "Display truncated at ", x_max,
      " months; tests use complete follow-up. Associations are non-causal."
    )
  }
  g$plot <- g$plot + ggplot2::labs(caption = caption_text)

  panels <- list(g$plot)
  heights <- c(0.62)
  pairwise_lines <- 0L

  if (nrow(pairwise) > 0L) {
    pair_text <- paste0(
      sanitize_display_labels(as.character(pairwise$group_1)),
      " vs ",
      sanitize_display_labels(as.character(pairwise$group_2)),
      ": ",
      pairwise$p_label
    )
    pair_note <- paste0(
      "Pairwise log-rank (Holm-adjusted): ",
      paste(pair_text, collapse = "; ")
    )
    wrapped <- strwrap(pair_note, width = 115)
    pairwise_lines <- length(wrapped)
    note_panel <- ggplot2::ggplot() +
      ggplot2::xlim(0, 1) +
      ggplot2::ylim(0, 1) +
      ggplot2::annotate(
        "text",
        x = 0.01,
        y = 0.5,
        label = paste(wrapped, collapse = "\n"),
        hjust = 0,
        vjust = 0.5,
        size = 2.55,
        lineheight = 1.12
      ) +
      ggplot2::theme_void()
    panels <- c(panels, list(note_panel))
    heights <- c(heights, 0.045 + 0.032 * pairwise_lines)
  }

  panels <- c(panels, list(g$table + ggplot2::theme(
    plot.title = ggplot2::element_text(size = 10)
  )))
  heights <- c(heights, 0.26)
  heights <- heights / sum(heights)

  combined <- cowplot::plot_grid(
    plotlist = panels,
    ncol = 1,
    rel_heights = heights,
    align = "v",
    axis = "lr"
  )
  total_height <- 7.5 + 0.25 * pairwise_lines

  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".png")),
    combined,
    width = 9,
    height = total_height,
    dpi = 300,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".pdf")),
    combined,
    width = 9,
    height = total_height,
    bg = "white"
  )

  list(global = global, pairwise = pairwise, median = median_table)
}

extract_cif_event1 <- function(ci_object) {
  curve_names <- setdiff(names(ci_object), "Tests")
  curve_names <- curve_names[vapply(ci_object[curve_names], function(x) {
    is.list(x) && all(c("time", "est", "var") %in% names(x))
  }, logical(1))]
  curve_names <- curve_names[grepl(" 1$", curve_names)]
  dplyr::bind_rows(lapply(curve_names, function(nm) {
    curve <- ci_object[[nm]]
    data.frame(
      group = sub(" 1$", "", nm),
      time_months = curve$time,
      cif = curve$est,
      se = sqrt(curve$var)
    )
  }))
}

save_chemo_cif <- function(data, variable, label, file_stub) {
  keep <- c("time_months", "event_type", variable)
  d <- data[stats::complete.cases(data[, keep, drop = FALSE]), keep, drop = FALSE]
  d[[variable]] <- droplevels(factor(d[[variable]]))
  if (nlevels(d[[variable]]) < 2L) return(NULL)

  ci <- cmprsk::cuminc(
    ftime = d$time_months,
    fstatus = d$event_type,
    group = d[[variable]],
    cencode = 0
  )
  gray_p <- if (!is.null(ci$Tests) && nrow(ci$Tests) >= 1L) ci$Tests[1, "pv"] else NA_real_
  curves <- extract_cif_event1(ci)

  p <- ggplot2::ggplot(curves, ggplot2::aes(x = time_months, y = 100 * cif, colour = group)) +
    ggplot2::geom_step(linewidth = 0.8) +
    ggplot2::coord_cartesian(xlim = c(0, 120)) +
    ggplot2::scale_x_continuous(breaks = seq(0, 120, by = 24)) +
    ggplot2::labs(
      title = paste("Non-haematologic SPM cumulative incidence by", label),
      subtitle = paste0("Gray test P ", fmt_p(gray_p)),
      x = "Months since index MPN diagnosis",
      y = "Cumulative incidence (%)",
      colour = label,
      caption = paste0(
        "Death and subsequent haematologic malignancy/registered MPN evolution ",
        "are competing events.\nAssociation is non-causal."
      )
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      plot.caption = ggplot2::element_text(size = 8, hjust = 0, lineheight = 1.1),
      plot.margin = ggplot2::margin(10, 15, 20, 10)
    )

  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".png")),
    p,
    width = 9.2,
    height = 6.6,
    dpi = 300,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".pdf")),
    p,
    width = 9.2,
    height = 6.6,
    bg = "white"
  )

  list(
    gray = data.frame(
      variable = variable,
      label = label,
      n = nrow(d),
      groups = nlevels(d[[variable]]),
      event_spm = sum(d$event_type == 1L),
      gray_p = gray_p,
      p_label = fmt_p(gray_p)
    ),
    curves = curves
  )
}

make_table1 <- function(data, strata, output_name) {
  variables <- c(
    "age_numeric", "age_grp", "sex", "race", "mpn_subtype",
    "year_grp", "urban_grp", "income_grp", "chemo_seer"
  )
  variables <- setdiff(variables, strata)
  keep <- unique(c(strata, variables))
  d <- droplevels(data[, keep, drop = FALSE])
  factor_vars <- variables[vapply(d[, variables, drop = FALSE], is.factor, logical(1))]
  tab <- tableone::CreateTableOne(
    vars = variables,
    strata = strata,
    data = d,
    factorVars = factor_vars,
    includeNA = TRUE,
    addOverall = TRUE,
    test = TRUE
  )
  mat <- print(
    tab,
    printToggle = FALSE,
    quote = FALSE,
    noSpaces = TRUE,
    showAllLevels = TRUE,
    missing = TRUE,
    test = TRUE,
    smd = TRUE,
    pDigits = 4,
    contDigits = 1,
    catDigits = 1,
    nonnormal = "age_numeric"
  )
  write_csv(
    data.frame(
      Characteristic = rownames(mat),
      mat,
      row.names = NULL,
      check.names = FALSE
    ),
    output_name
  )
}

cox_term_label <- function(term) {
  labels <- c(
    "age_grp50-64" = "Age 50-64 (vs <50)",
    "age_grp65-74" = "Age 65-74 (vs <50)",
    "age_grp75-84" = "Age 75-84 (vs <50)",
    "age_grp85+" = "Age 85+ (vs <50)",
    "sexMale" = "Male (vs female)",
    "raceAmerican Indian/Alaska Native" = "American Indian/Alaska Native (vs White)",
    "raceAsian or Pacific Islander" = "Asian/Pacific Islander (vs White)",
    "raceBlack" = "Black (vs White)",
    "raceUnknown" = "Unknown race (vs White)",
    "mpn_subtypeET" = "ET (vs PV)",
    "mpn_subtypePMF" = "PMF (vs PV)",
    "year_grp2009-2013" = "Diagnosed 2009-2013 (vs 2004-2008)",
    "year_grp2014-2018" = "Diagnosed 2014-2018 (vs 2004-2008)",
    "year_grp2019-2023" = "Diagnosed 2019-2023 (vs 2004-2008)",
    "urban_grpNonmetropolitan" = "Nonmetropolitan (vs metropolitan)",
    "urban_grpUnknown" = "Unknown residence (vs metropolitan)",
    "income_grp$60,000-$74,999" = "Income $60,000-$74,999 (vs <$60,000)",
    "income_grp$75,000-$89,999" = "Income $75,000-$89,999 (vs <$60,000)",
    "income_grp>=$90,000" = "Income >=$90,000 (vs <$60,000)",
    "income_grpUnknown" = "Unknown income (vs <$60,000)",
    "chemo_seerYes" = "Chemotherapy yes (vs no/unknown)"
  )
  unname(ifelse(term %in% names(labels), labels[term], term))
}

save_interval_forest <- function(
  estimates,
  title,
  subtitle,
  file_stub,
  colour,
  caption
) {
  d <- estimates |>
    dplyr::filter(
      is.finite(HR), is.finite(conf_low), is.finite(conf_high),
      HR > 0, conf_low > 0, conf_high > 0
    ) |>
    dplyr::mutate(
      interval = factor(
        interval,
        levels = c("0-12 months", ">12-60 months", ">60 months")
      ),
      term_label = cox_term_label(term)
    )
  term_order <- unique(cox_term_label(estimates$term))
  d$term_label <- factor(d$term_label, levels = rev(term_order))

  p <- ggplot2::ggplot(
    d,
    ggplot2::aes(x = HR, y = term_label)
  ) +
    ggplot2::geom_vline(
      xintercept = 1,
      linetype = "dashed",
      colour = "grey45",
      linewidth = 0.45
    ) +
    ggplot2::geom_segment(
      ggplot2::aes(x = conf_low, xend = conf_high, yend = term_label),
      colour = colour,
      linewidth = 0.65
    ) +
    ggplot2::geom_point(
      colour = colour,
      size = 1.8
    ) +
    ggplot2::facet_grid(. ~ interval) +
    ggplot2::scale_x_log10(
      breaks = scales::breaks_log(n = 5),
      labels = scales::label_number(accuracy = 0.01)
    ) +
    ggplot2::labs(
      title = title,
      subtitle = subtitle,
      x = "Hazard ratio (log scale)",
      y = NULL,
      caption = caption
    ) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(
      strip.background = ggplot2::element_rect(fill = "grey94"),
      strip.text = ggplot2::element_text(face = "bold"),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      plot.caption = ggplot2::element_text(size = 8, hjust = 0, lineheight = 1.1),
      plot.margin = ggplot2::margin(10, 15, 15, 10)
    )

  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".png")),
    p,
    width = 13,
    height = 9.2,
    dpi = 300,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", paste0(file_stub, ".pdf")),
    p,
    width = 13,
    height = 9.2,
    bg = "white"
  )
}

make_td_panel <- function(data, title, effect_levels, effect_colours) {
  interval_levels <- c("0-12 months", ">12-60 months", ">60 months")
  d <- data |>
    dplyr::mutate(
      interval = factor(interval, levels = interval_levels),
      effect = factor(effect, levels = effect_levels),
      interval_number = as.numeric(interval),
      effect_number = as.numeric(effect),
      y_position = interval_number +
        (effect_number - (length(effect_levels) + 1) / 2) * 0.18
    )

  p <- ggplot2::ggplot(d) +
    ggplot2::geom_vline(
      xintercept = 1,
      linetype = "dashed",
      colour = "grey45",
      linewidth = 0.5
    ) +
    ggplot2::geom_segment(
      data = dplyr::filter(d, is.finite(HR)),
      ggplot2::aes(
        x = conf_low, xend = conf_high,
        y = y_position, yend = y_position,
        colour = effect
      ),
      linewidth = 0.8
    ) +
    ggplot2::geom_point(
      data = dplyr::filter(d, is.finite(HR)),
      ggplot2::aes(x = HR, y = y_position, colour = effect, shape = effect),
      size = 2.6
    ) +
    ggplot2::geom_text(
      data = dplyr::filter(d, !is.finite(HR)),
      ggplot2::aes(x = 1.1, y = y_position, label = "NE", colour = effect),
      hjust = 0,
      size = 3.2,
      show.legend = FALSE
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq_along(interval_levels),
      labels = interval_levels,
      limits = c(0.55, 3.45)
    ) +
    ggplot2::scale_x_log10(
      limits = c(0.8, 13),
      breaks = c(1, 2, 3, 5, 10),
      labels = scales::label_number(accuracy = 0.1)
    ) +
    ggplot2::scale_colour_manual(values = effect_colours, drop = FALSE) +
    ggplot2::scale_shape_manual(
      values = seq(16, length.out = length(effect_levels)),
      drop = FALSE
    ) +
    ggplot2::labs(
      title = title,
      x = "Adjusted hazard ratio (log scale)",
      y = "Time since index MPN diagnosis",
      colour = NULL,
      shape = NULL
    ) +
    ggplot2::theme_bw(base_size = 10.5) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      legend.position = "top",
      legend.justification = "left",
      plot.title = ggplot2::element_text(face = "bold", size = 11),
      plot.margin = ggplot2::margin(8, 8, 8, 8)
    )
  p
}

save_time_dependent_forest_figures <- function(
  spm_estimates,
  tumor_primary_estimates
) {
  spm_source <- spm_estimates |>
    dplyr::filter(term == "spm_td") |>
    dplyr::mutate(effect = "SPM present vs not yet present")
  primary_source <- tumor_primary_estimates |>
    dplyr::filter(stringr::str_starts(term, "tumor_count_td_primary")) |>
    dplyr::mutate(
      effect = dplyr::recode(
        term,
        "tumor_count_td_primary2" = "2 vs 1 tumour",
        "tumor_count_td_primary3+" = "3+ vs 1 tumour"
      )
    )
  panel_a <- make_td_panel(
    spm_source,
    "Time-dependent SPM status",
    "SPM present vs not yet present",
    c("SPM present vs not yet present" = "#147D8C")
  )
  panel_b <- make_td_panel(
    primary_source,
    "Time-dependent accumulated tumour count",
    c("2 vs 1 tumour", "3+ vs 1 tumour"),
    c("2 vs 1 tumour" = "#D97706", "3+ vs 1 tumour" = "#7C3AED")
  )
  combined_core <- cowplot::plot_grid(
    panel_a,
    panel_b,
    labels = c("A", "B"),
    label_size = 13,
    ncol = 2,
    align = "hv"
  )
  caption_text <- paste(
    "Interval-specific adjusted hazard ratios with 95% confidence intervals.",
    "Models adjust for age, sex, race, MPN subtype, diagnosis era, residence,",
    "household income, and SEER chemotherapy recode. Associations are non-causal."
  )
  caption_panel <- ggplot2::ggplot() +
    ggplot2::xlim(0, 1) +
    ggplot2::ylim(0, 1) +
    ggplot2::annotate(
      "text",
      x = 0.01,
      y = 0.55,
      label = paste(strwrap(caption_text, width = 175), collapse = "\n"),
      hjust = 0,
      vjust = 0.5,
      size = 2.75,
      lineheight = 1.05
    ) +
    ggplot2::theme_void()
  combined <- cowplot::plot_grid(
    combined_core,
    caption_panel,
    ncol = 1,
    rel_heights = c(0.92, 0.08)
  )
  ggplot2::ggsave(
    file.path(
      OUTPUT_DIR, "figures",
      "Forest_time_dependent_SPM_and_tumor_count_intervals.png"
    ),
    combined,
    width = 12,
    height = 5.6,
    dpi = 300,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(
      OUTPUT_DIR, "figures",
      "Forest_time_dependent_SPM_and_tumor_count_intervals.pdf"
    ),
    combined,
    width = 12,
    height = 5.6,
    bg = "white"
  )

  dplyr::bind_rows(
    spm_source |> dplyr::mutate(panel = "A: time-dependent SPM"),
    primary_source |> dplyr::mutate(panel = "B: time-dependent tumour count")
  )
}

run_analysis <- function() {
  if (length(missing_packages)) {
    stop(
      "Missing required R packages: ", paste(missing_packages, collapse = ", "),
      ". Install them on the server before rerunning; this script does not install packages."
    )
  }
  if (is.na(INDEX_CSV) || !file.exists(INDEX_CSV)) {
    stop("Index CSV not found. Set SEER_MPN_INDEX_CSV to its full path.")
  }
  if (is.na(V04_MASTER_CSV) || !file.exists(V04_MASTER_CSV)) {
    stop("v0.4 patient-master CSV not found. Set SEER_MPN_V04_MASTER_CSV.")
  }
  if (is.na(RAW_TUMOR_TSV) || !file.exists(RAW_TUMOR_TSV)) {
    stop("Tumour-level SEER TSV not found. Set SEER_MPN_RAW_TUMOR_TSV.")
  }

  suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(stringr)
    library(survival)
    library(broom)
    library(ggplot2)
    library(survminer)
    library(cowplot)
    library(tableone)
    library(cmprsk)
  })

  message("Reading index data: ", INDEX_CSV)
  index_raw <- data.table::fread(
    INDEX_CSV,
    encoding = "UTF-8",
    na.strings = c("", "NA"),
    colClasses = list(character = c("Patient ID", "Record number recode")),
    data.table = FALSE
  )
  message("Reading v0.4 patient master: ", V04_MASTER_CSV)
  v04 <- data.table::fread(
    V04_MASTER_CSV,
    encoding = "UTF-8",
    na.strings = c("", "NA"),
    colClasses = list(character = "patient_id"),
    data.table = FALSE
  )

  required_index <- c(
    "Patient ID", "Record number recode",
    "Total number of in situ/malignant tumors for patient",
    "Year of diagnosis", "Age recode with single ages and 85+",
    "Sex", "Race recode (W, B, AI, API)", "ICD-O-3 Hist/behav",
    "Chemotherapy recode (yes, no/unk) (2004+)",
    "Vital status recode (study cutoff used)", "Survival months",
    "Rural-Urban Continuum Code",
    "Median household income inflation adj to 2024"
  )
  required_v04 <- c(
    "patient_id", "event_type", "event_spm", "event_death", "time_months",
    "first_nonaml_spm_time_raw", "first_nonaml_spm_site"
  )
  missing_index <- setdiff(required_index, names(index_raw))
  missing_v04 <- setdiff(required_v04, names(v04))
  if (length(missing_index)) stop("Missing index columns: ", paste(missing_index, collapse = ", "))
  if (length(missing_v04)) stop("Missing v0.4 columns: ", paste(missing_v04, collapse = ", "))

  if (anyDuplicated(index_raw[["Patient ID"]])) stop("Index CSV contains duplicate patient IDs.")
  if (anyDuplicated(v04$patient_id)) stop("v0.4 patient master contains duplicate patient IDs.")

  index <- index_raw |>
    dplyr::transmute(
      patient_id = as.character(.data[["Patient ID"]]),
      index_record_number = as.character(.data[["Record number recode"]]),
      tumor_count_final = as.integer(.data[["Total number of in situ/malignant tumors for patient"]]),
      index_year = as.integer(.data[["Year of diagnosis"]]),
      age_numeric = as.numeric(stringr::str_extract(
        .data[["Age recode with single ages and 85+"]], "[0-9]+"
      )),
      sex = factor(.data[["Sex"]]),
      race = stats::relevel(factor(.data[["Race recode (W, B, AI, API)"]]), ref = "White"),
      mpn_code = stringr::str_extract(.data[["ICD-O-3 Hist/behav"]], "^(9950|9961|9962)"),
      rural_urban_raw = as.character(.data[["Rural-Urban Continuum Code"]]),
      income_raw = as.character(.data[["Median household income inflation adj to 2024"]]),
      chemo_raw = as.character(.data[["Chemotherapy recode (yes, no/unk) (2004+)"]]),
      os_months = as.numeric(.data[["Survival months"]]),
      vital_status = as.character(.data[["Vital status recode (study cutoff used)"]])
    ) |>
    dplyr::mutate(
      mpn_subtype = factor(
        dplyr::recode(mpn_code, `9950` = "PV", `9961` = "PMF", `9962` = "ET"),
        levels = c("PV", "ET", "PMF")
      ),
      age_grp = factor(
        dplyr::case_when(
          age_numeric < 50 ~ "<50",
          age_numeric < 65 ~ "50-64",
          age_numeric < 75 ~ "65-74",
          age_numeric < 85 ~ "75-84",
          !is.na(age_numeric) ~ "85+",
          TRUE ~ NA_character_
        ),
        levels = c("<50", "50-64", "65-74", "75-84", "85+")
      ),
      year_grp = factor(
        dplyr::case_when(
          index_year <= 2008 ~ "2004-2008",
          index_year <= 2013 ~ "2009-2013",
          index_year <= 2018 ~ "2014-2018",
          TRUE ~ "2019-2023"
        ),
        levels = c("2004-2008", "2009-2013", "2014-2018", "2019-2023")
      ),
      urban_grp = factor(
        recode_urban(rural_urban_raw),
        levels = c("Metropolitan", "Nonmetropolitan", "Unknown")
      ),
      income_lower = parse_income(income_raw),
      income_grp = factor(
        dplyr::case_when(
          is.na(income_lower) ~ "Unknown",
          income_lower < 60000 ~ "<$60,000",
          income_lower < 75000 ~ "$60,000-$74,999",
          income_lower < 90000 ~ "$75,000-$89,999",
          TRUE ~ ">=$90,000"
        ),
        levels = c(
          "<$60,000", "$60,000-$74,999", "$75,000-$89,999",
          ">=$90,000", "Unknown"
        )
      ),
      chemo_recode = recode_chemo(chemo_raw),
      chemo_seer = factor(
        chemo_recode,
        levels = c("No/Unknown", "No", "Unknown", "Yes")
      ),
      death_any = as.integer(vital_status == "Dead")
    )

  event_data <- v04 |>
    dplyr::transmute(
      patient_id = as.character(patient_id),
      event_type = as.integer(event_type),
      event_spm = as.integer(event_spm),
      event_death = as.integer(event_death),
      time_months = as.numeric(time_months),
      first_nonaml_spm_time_raw = as.numeric(first_nonaml_spm_time_raw),
      first_nonaml_spm_site = as.character(first_nonaml_spm_site)
    )

  joined <- index |>
    dplyr::inner_join(event_data, by = "patient_id")

  event_time_qc <- data.frame(
    check = c(
      "index_rows",
      "v04_master_rows",
      "joined_rows",
      "joined_unique_patients",
      "missing_event_time_months",
      "nonpositive_event_time_months",
      "missing_os_months",
      "nonpositive_os_months",
      "minimum_event_time_months",
      "maximum_event_time_months"
    ),
    value = c(
      nrow(index),
      nrow(event_data),
      nrow(joined),
      dplyr::n_distinct(joined$patient_id),
      sum(is.na(joined$time_months)),
      sum(!is.na(joined$time_months) & joined$time_months <= 0),
      sum(is.na(joined$os_months)),
      sum(!is.na(joined$os_months) & joined$os_months <= 0),
      suppressWarnings(min(joined$time_months, na.rm = TRUE)),
      suppressWarnings(max(joined$time_months, na.rm = TRUE))
    )
  )
  write_qc(event_time_qc, "event_and_OS_time_internal_consistency.csv")

  main <- joined |>
    dplyr::filter(
      index_year >= 2004L,
      index_year <= 2023L,
      os_months > 0,
      time_months > 0,
      event_type %in% 0:3
    ) |>
    dplyr::mutate(
      event_label = factor(
        event_type,
        levels = 0:3,
        labels = c(
          "No first event observed", "Non-haematological SPM", "Death",
          "Haematological/MPN competing event"
        )
      )
    )

  if (!nrow(main)) stop("No eligible records after joining and filtering.")
  if (anyDuplicated(main$patient_id)) stop("Joined analysis cohort contains duplicate patient IDs.")
  stopifnot(nrow(main) == 85680L,
            sum(main$income_grp == "Unknown") == 83L,
            sum(main$income_grp == "<$60,000") == 11569L,
            sum(main$death_any) == 30526L)
  write_qc(main |> dplyr::count(event_label, income_grp, name = "n") |>
             dplyr::group_by(event_label) |>
             dplyr::mutate(percent_within_disposition = 100 * n / sum(n)) |>
             dplyr::ungroup(), "income_groups_by_first_event.csv")
  write_qc(main |> dplyr::count(income_raw, income_grp, name = "n"),
           "income_raw_to_corrected_group.csv")

  chemo_counts <- as.data.frame(table(main$chemo_raw, useNA = "ifany"))
  names(chemo_counts) <- c("raw_value", "n")
  chemo_recoded_counts <- as.data.frame(table(main$chemo_seer, useNA = "ifany"))
  names(chemo_recoded_counts) <- c("analysis_group", "n")
  write_qc(chemo_counts, "chemotherapy_raw_counts.csv")
  write_qc(chemo_recoded_counts, "chemotherapy_analysis_group_counts.csv")

  observed_chemo <- unique(as.character(stats::na.omit(main$chemo_seer)))
  has_combined_no_unknown <- "No/Unknown" %in% observed_chemo
  explicit_three_level <- !has_combined_no_unknown &&
    all(c("Yes", "No", "Unknown") %in% observed_chemo)

  chemo_status <- data.frame(
    check = c(
      "combined_no_unknown_present",
      "explicit_yes_no_unknown_available",
      "primary_analysis",
      "three_group_analysis"
    ),
    value = c(
      as.character(has_combined_no_unknown),
      as.character(explicit_three_level),
      if (has_combined_no_unknown) "Yes versus No/Unknown" else paste(observed_chemo, collapse = " versus "),
      if (explicit_three_level) "RUN" else "SKIPPED: No and Unknown are not separately identifiable"
    )
  )
  write_qc(chemo_status, "chemotherapy_grouping_decision.csv")

  urban_mapping <- as.data.frame(table(
    raw_value = main$rural_urban_raw,
    corrected_group = main$urban_grp,
    useNA = "ifany"
  ))
  urban_mapping <- urban_mapping[urban_mapping$Freq > 0, , drop = FALSE]
  write_qc(urban_mapping, "rural_urban_corrected_mapping_counts.csv")

  cohort_qc <- dplyr::bind_rows(
    data.frame(check = "analysis_rows", category = "all", n = nrow(main)),
    data.frame(check = "unique_patients", category = "all", n = dplyr::n_distinct(main$patient_id)),
    data.frame(
      check = "event_type",
      category = names(table(main$event_type)),
      n = as.integer(table(main$event_type))
    ),
    data.frame(
      check = "mpn_subtype",
      category = names(table(main$mpn_subtype)),
      n = as.integer(table(main$mpn_subtype))
    )
  )
  write_qc(cohort_qc, "cohort_counts.csv")

  exclusions <- data.frame(
    excluded_analysis = c(
      "KM of OS by ever-observed SPM",
      "Baseline/KM analysis using total tumour count",
      "Chemotherapy Yes/No/Unknown three-group KM when only No/Unknown is available",
      "Chemotherapy chi-square test for SPM"
    ),
    reason = c(
      "Immortal-time bias: future SPM status is not a baseline exposure; replaced by time-dependent Cox.",
      "The SEER total tumour count includes tumours recorded after index MPN and can leak future outcome information.",
      "SEER combines No and Unknown in the supplied recode; categories must not be invented.",
      "SPM is a time-to-event outcome with competing risks; replaced by Gray test and cumulative-incidence curves."
    )
  )
  write_qc(exclusions, "excluded_analyses_and_replacements.csv")

  # Corrected baseline tables: no total tumour count and no ever-SPM variable.
  make_table1(main, "mpn_subtype", "Table1_corrected_by_MPN_subtype.csv")
  make_table1(main, "event_label", "Table1_corrected_by_first_event.csv")

  # Baseline-only KM analyses. Neither tumour count nor ever-SPM is included.
  km_variables <- c(
    age_grp = "Age group",
    sex = "Sex",
    race = "Race",
    mpn_subtype = "MPN subtype",
    year_grp = "Diagnosis era",
    urban_grp = "Rural/urban residence",
    income_grp = "Area-level median household income",
    chemo_seer = "Chemotherapy (SEER recode)"
  )
  km_global <- list()
  km_pairwise <- list()
  km_medians <- list()
  for (v in names(km_variables)) {
    result <- save_km(
      main,
      variable = v,
      label = unname(km_variables[v]),
      file_stub = paste0("KM_OS_corrected_", v)
    )
    if (is.null(result)) next
    km_global[[v]] <- result$global
    km_pairwise[[v]] <- result$pairwise
    km_medians[[v]] <- result$median
  }
  write_csv(dplyr::bind_rows(km_global), "KM_OS_corrected_overall_logrank.csv")
  write_csv(dplyr::bind_rows(km_pairwise), "KM_OS_corrected_pairwise_Holm.csv")
  write_csv(dplyr::bind_rows(km_medians), "KM_OS_corrected_medians.csv")

  # Valid SEER chemotherapy analysis: OS KM/log-rank and competing-risk CIF/Gray.
  chemo_cif <- save_chemo_cif(
    main,
    variable = "chemo_seer",
    label = "Chemotherapy (SEER recode)",
    file_stub = "CIF_nonhematologic_SPM_by_chemo_SEER"
  )
  if (!is.null(chemo_cif)) {
    write_csv(chemo_cif$gray, "Gray_test_nonhematologic_SPM_by_chemo_SEER.csv")
    write_csv(chemo_cif$curves, "CIF_source_nonhematologic_SPM_by_chemo_SEER.csv")
  }

  # First non-haematologic SPM type among primary endpoint events. A horizontal
  # bar chart is used instead of a pie chart because the category labels and
  # relative frequencies remain readable when many tumour sites are present.
  spm_types <- main |>
    dplyr::filter(
      event_type == 1L,
      !is.na(first_nonaml_spm_site),
      nzchar(first_nonaml_spm_site)
    ) |>
    dplyr::mutate(spm_type = site_group(first_nonaml_spm_site)) |>
    dplyr::count(spm_type, name = "n", sort = TRUE) |>
    dplyr::mutate(
      percent = 100 * n / sum(n),
      label = sprintf("%s: %s (%.1f%%)", spm_type, format(n, big.mark = ","), percent),
      spm_type_plot = factor(spm_type, levels = rev(spm_type))
    )
  write_csv(
    spm_types |>
      dplyr::select(spm_type, n, percent, label),
    "SPM_type_distribution_primary_endpoint.csv"
  )
  p_spm_type <- ggplot2::ggplot(
    spm_types,
    ggplot2::aes(x = n, y = spm_type_plot)
  ) +
    ggplot2::geom_col(width = 0.72, fill = "#0F6B78") +
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%s (%.1f%%)", format(n, big.mark = ","), percent)),
      hjust = -0.05,
      size = 3.1
    ) +
    ggplot2::scale_x_continuous(
      labels = scales::comma,
      expand = ggplot2::expansion(mult = c(0, 0.22))
    ) +
    ggplot2::labs(
      title = "First non-haematologic malignant SPM type",
      subtitle = paste0("Primary endpoint events; n=", format(sum(spm_types$n), big.mark = ",")),
      x = "Number of patients",
      y = NULL,
      caption = "Tumour type is based on the first non-haematologic malignant SPM after index MPN."
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      plot.caption = ggplot2::element_text(hjust = 0, size = 8)
    )
  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", "Bar_first_nonhematologic_SPM_type.png"),
    p_spm_type,
    width = 9,
    height = 7.2,
    dpi = 300,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(OUTPUT_DIR, "figures", "Bar_first_nonhematologic_SPM_type.pdf"),
    p_spm_type,
    width = 9,
    height = 7.2,
    bg = "white"
  )

  # A three-level chemotherapy analysis is only legitimate if distinct No and
  # Unknown values exist and no combined No/Unknown records remain.
  if (explicit_three_level) {
    main$chemo_three_group <- factor(
      as.character(main$chemo_seer),
      levels = c("No", "Yes", "Unknown")
    )
    chemo_3_km <- save_km(
      main,
      variable = "chemo_three_group",
      label = "Chemotherapy",
      file_stub = "KM_OS_chemo_explicit_Yes_No_Unknown"
    )
    if (!is.null(chemo_3_km)) {
      write_csv(chemo_3_km$global, "KM_OS_chemo_three_group_logrank.csv")
      write_csv(chemo_3_km$pairwise, "KM_OS_chemo_three_group_pairwise_Holm.csv")
      write_csv(chemo_3_km$median, "KM_OS_chemo_three_group_medians.csv")
    }
    chemo_3_cif <- save_chemo_cif(
      main,
      variable = "chemo_three_group",
      label = "Chemotherapy",
      file_stub = "CIF_nonhematologic_SPM_chemo_explicit_Yes_No_Unknown"
    )
    if (!is.null(chemo_3_cif)) {
      write_csv(chemo_3_cif$gray, "Gray_test_SPM_chemo_three_group.csv")
      write_csv(chemo_3_cif$curves, "CIF_source_SPM_chemo_three_group.csv")
    }
  }

  predictors <- c(
    "age_grp", "sex", "race", "mpn_subtype", "year_grp",
    "urban_grp", "income_grp", "chemo_seer"
  )
  cox_data <- droplevels(main[stats::complete.cases(
    main[, c("os_months", "death_any", "time_months", "event_type", predictors), drop = FALSE]
  ), , drop = FALSE])
  stopifnot(nrow(cox_data) == nrow(main))
  write_qc(data.frame(analysis = "Baseline Cox", n = nrow(cox_data),
                     income_unknown = sum(cox_data$income_grp == "Unknown"),
                     os_deaths = sum(cox_data$death_any),
                     first_spm_events = sum(cox_data$event_type == 1L),
                     race_reference = levels(cox_data$race)[1]),
           "income_corrected_model_cohort.csv")

  # Standard Cox estimates are retained as average associations, but PH
  # violations are flagged and interval-specific estimates are also generated.
  os_fit <- survival::coxph(
    stats::reformulate(predictors, response = "Surv(os_months, death_any)"),
    data = cox_data,
    ties = "efron",
    x = TRUE,
    y = TRUE,
    model = FALSE
  )
  write_csv(tidy_cox(os_fit, "OS Cox"), "Cox_OS_corrected_average_HR.csv")
  write_csv(ph_table(os_fit, "OS Cox"), "Cox_OS_corrected_PH_diagnostics.csv")

  cs_data <- cox_data
  cs_data$spm_event <- as.integer(cs_data$event_type == 1L)
  cs_fit <- survival::coxph(
    stats::reformulate(predictors, response = "Surv(time_months, spm_event)"),
    data = cs_data,
    ties = "efron",
    x = TRUE,
    y = TRUE,
    model = FALSE
  )
  write_csv(
    tidy_cox(cs_fit, "Cause-specific Cox for non-haematologic SPM"),
    "Cause_specific_Cox_SPM_corrected_average_HR.csv"
  )
  write_csv(
    ph_table(cs_fit, "Cause-specific Cox for non-haematologic SPM"),
    "Cause_specific_Cox_SPM_corrected_PH_diagnostics.csv"
  )

  os_piecewise <- piecewise_cox(
    cox_data,
    "os_months",
    "death_any",
    predictors,
    "OS Cox"
  )
  write_csv(os_piecewise$estimates, "Cox_OS_corrected_interval_specific_HR.csv")
  write_csv(os_piecewise$counts, "Cox_OS_corrected_interval_counts.csv")

  cs_piecewise <- piecewise_cox(
    cs_data,
    "time_months",
    "spm_event",
    predictors,
    "Cause-specific Cox for non-haematologic SPM"
  )
  write_csv(
    cs_piecewise$estimates,
    "Cause_specific_Cox_SPM_corrected_interval_specific_HR.csv"
  )
  write_csv(
    cs_piecewise$counts,
    "Cause_specific_Cox_SPM_corrected_interval_counts.csv"
  )
  save_interval_forest(
    os_piecewise$estimates,
    title = "Interval-specific associations with all-cause mortality",
    subtitle = "Multivariable Cox models fitted separately within prespecified follow-up intervals",
    file_stub = "Forest_OS_multivariable_interval_specific",
    colour = "#1B6CA8",
    caption = paste0(
      "Points are adjusted hazard ratios; bars are 95% confidence intervals. ",
      "Separate interval estimates are emphasised because the global proportional-hazards test was significant. ",
      "Associations are non-causal."
    )
  )
  save_interval_forest(
    cs_piecewise$estimates,
    title = "Interval-specific associations with non-haematologic SPM",
    subtitle = "Cause-specific Cox models; competing events are censored at their occurrence",
    file_stub = "Forest_cause_specific_SPM_interval_specific",
    colour = "#A04732",
    caption = paste0(
      "Points are adjusted cause-specific hazard ratios; bars are 95% confidence intervals. ",
      "Separate interval estimates are emphasised because the global proportional-hazards test was significant. ",
      "Associations are non-causal."
    )
  )

  # Time-dependent SPM exposure for OS. This replaces the biased ever-SPM KM.
  td_base <- droplevels(main) |>
    dplyr::mutate(
      analysis_id = seq_len(dplyr::n()),
      spm_clock = dplyr::if_else(
        !is.na(first_nonaml_spm_time_raw) &
          first_nonaml_spm_time_raw > 0 &
          first_nonaml_spm_time_raw <= os_months,
        first_nonaml_spm_time_raw,
        Inf
      )
    ) |>
    dplyr::filter(os_months > 0)
  td <- survival::tmerge(
    td_base,
    td_base,
    id = analysis_id,
    tstop = os_months,
    death = event(os_months, death_any)
  )
  td <- survival::tmerge(
    td,
    td_base,
    id = analysis_id,
    spm_td = tdc(spm_clock)
  )
  td_fit <- survival::coxph(
    stats::reformulate(
      c("spm_td", predictors),
      response = "Surv(tstart, tstop, death)"
    ),
    data = td,
    ties = "efron",
    cluster = analysis_id,
    x = TRUE,
    y = TRUE,
    model = FALSE
  )
  write_csv(
    tidy_cox(td_fit, "OS Cox with time-dependent SPM"),
    "Cox_OS_time_dependent_SPM_corrected.csv"
  )
  write_csv(
    ph_table(td_fit, "OS Cox with time-dependent SPM"),
    "Cox_OS_time_dependent_SPM_PH_diagnostics.csv"
  )

  td_piecewise <- piecewise_time_dependent_cox(td, predictors)
  write_csv(
    td_piecewise$estimates,
    "Cox_OS_time_dependent_SPM_interval_specific_HR.csv"
  )
  write_csv(
    td_piecewise$counts,
    "Cox_OS_time_dependent_SPM_interval_counts.csv"
  )

  # -----------------------------------------------------------------------
  # Accumulated tumour burden during follow-up
  # -----------------------------------------------------------------------
  # The SEER final tumour-count field is retained for descriptive reporting,
  # but is never used as a baseline KM group or baseline model predictor.
  final_tumor_description <- main |>
    dplyr::mutate(
      final_tumor_count_group = tumor_count_group_exploratory(tumor_count_final)
    ) |>
    dplyr::group_by(final_tumor_count_group) |>
    dplyr::summarise(
      n = dplyr::n(),
      deaths = sum(death_any == 1L),
      observed_death_percent = 100 * deaths / n,
      nonhematologic_spm_events = sum(event_type == 1L),
      median_observed_os_months = stats::median(os_months),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      interpretation = paste(
        "End-of-follow-up descriptive variable only; not a baseline predictor",
        "and not a causal comparison."
      )
    )
  write_csv(
    final_tumor_description,
    "Tumor_count_final_descriptive_only.csv"
  )

  message("Reading tumour-level SEER data: ", RAW_TUMOR_TSV)
  raw_tumor <- data.table::fread(
    RAW_TUMOR_TSV,
    sep = "\t",
    encoding = "UTF-8",
    na.strings = c("", "NA", "Unknown"),
    select = c(
      "Patient ID", "Record number recode", "Survival months",
      "Behavior code ICD-O-3"
    ),
    colClasses = list(
      character = c(
        "Patient ID", "Record number recode", "Survival months",
        "Behavior code ICD-O-3"
      )
    ),
    data.table = FALSE
  )
  raw_tumor <- raw_tumor |>
    dplyr::transmute(
      patient_id = as.character(.data[["Patient ID"]]),
      record_number = as.character(.data[["Record number recode"]]),
      record_survival_months = suppressWarnings(as.numeric(.data[["Survival months"]])),
      behavior = as.character(.data[["Behavior code ICD-O-3"]])
    )

  tumor_lookup <- main |>
    dplyr::select(
      patient_id, index_record_number, os_months, death_any,
      tumor_count_final
    )
  tumor_records <- raw_tumor |>
    dplyr::inner_join(tumor_lookup, by = "patient_id") |>
    dplyr::mutate(
      is_index_record = record_number == index_record_number,
      countable_tumor = stringr::str_detect(
        normalise_text(behavior),
        "malignant|in situ"
      ),
      tumor_time_months = os_months - record_survival_months
    )

  tumor_records_distinct <- tumor_records |>
    dplyr::filter(!is_index_record, countable_tumor) |>
    dplyr::distinct(
      patient_id, record_number, tumor_time_months,
      os_months, death_any, tumor_count_final
    )
  valid_tumor_records_all <- tumor_records_distinct |>
    dplyr::filter(
      !is.na(tumor_time_months),
      tumor_time_months >= 1,
      tumor_time_months <= os_months
    )

  death_tie_ids <- valid_tumor_records_all |>
    dplyr::filter(death_any == 1L, tumor_time_months == os_months) |>
    dplyr::distinct(patient_id)

  reconstructed_counts <- valid_tumor_records_all |>
    dplyr::count(patient_id, name = "post_index_tumor_records")
  reconstruction <- tumor_lookup |>
    dplyr::left_join(reconstructed_counts, by = "patient_id") |>
    dplyr::mutate(
      post_index_tumor_records = dplyr::coalesce(post_index_tumor_records, 0L),
      reconstructed_final_count = 1L + post_index_tumor_records,
      exact_match = reconstructed_final_count == tumor_count_final
    )
  reconstruction_cross_tab <- as.data.frame(table(
    recorded_final = tumor_count_group_exploratory(reconstruction$tumor_count_final),
    reconstructed_final = tumor_count_group_exploratory(reconstruction$reconstructed_final_count),
    useNA = "ifany"
  ))
  reconstruction_cross_tab <- reconstruction_cross_tab[
    reconstruction_cross_tab$Freq > 0,
    ,
    drop = FALSE
  ]
  write_qc(
    reconstruction_cross_tab,
    "tumor_count_recorded_vs_reconstructed_groups.csv"
  )

  tumor_time_qc <- data.frame(
    check = c(
      "raw_tumor_rows",
      "raw_rows_linked_to_main_cohort",
      "nonindex_countable_tumor_records",
      "records_missing_derived_time",
      "records_with_derived_time_lt_1",
      "valid_post_index_tumor_records",
      "patients_with_valid_post_index_tumor",
      "patients_with_tumor_death_same_month_retained",
      "patients_exact_final_count_reconstruction",
      "patients_total_for_reconstruction",
      "exact_reconstruction_percent"
    ),
    value = c(
      nrow(raw_tumor),
      nrow(tumor_records),
      nrow(tumor_records_distinct),
      sum(is.na(tumor_records_distinct$tumor_time_months)),
      sum(!is.na(tumor_records_distinct$tumor_time_months) &
        tumor_records_distinct$tumor_time_months < 1),
      nrow(valid_tumor_records_all),
      dplyr::n_distinct(valid_tumor_records_all$patient_id),
      nrow(death_tie_ids),
      sum(reconstruction$exact_match, na.rm = TRUE),
      nrow(reconstruction),
      100 * mean(reconstruction$exact_match, na.rm = TRUE)
    )
  )
  write_qc(tumor_time_qc, "tumor_count_time_reconstruction_qc.csv")

  # Follow the locked Python endpoint convention: when a subsequent tumour and
  # death share the same inferred SEER month, retain the patient and give the
  # tumour event priority rather than excluding the entire patient.
  tumor_main <- droplevels(main)
  tumor_events <- valid_tumor_records_all |>
    dplyr::count(patient_id, tumor_time_months, name = "new_tumors")

  # Landmark KM avoids assigning future final tumour counts at MPN diagnosis.
  landmark_global <- list()
  landmark_pairwise <- list()
  landmark_medians <- list()
  landmark_counts <- list()
  for (landmark in c(12, 36, 60)) {
    accumulated <- tumor_events |>
      dplyr::filter(tumor_time_months <= landmark) |>
      dplyr::group_by(patient_id) |>
      dplyr::summarise(additional_tumors = sum(new_tumors), .groups = "drop")
    landmark_data <- tumor_main |>
      dplyr::filter(os_months > landmark) |>
      dplyr::left_join(accumulated, by = "patient_id") |>
      dplyr::mutate(
        additional_tumors = dplyr::coalesce(additional_tumors, 0L),
        tumor_count_at_landmark = 1L + additional_tumors,
        tumor_count_landmark_primary = tumor_count_group_primary(tumor_count_at_landmark),
        tumor_count_landmark_exploratory = tumor_count_group_exploratory(tumor_count_at_landmark),
        post_landmark_months = os_months - landmark,
        post_landmark_death = death_any
      )

    landmark_counts[[paste0(landmark, "_primary")]] <- landmark_data |>
      dplyr::count(tumor_count_landmark_primary, name = "n") |>
      dplyr::mutate(landmark_months = landmark, grouping = "1, 2, 3+")
    for (specification in "primary") {
      variable <- if (specification == "primary") {
        "tumor_count_landmark_primary"
      } else {
        "tumor_count_landmark_exploratory"
      }
      grouping_label <- if (specification == "primary") "1, 2, 3+" else "1, 2, 3, 4+"
      result <- save_km(
        landmark_data,
        variable = variable,
        label = paste0("Tumours accrued by ", landmark, " months"),
        file_stub = paste0(
          "KM_landmark_", landmark, "m_tumor_count_", specification
        ),
        x_max = 120,
        time_col = "post_landmark_months",
        event_col = "post_landmark_death",
        x_label = paste0("Months since the ", landmark, "-month landmark"),
        title_prefix = paste0(landmark, "-month landmark overall survival by"),
        caption_text = paste0(
          "Includes patients alive beyond ", landmark,
          " months; groups use tumours registered by the landmark. ",
          "Pairwise tests are Holm-adjusted. Associations are non-causal."
        )
      )
      if (is.null(result)) next
      landmark_global[[paste(landmark, specification)]] <- result$global |>
        dplyr::mutate(landmark_months = landmark, grouping = grouping_label)
      landmark_pairwise[[paste(landmark, specification)]] <- result$pairwise |>
        dplyr::mutate(landmark_months = landmark, grouping = grouping_label)
      landmark_medians[[paste(landmark, specification)]] <- result$median |>
        dplyr::mutate(landmark_months = landmark, grouping = grouping_label)
    }
  }
  write_csv(
    dplyr::bind_rows(landmark_counts),
    "Tumor_count_landmark_group_counts.csv"
  )
  write_csv(
    dplyr::bind_rows(landmark_global),
    "Tumor_count_landmark_KM_overall_logrank.csv"
  )
  write_csv(
    dplyr::bind_rows(landmark_pairwise),
    "Tumor_count_landmark_KM_pairwise_Holm.csv"
  )
  write_csv(
    dplyr::bind_rows(landmark_medians),
    "Tumor_count_landmark_KM_medians.csv"
  )

  # Time-dependent Cox: tumour count changes only after a subsequent tumour is
  # registered. This estimates association with subsequent mortality and does
  # not treat the final end-of-follow-up count as known at baseline.
  tumor_td_base <- tumor_main |>
    dplyr::mutate(analysis_id = seq_len(dplyr::n()))
  tumor_td_events <- tumor_events |>
    dplyr::inner_join(
      tumor_td_base |>
        dplyr::select(patient_id, analysis_id),
      by = "patient_id"
    ) |>
    dplyr::select(analysis_id, tumor_time_months, new_tumors)
  tumor_td <- survival::tmerge(
    tumor_td_base,
    tumor_td_base,
    id = analysis_id,
    tstop = os_months,
    death = event(os_months, death_any)
  )
  tumor_td <- survival::tmerge(
    tumor_td,
    tumor_td_events,
    id = analysis_id,
    additional_tumors_td = cumtdc(tumor_time_months, new_tumors)
  )
  tumor_td$additional_tumors_td[is.na(tumor_td$additional_tumors_td)] <- 0
  tumor_td$current_tumor_count <- 1L + tumor_td$additional_tumors_td
  tumor_td$tumor_count_td_primary <- tumor_count_group_primary(
    tumor_td$current_tumor_count
  )
  tumor_td$tumor_count_td_exploratory <- tumor_count_group_exploratory(
    tumor_td$current_tumor_count
  )

  td_state_summary <- tumor_td |>
    dplyr::mutate(interval_months = tstop - tstart) |>
    dplyr::group_by(tumor_count_td_exploratory) |>
    dplyr::summarise(
      risk_intervals = dplyr::n(),
      patients_ever_in_state = dplyr::n_distinct(analysis_id),
      person_months = sum(interval_months),
      deaths = sum(death == 1L),
      .groups = "drop"
    )
  write_csv(td_state_summary, "Tumor_count_time_dependent_state_summary.csv")

  tumor_interval_results <- list()
  for (specification in "primary") {
    exposure <- if (specification == "primary") {
      "tumor_count_td_primary"
    } else {
      "tumor_count_td_exploratory"
    }
    model_label <- if (specification == "primary") {
      "OS Cox with time-dependent tumour count (1, 2, 3+)"
    } else {
      "OS Cox with time-dependent tumour count (1, 2, 3, 4+; exploratory)"
    }
    fit <- survival::coxph(
      stats::reformulate(
        c(exposure, predictors),
        response = "Surv(tstart, tstop, death)"
      ),
      data = tumor_td,
      ties = "efron",
      cluster = analysis_id,
      x = TRUE,
      y = TRUE,
      model = FALSE
    )
    write_csv(
      tidy_cox(fit, model_label),
      paste0("Cox_OS_time_dependent_tumor_count_", specification, ".csv")
    )
    write_csv(
      ph_table(fit, model_label),
      paste0(
        "Cox_OS_time_dependent_tumor_count_", specification,
        "_PH_diagnostics.csv"
      )
    )
    interval_fit <- piecewise_time_dependent_cox(
      tumor_td,
      predictors,
      exposure = exposure,
      model_label = model_label
    )
    tumor_interval_results[[specification]] <- interval_fit$estimates
    write_csv(
      interval_fit$estimates,
      paste0(
        "Cox_OS_time_dependent_tumor_count_", specification,
        "_interval_specific_HR.csv"
      )
    )
    write_csv(
      interval_fit$counts,
      paste0(
        "Cox_OS_time_dependent_tumor_count_", specification,
        "_interval_counts.csv"
      )
    )
  }

  td_forest_source <- save_time_dependent_forest_figures(
    td_piecewise$estimates,
    tumor_interval_results$primary
  )
  write_csv(
    td_forest_source,
    "Forest_time_dependent_SPM_and_tumor_count_source.csv"
  )

  # Deliberately do not save patient-level cohorts or model RDS objects.
  # Aggregate tables plus code/input hashes and sessionInfo are the auditable outputs.
  invisible(list(n = nrow(main), chemo_three_level = explicit_three_level))
}

START_TIME <- Sys.time()
STATUS <- "FAILED"
ERROR_MESSAGE <- ""
RESULT <- NULL

message("Corrected analysis started: ", format(START_TIME, tz = "UTC", usetz = TRUE))
message("Project root: ", PROJECT_ROOT)
message("Output directory: ", OUTPUT_DIR)
message("Random seed: ", SEED)

tryCatch(
  {
    RESULT <- run_analysis()
    STATUS <- "PASS"
  },
  error = function(e) {
    ERROR_MESSAGE <<- conditionMessage(e)
    message("ERROR: ", ERROR_MESSAGE)
  }
)

END_TIME <- Sys.time()

input_paths <- c(
  index_csv = INDEX_CSV,
  v04_master_csv = V04_MASTER_CSV,
  raw_tumor_tsv = RAW_TUMOR_TSV
)
input_hashes <- do.call(rbind, lapply(seq_along(input_paths), function(i) {
  p <- unname(input_paths[i])
  present <- length(p) == 1L && !is.na(p) && nzchar(p) && file.exists(p)
  info <- if (present) file.info(p) else NULL
  data.frame(
    role = names(input_paths)[i],
    path = if (length(p) == 1L) p else NA_character_,
    exists = present,
    size_bytes = if (present) info$size else NA_real_,
    modified_time = if (present) {
      format(info$mtime, tz = "UTC", usetz = TRUE)
    } else {
      NA_character_
    },
    md5 = if (present) unname(tools::md5sum(p)) else NA_character_,
    stringsAsFactors = FALSE
  )
}))
utils::write.csv(
  input_hashes,
  file.path(OUTPUT_DIR, "qc", "input_file_checksums.csv"),
  row.names = FALSE,
  na = ""
)

script_md5 <- if (!is.na(SCRIPT_FILE) && file.exists(SCRIPT_FILE)) {
  unname(tools::md5sum(SCRIPT_FILE))
} else {
  NA_character_
}

manifest <- data.frame(
  status = STATUS,
  error_message = ERROR_MESSAGE,
  start_time_utc = format(START_TIME, tz = "UTC", usetz = TRUE),
  end_time_utc = format(END_TIME, tz = "UTC", usetz = TRUE),
  elapsed_minutes = as.numeric(difftime(END_TIME, START_TIME, units = "mins")),
  seed = SEED,
  script_path = SCRIPT_FILE,
  script_md5 = script_md5,
  index_csv = INDEX_CSV,
  v04_master_csv = V04_MASTER_CSV,
  raw_tumor_tsv = RAW_TUMOR_TSV,
  output_dir = OUTPUT_DIR,
  analysis_rows = if (!is.null(RESULT)) RESULT$n else NA_integer_,
  explicit_chemo_three_level_run = if (!is.null(RESULT)) RESULT$chemo_three_level else FALSE,
  stringsAsFactors = FALSE
)
utils::write.csv(
  manifest,
  file.path(OUTPUT_DIR, "logs", "run_manifest.csv"),
  row.names = FALSE,
  na = ""
)

writeLines(
  capture.output(sessionInfo()),
  file.path(OUTPUT_DIR, "logs", "sessionInfo.txt"),
  useBytes = TRUE
)

message("Corrected analysis finished with status: ", STATUS)
message("Finished: ", format(END_TIME, tz = "UTC", usetz = TRUE))
close_log()

if (!identical(STATUS, "PASS")) quit(save = "no", status = 1L)
