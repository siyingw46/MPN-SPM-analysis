# Curated package: never execute against the original project directory.
mpn_run_guard <- Sys.getenv("MPN_RUN_ROOT")
if (!nzchar(mpn_run_guard) || !file.exists(file.path(mpn_run_guard, ".mpn_reproduction_run")))
  stop("Use run_pipeline.py with a new marked run directory.")
if (normalizePath(Sys.getenv("MPN_PROJECT_ROOT"), mustWork=TRUE) != normalizePath(mpn_run_guard, mustWork=TRUE))
  stop("Project and run roots must match.")
mpn_args_guard <- commandArgs(trailingOnly=TRUE)
if (length(mpn_args_guard) && normalizePath(mpn_args_guard[1],mustWork=TRUE) != normalizePath(mpn_run_guard,mustWork=TRUE))
  stop("Argument root differs from protected run root.")
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(data.table); library(dplyr); library(survival); library(ggplot2)})
root <- normalizePath(Sys.getenv("MPN_PROJECT_ROOT", unset = getwd()), mustWork = TRUE)
base <- file.path(root, "_codex_workspace", "25_income_reanalysis_20261001")
out <- file.path(base, "traditional_outputs")
pub <- file.path(base, "replacement_files")
dir.create(pub, showWarnings = FALSE)
read_result <- function(name) fread(file.path(out, "tables", name), data.table = FALSE)
save_csv <- function(x, name) fwrite(as.data.table(x), file.path(pub, name), na = "")

manifest <- fread(file.path(out, "logs", "run_manifest.csv"), data.table = FALSE)
stopifnot(manifest$status == "PASS", manifest$analysis_rows == 85680L)
income <- fread(file.path(out, "qc", "income_groups_by_first_event.csv"), data.table = FALSE)
overall <- income |> group_by(income_grp) |> summarise(n = sum(n), .groups = "drop") |>
  mutate(event_label = "Overall", percent_within_disposition = 100 * n / sum(n))
save_csv(bind_rows(overall, income), "Table1_income_counts_and_percentages.csv")
save_csv(read_result("Table1_corrected_by_first_event.csv"), "Table1_complete_corrected.csv")
save_csv(fread(file.path(base, "exclusion_outputs", "tables", "excluded_vs_included_baseline.csv"), data.table = FALSE),
         "Table_S1B_Included_and_excluded_corrected.csv")
stopifnot(sum(income$n) == 85680L,
          identical(as.integer(income$n[income$income_grp == "Unknown"]), c(66L, 3L, 11L, 3L)))

# Table S3 keeps the established eight analyses; only the income analysis differs.
for (spec in list(c("KM_OS_corrected_medians.csv", "Table_S3_Medians_corrected.csv"),
                  c("KM_OS_corrected_overall_logrank.csv", "Table_S3_Overall_logrank_corrected.csv"),
                  c("KM_OS_corrected_pairwise_Holm.csv", "Table_S3_Pairwise_Holm_corrected.csv"))) {
  new <- read_result(spec[1])
  save_csv(new, spec[2])
}

index <- fread(file.path(root, "MPN", "SEER_MPN_raw_index_records_v0.4_85680.csv"), data.table = FALSE)
unknown <- index[["Median household income inflation adj to 2024"]] == "Unknown/missing/no match/Not 1990-2024"
u_time <- as.numeric(index[["Survival months"]][unknown])
u_death <- index[["Vital status recode (study cutoff used)"]][unknown] == "Dead"
u_os_counts <- data.frame(interval = c("0-12 months", ">12-60 months", ">60 months"),
                         income_unknown_patients_entering_interval = c(sum(u_time > 0), sum(u_time > 12), sum(u_time > 60)),
                         income_unknown_deaths = c(sum(u_death & u_time <= 12),
                           sum(u_death & u_time > 12 & u_time <= 60), sum(u_death & u_time > 60)))
stopifnot(u_os_counts$income_unknown_deaths[1] == 0L)
save_csv(u_os_counts, "Unknown_income_OS_event_support.csv")

comparisons <- list()
specs <- list(
  c("Cause_specific_Cox_SPM_corrected_interval_specific_HR.csv",
    "unused", "Baseline SPM"),
  c("Cox_OS_corrected_interval_specific_HR.csv",
    "unused", "Baseline OS"),
  c("Cox_OS_time_dependent_SPM_interval_specific_HR.csv", "unused", "Time-varying SPM"),
  c("Cox_OS_time_dependent_tumor_count_primary_interval_specific_HR.csv", "unused", "Time-varying tumour count")
)
for (spec in specs) {
  new <- read_result(spec[1])
  # Zero observed deaths in Unknown income during 0-12 months lead to a
  # non-finite maximum-likelihood coefficient. Retain the fitted numerical
  # output in traditional_outputs but do not publish its unstable Wald estimate.
  new$reporting_note <- ""
  bad <- new$term == "income_grpUnknown" & new$interval == "0-12 months" & spec[3] != "Baseline SPM"
  new$HR_CI[bad] <- "NE"
  new$p_label[bad] <- "NE"
  new[, c("HR", "conf_low", "conf_high", "p_value")][bad, ] <- NA_real_
  new$reporting_note[bad] <- "Unknown income: no deaths in this interval; the HR and Wald CI are not reliably estimable."
  save_csv(new, paste0("Corrected_", spec[1]))
}
source_table2 <- read_result("Forest_time_dependent_SPM_and_tumor_count_source.csv")
save_csv(source_table2, "Table2_corrected_time_varying_estimates.csv")

label_fun <- function(term) {
  labs <- c("age_grp50-64" = "Age 50-64 (vs <50)", "age_grp65-74" = "Age 65-74 (vs <50)",
    "age_grp75-84" = "Age 75-84 (vs <50)", "age_grp85+" = "Age 85+ (vs <50)", "sexMale" = "Male (vs female)",
    "raceAmerican Indian/Alaska Native" = "AI/AN (vs White)",
    "raceAsian or Pacific Islander" = "Asian/Pacific Islander (vs White)", "raceBlack" = "Black (vs White)",
    "raceUnknown" = "Unknown race (vs White)", "mpn_subtypeET" = "ET (vs PV)", "mpn_subtypePMF" = "PMF (vs PV)",
    "year_grp2009-2013" = "2009-2013 (vs 2004-2008)", "year_grp2014-2018" = "2014-2018 (vs 2004-2008)",
    "year_grp2019-2023" = "2019-2023 (vs 2004-2008)", "urban_grpNonmetropolitan" = "Nonmetropolitan (vs metropolitan)",
    "urban_grpUnknown" = "Unknown residence (vs metropolitan)",
    "income_grp$60,000-$74,999" = "Income $60,000-$74,999 (vs <$60,000)",
    "income_grp$75,000-$89,999" = "Income $75,000-$89,999 (vs <$60,000)",
    "income_grp>=$90,000" = "Income >=$90,000 (vs <$60,000)", "income_grpUnknown" = "Unknown income (vs <$60,000)",
    "chemo_seerYes" = "Initial chemotherapy (vs No/Unknown)")
  unname(ifelse(term %in% names(labs), labs[term], term))
}
forest <- function(filename, title, colour, os = FALSE) {
  d <- read_result(filename)
  d$interval <- factor(d$interval, levels = c("0-12 months", ">12-60 months", ">60 months"))
  d$label <- factor(label_fun(d$term), levels = rev(unique(label_fun(d$term))))
  omit <- os & d$term == "income_grpUnknown" & d$interval == "0-12 months"
  p <- ggplot(d[!omit, ], aes(HR, label)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey45", linewidth = 0.35) +
    geom_segment(aes(x = conf_low, xend = conf_high, yend = label), colour = colour, linewidth = 0.5) +
    geom_point(colour = colour, size = 1.6) + facet_grid(. ~ interval) + scale_x_log10() +
    labs(title = title, x = "Adjusted hazard ratio (log scale)", y = NULL,
         caption = if (os) "Unknown income at 0-12 months: NE (no deaths). Points and bars indicate HRs and 95% CIs." else "Points and bars indicate cause-specific HRs and 95% CIs.") +
    theme_bw(base_size = 9) + theme(axis.text.y = element_text(size = 7.4),
      plot.title = element_text(size = 12), plot.caption = element_text(size = 8), panel.grid.minor = element_blank())
  if (os) p <- p + geom_text(data = d[omit, ], aes(x = 1, label = "NE"), size = 2.8)
  p
}
p_spm <- forest("Cause_specific_Cox_SPM_corrected_interval_specific_HR.csv", "Baseline factors and non-haematological SPM", "#A04732")
p_os <- forest("Cox_OS_corrected_interval_specific_HR.csv", "Baseline factors and all-cause mortality", "#1B6CA8", os = TRUE)
combo <- cowplot::plot_grid(p_spm, p_os, labels = c("A", "B"), ncol = 1)
for (ext in c("pdf", "png")) {
  ggsave(file.path(pub, paste0("Figure_S3A_corrected.", ext)), p_spm, width = 11.5, height = 7.5, dpi = 300, bg = "white")
  ggsave(file.path(pub, paste0("Figure_S3B_corrected.", ext)), p_os, width = 11.5, height = 7.5, dpi = 300, bg = "white")
  ggsave(file.path(pub, paste0("Figure_S3_combined_corrected.", ext)), combo, width = 11.5, height = 15, dpi = 300, bg = "white")
  src <- file.path(out, "figures", paste0("Forest_time_dependent_SPM_and_tumor_count_intervals.", ext))
  dst <- file.path(pub, paste0("Figure3_corrected.", ext))
  # Both paths are this run's generated outputs. Refresh the publication copy
  # so PDF creation metadata matches the final executed run as well as data.
  stopifnot(file.copy(src, dst, overwrite = TRUE))
  stopifnot(unname(tools::md5sum(src)) == unname(tools::md5sum(dst)))
}
writeLines(capture.output(sessionInfo()), file.path(base, "validation_sessionInfo.txt"))
cat("Validation PASS: cohort, income counts, unchanged inputs, unchanged non-income KM, and corrected replacement figures checked.\n")
