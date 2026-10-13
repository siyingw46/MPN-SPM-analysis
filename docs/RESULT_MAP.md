# Publication-result navigation

This is a **source-output map**, not a claim that the exported CSV/PDF is identical to the manually formatted V5 submission file. The `report` stage writes `final_results/RESULT_TO_SOURCE_MAP.csv`, which gives the exact input and output relative paths for every exported table section. Use that generated catalog as the authoritative section-level trace after a complete run. All paths below are relative to the **restricted run directory**, not this public repository.

| Manuscript item | Report-stage scientific source | Upstream analysis / important qualification |
|---|---|---|
| Table 1 | `final_results/tables/Table_1/Baseline_characteristics.csv` | Corrected R baseline table via `r_models` and `r_figures`. |
| Table 2 | `final_results/tables/Table_2/Time_varying_HR.csv` | Corrected time-varying Cox output via `r_models` and `r_figures`. |
| Table 3 | `final_results/tables/Table_3/AUC_IBS_5000.csv`; `Concordance_point_estimates.csv` in the same folder | `locked`, final DeepHit prediction replacement, then 5,000 bootstrap; concordance file is a separate point-estimate source. |
| Table 4 | `final_results/tables/Table_4/Calibration.csv` | Locked-test calibration from `locked`; it is not a bootstrap confidence-interval table. |
| Tables S1–S5 | Corresponding `final_results/tables/Table_S1/` through `Table_S5/` | Cohort/split, CIF/Gray, KM, adjusted Cox/PH, and landmark source sections. Consult generated `RESULT_TO_SOURCE_MAP.csv` for each section. |
| Table S6 | `final_results/tables/Table_S6/` | Development base models, ensemble retention/weights, and expanded DeepHit settings; multiple source sections, not one CSV. |
| Table S7 | `final_results/tables/Table_S7/` | Locked-test performance, paired differences, and risk strata; the 5,000-bootstrap aggregates use the fixed locked prediction bundle. |
| Table S8 | `final_results/tables/Table_S8/` | Standard sensitivity/subgroup results plus R event-time analyses and **refitted** alternative-event models. These sections do not all use the same risk set or the Table S7 prediction bundle. |
| Figures 1–2 | `final_results/figures/` and `deliverables/v0.4/figures/` | Cohort-flow and competing-risk source plots/data. Figure 2 may include an author-composed layout beyond the exported source. |
| Figure 3 | `final_results/figures/Figure_3.pdf` | Corrected R forest-plot source copied by `report`; inspect final panel layout separately. |
| Figure 4 | `final_results/figures/Figure_4_locked_test_performance.pdf` | Locked-test comparison plus monthly AUC/Brier curves; the bootstrap supplies uncertainty limits. |
| Figures 5–6 | `final_results/figures/` and `deliverables/v0.4/figures/main/` | Calibration and DCA/risk-stratification plots. Confirm final manuscript numbering and composed layout against the submission files. |
| Figures S1–S6 | `final_results/figures/`, with some KM component PDFs prefixed `S2_` and S4 landmark component PDFs | Supplementary scientific plots/components. Figure S5 is model sensitivity/subgroup performance; Figure S6 is interpretation. Some final multipanel arrangements are assembled outside the script. |

## Which outputs are safe to share?

`final_results/` is an output collection, **not** automatically an upload-ready public directory. Review every generated file and every small aggregate cell against the SEER agreement and institutional rules. Never publish the run directory, raw export, patient master, patient-level prediction bundles (`*.npz`), patient-level SHAP samples, fitted models, or logs containing restricted paths/identifiers.

The public `reference_metrics/` directory contains selected **aggregate** local-rerun comparisons only. `compare_results.py` checks four bootstrap tables against those references; it does not audit every statement in the paper. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) and [reference metrics README](../reference_metrics/README.md) for the exact validation boundary.
