# Aggregate reference results for the local-rerun version

The four CSVs in this directory are the **new official local-rerun** 5,000-bootstrap aggregates, copied without editing from the controlled `run02/revision/04_bootstrap5000/model_root/final_test_evaluation/` output. They are the default reference files for `compare_results.py`. They are not the historical V4 values.

`extended/` contains only aggregate, non-patient-level checks:

| File | Controlled source | Repeatability boundary |
|---|---|---|
| `development_comparison_final.csv` | `run01/final_results/development_comparison_final.csv` | Main base OOF outputs matched run02; this final comparison replaces the DeepHit row with the expanded-development result. |
| `ensemble_retention.csv` | `run02/output/models/model_development_v0.4/ensemble_oof_evaluation/table_ensemble_retention_decision.csv` | Ridge decision matched; one non-decision summary differed at floating-point precision. |
| `test_calibration_summary.csv` | `run02/output/models/model_development_v0.4/final_test_evaluation/test_calibration_summary.csv` | Locked predictions matched byte-for-byte; some derived non-primary values can vary at machine precision. |
| `risk_strata_summary.csv`, `risk_strata_Gray.csv` | `run01/final_results/` | Derived during the full run; the development and locked prediction bundles matched run02. |
| `Table_S8_A_model_sensitivity.csv`, `Table_S8_B_subgroups.csv`, `Table_S8_E_refitted_models.csv` | `run01/final_results/tables/Table_S8/` | Completed in the first full run only; run02 did not repeat these analyses. |
| `deephit_retraining_summary.json` | `run02/revision/05_deephit_retraining/` | Summary and final predictions matched run01. |

In the table above, `run01/` and `run02/` identify the authors' controlled local runs. Their full restricted directories are not part of this repository. These aggregate files do not replace the scripts or authorised SEER input needed to recompute them. They must not be interpreted as external validation. The authors should verify that publication of every aggregate table complies with their data-use agreement, especially any small-cell rule.
