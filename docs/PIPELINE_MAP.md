# Pipeline and script map

The only full-analysis entrypoint is `run_pipeline.py`. It invokes the scripts below; do not execute them individually to reconstruct a manuscript result. `python run_pipeline.py plan` prints dependencies without fitting a model. A real run needs an authorised input, `init`, an external restricted run directory, and an explicit `--allow-analysis` for each stage. The stage order is defined in `run_pipeline.py:STAGES`.

| Phase | Stage | Must follow | Invoked scripts, in order | Main role / where to look after completion |
|---|---|---|---|---|
| Cohort | `data` | `init` | `01_build_datasets.py`; `02_validate_datasets.py`; `build_r_inputs.py` | Construct and check the patient cohort and R inputs under the restricted run's `data/`, `ML/`, and validation outputs. Contains patient-level files; **never publish the run directory**. |
| Description | `descriptive` | `data` | `20_v04_descriptive_delivery.py` | Baseline, first-event, and CIF source tables under `deliverables/v0.4/tables/source_data/`. |
| Traditional analysis | `r_models` | `data` | `traditional.R`; `exclusion_comparison.R` | KM, Cox, time-varying, landmark, and exclusion results under the restricted run's `_codex_workspace/25_income_reanalysis_20261001/` and related subdirectories. |
| Traditional figures/tables | `r_figures` | `r_models` | `publication_tables.R`; `figure_S2.R` | Corrected Table 1/2/S1/S3/S4 sources and KM Figure S2 components. |
| Event timing | `event_r` | `descriptive` | `event_timing.R`; `latency6.R` | Same-month event and 3/6-month conditional-CIF sensitivity outputs under `_codex_workspace/09_event_timing_sensitivity/`. |
| Base models | `base_oof` | `data` | `11_v03_nested_oof.py`; `12_v03_oof_audit.py`; `13_v03_aggregate_base_oof.py` | Nested development out-of-fold predictions and aggregate base-model metrics under `output/models/model_development_v0.4/`. |
| Ensembles | `ensemble_oof` | `base_oof` | `14_v03_nested_ensembles.py`; `15_v03_audit_and_evaluate_ensembles.py` | Ensemble development, weights, and retention decision under `output/models/model_development_v0.4/ensemble_oof_evaluation/`. |
| Locked evaluation | `locked` | `ensemble_oof` | `16_v03_fit_final_and_evaluate_test.py` | Fixed-model predictions, metrics, calibration, and DCA under `output/models/model_development_v0.4/final_test_evaluation/`. Prediction files are patient-level; keep restricted. |
| Model sensitivity | `ml_sensitivity` | `locked`, `descriptive` | `19_v04_build_sensitivity_datasets.py`; `21_v04_sensitivity_xgboost.py` | Chemotherapy-free/temporal and PV/ET/PMF analyses; see `deliverables/v0.4/tables/source_data/`. |
| Additional analyses | `revision` | `data` | `01_core_revision_analyses.py` | Alternative event-definition model refits and interpretation outputs under `revision/`; includes patient-level SHAP sample output, which must remain restricted. |
| DeepHit | `deephit` | `locked` | `02_deephit_retraining.py` | Expanded development-data search, final training, and locked predictions under `revision/05_deephit_retraining/`. |
| Bootstrap | `bootstrap` | `deephit` | `04_prepare_bootstrap5000_inputs.py`; `17_v03_bootstrap_test_metrics.py` | Final fixed-prediction, **5,000-resample** aggregate evaluation under `revision/04_bootstrap5000/model_root/final_test_evaluation/`. |
| Paper sources | `report` | `r_figures`, `event_r`, `ml_sensitivity`, `revision`, `bootstrap` | `final_reporting.py` | Scientific source tables and figure files under `final_results/`; consult [result map](RESULT_MAP.md). Manual Word/Illustrator layout is not reproduced byte-for-byte. |

The table follows the runner's actual dependencies. Several branches can in principle complete after the shared cohort stage, but the [README](../README.md) recommends the established serial order for a controlled reproduction. Do not infer that a stage passed merely because its output directory exists; inspect restricted `status.json` and stage logs.

## Filenames that look older but are active

- `v03`/`v04` in filenames and `config/model_registry_v0.3.yml` are historical internal labels. The runner still invokes them for the **v0.4** analysis. Do not rename them without updating imports, runner references, tests, and provenance hashes.
- `scripts/revision/` contains analyses incorporated after the initial pipeline; it is **not** a folder to skip. The final DeepHit/5,000-bootstrap stages run from it.
- `modeling_core.py`, `modeling_v03_core.py`, `modeling_v03_ensemble.py`, `runtime_paths.py`, `publication_functions.py`, and `interpretation_figure.py` are support modules imported by stages. They are not extra independent stages.
- `tests/` contains synthetic/static checks, not the clinical analysis or proof that another computer can exactly reproduce the authors' run.

The implementation retains some historical version words in internal docstrings. This navigation guide identifies their current role; it does **not** change, rename, or rerun scientific code.
