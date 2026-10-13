# Second primary malignancy after classical MPN: analysis code

## Start here

**Run `run_pipeline.py`, not the numbered files in `scripts/`.** The numbered Python and R files are called by the stage runner in a fixed order. Their `v03`/`v04` names reflect development history, not a request to run an older analysis. The reported analysis configuration is **v0.4**, using 5,000 patient-level bootstrap resamples.

| What you want to do | Start with | What happens |
|---|---|---|
| Understand the workflow | [Pipeline and script map](docs/PIPELINE_MAP.md) | Lists all 13 stages, prerequisites, scripts, and output locations; no analysis runs. |
| Check your computer | `python check_environment.py` and `python run_pipeline.py plan` | Checks software and prints stage dependencies; no model is fitted. |
| Reproduce the analysis | Read [INPUT_CONTRACT.md](INPUT_CONTRACT.md), then follow **Run instructions** below | Requires your own authorised SEER export and a new restricted run directory outside this repository. |
| Find a paper table or figure | [Result-to-source map](docs/RESULT_MAP.md) | Shows where the pipeline writes scientific source results. Final Word/Illustrator layouts are separate. |

The [Chinese quick start](README_ZH.md) gives the same entry sequence. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the verification scope and its limits.

This repository contains the Python and R analysis code for a SEER-based study of non-haematological second primary malignancies after polycythaemia vera, essential thrombocythaemia, or primary myelofibrosis. It corresponds to the controlled **local-rerun result set** selected by the authors for reporting; it is not an archive of the earlier historical ML outputs.

The repository includes scripts, configurations, synthetic tests, the observed software environment, and aggregate reference metrics. It does **not** include SEER case-level records, patient-level predictions, SHAP values, fitted models, model parameters, or checkpoints. Do not commit a run directory or a SEER*Stat case-listing export.

## Data access and input

Researchers must obtain their own authorised access to the applicable SEER database and produce the labelled, tab-delimited tumour-level export specified in [INPUT_CONTRACT.md](INPUT_CONTRACT.md). The original extraction used SEER*Stat **9.0.43.0** and the November 2025 submission of SEER Research Limited-Field Data, 21 registries excluding Illinois, covering diagnosis years 2000–2023. The input export and any optional reference cohort must remain in controlled local storage. Access and permitted use are governed by the researcher's SEER agreement: <https://seer.cancer.gov/data/access.html>.

The exact 29 expected input labels are declared as `RAW_FIELDS` in `scripts/python/01_build_datasets.py`. The export must retain all available records for people selected through the first-primary MPN index record, including both SEER sequence categories *One primary only* and *1st of 2 or more primaries*. Numeric-only sequence values are not accepted by this parser. See [INPUT_CONTRACT.md](INPUT_CONTRACT.md) before attempting a run.

## Software environment

The controlled reruns used Windows 11 x64, CPython **3.12.10**, R **4.6.1**, and random seed **20260723**. The authors completed a controlled full rerun on a local computer and repeated key stages in a second run. This public code package has not yet been installed from scratch and run end-to-end on another independent computer. `environment/requirements-observed-run02.txt` records all 59 observed Python distributions; `environment/R_sessionInfo_run01.txt` records the R session. `environment/requirements-rebuild-windows-cpu.txt` is a candidate installation recipe, **not** proof that a clean installation has been tested. See [environment/SOFTWARE_VERSIONS.md](environment/SOFTWARE_VERSIONS.md).

No package installer or patient data are bundled. Install the matching software in an authorised local environment before running the code. The repository does not automatically download packages or data.

## Run instructions

Use PowerShell from the repository root. First inspect the plan and environment; neither command fits a model:

```powershell
& 'PATH_TO_PYTHON_3_12_10' check_environment.py
& 'PATH_TO_PYTHON_3_12_10' run_pipeline.py plan
```

Create a **new directory outside the repository** for restricted inputs and all generated outputs:

```powershell
& 'PATH_TO_PYTHON_3_12_10' run_pipeline.py init `
  --run-dir 'NEW_EMPTY_RESTRICTED_RUN_DIRECTORY' `
  --raw 'AUTHORISED_LABELLED_SEER_EXPORT.txt'
```

If an authorised reference patient-level cohort from the same export is available, add `--reference-cohort 'AUTHORISED_REFERENCE_COHORT.csv'` to validate patient order, split, event time, and event type. Without it, cohort equivalence to the authors' local run remains unverified.

Run the stages **one at a time** in this order: `data`, `descriptive`, `r_models`, `r_figures`, `event_r`, `base_oof`, `ensemble_oof`, `locked`, `ml_sensitivity`, `revision`, `deephit`, `bootstrap`, `report`. See the [stage map](docs/PIPELINE_MAP.md) for what each stage does and where its outputs go.

```powershell
& 'PATH_TO_PYTHON_3_12_10' run_pipeline.py run `
  --run-dir 'NEW_EMPTY_RESTRICTED_RUN_DIRECTORY' `
  --stage STAGE_NAME `
  --rscript 'PATH_TO_RSCRIPT' `
  --r-library 'PATH_TO_LOCAL_R_LIBRARY' `
  --allow-analysis
```

The runner never silently executes all stages, retries a failed stage, or overwrites an attempted stage. Inspect the local stage log if a command fails. The `bootstrap` stage uses **5,000 patient-level resamples**, not the obsolete 1,000-resample implementation.

After a complete run, compare the four aggregate locked-test bootstrap tables with the bundled reference metrics:

```powershell
& 'PATH_TO_PYTHON_3_12_10' compare_results.py --run-dir 'NEW_EMPTY_RESTRICTED_RUN_DIRECTORY'
```

This comparison is **not** a complete verification of every table, figure, sensitivity analysis, or inference. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the validated scope and limitations.

## Reporting figures without refitting models

For an existing completed restricted run, set `MPN_RUN_ROOT` to that run and use `scripts/python/final_reporting.py --figure4-only --output-pdf NEW.pdf` or `--figureS5-only --output-pdf NEW.pdf`. These commands draw figures from existing outputs; they do not retrain or bootstrap models. They refuse to overwrite an existing PDF.

## Repository boundaries

- `scripts/`, `config/`, `tests/`: analysis code, model settings, and synthetic checks.
- `environment/`: observed versions and a candidate reconstruction recipe.
- `reference_metrics/`: aggregate outputs for comparison only; not raw data or independent external validation.
- `docs/`: stage-by-stage navigation and publication-result provenance.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): what was and was not repeated in controlled local runs.

The scripts retain some legacy directory names (including `_codex_workspace`) as **relative internal paths** because the controlled runner constructs that layout inside the restricted run directory. Do not place patient-level run outputs inside the public repository. **The authors have intentionally not provided a code licence.** Public access to this repository does not itself grant a general right to reuse, modify, or redistribute the code; contact the authors for permissions beyond those provided by applicable law or platform terms. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before publishing.
