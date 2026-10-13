# Input and estimand contract

1. The original export is tumour-level, tab-delimited, UTF-8/BOM-compatible and contains named columns. All 29 expected labels are listed in `scripts/python/01_build_datasets.py:RAW_FIELDS`.
2. The original SEER selection includes both **One primary only** (00) and **1st of 2 or more primaries** (01), with all available records for selected people. The code consumes these value labels, not raw numeric sequence codes. Numeric exports must not silently enter this parser.
3. Required fields include patient identifier, record number, sequence number, year, morphology/behaviour, survival months, vital status and the eight predictor domains. Exact spelling matters.
4. Index selection, synchronous exclusions, derived intervals and ties follow the supplied v0.4 implementation. This package does not repair unavailable registry records or silently exclude sequence-gap patients.
5. Main event: first subsequent non-haematological malignant SPM; death and haematological/MPN competing events retain their implemented coding. Time origin remains MPN diagnosis.
6. R income grouping uses exact labels and a separate Unknown category. The ML income predictor retains its original categorical representation; do not replace it with the R grouped numeric representation.
7. The generated R index input uses the raw index columns retained in the patient master and the ML cohort's row order; no second patient-selection algorithm is used. In the controlled local run, the 85,680-row model cohort matched the authorised reference on patient order, split, event time and event type. This does not establish that unobserved registry records could be recovered.
8. The main cohort/split counts are assertions, not substitutes for checking identities and input provenance. Use `--reference-cohort` to validate patient order, split, event and time against the authorised historical file.
9. The original export and processed files remain controlled inputs. Do not publish them with this code package. Input format checking and cohort comparison execute locally only when the researcher explicitly runs the stages.

## Distinct populations that must not be collapsed

- R 3/6-month CIF analyses exclude early first events **and early censoring**, then estimate conditional CIF at horizons since diagnosis.
- The Python early-event sensitivity excludes first events in months1–2 but retains early-censored patients, as implemented in the supplied analysis.
- The six-month analysis does not retrain/evaluate XGBoost–Cox.
- S7 evaluates the original locked predictions (with final DeepHit replacement); S8E and interpretability use refitted models.

These are retained deliberately. Packaging must not change populations merely to make sample sizes match.
