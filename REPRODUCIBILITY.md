# Reproducibility scope

This is a code-and-aggregate-results release candidate, not a data release or an independent validation study.

## Verified in the controlled local environment

- One full run completed the 13 explicitly ordered pipeline stages.
- A second controlled run repeated cohort preparation, base-model development, ensemble development, locked-test prediction, expanded DeepHit training, and 5,000 patient-level bootstrap resamples.
- The 85,680-row modelling cohort's patient order, development/test assignment, event time, and event type matched the authors' authorised reference in the local audit.
- The locked prediction bundle, final DeepHit prediction bundle, and four publication-facing bootstrap aggregate CSVs matched between the two local runs. The primary model selection was not redone from locked-test results.
- The traditional R audit compared 38 aggregate CSVs and 33,732 numeric cells within its defined scope.

These statements describe the authors' controlled local audit. The restricted logs, input hashes, reference cohort, and patient-level predictions are **not** distributed. A third party cannot check those particular identity comparisons without independently authorised data and the relevant reference files.

## Not established

- The older historical ML outputs were not reproduced exactly; the manuscript uses the coherent local-rerun result set instead.
- A clean install on another computer or cross-platform bitwise equality has not been tested.
- Sensitivity, subgroup, risk-stratification, DCA, and interpretation analyses were completed in the full run but were not all independently repeated in the second run.
- No external-cohort validation or prospective clinical-impact evaluation was performed.
- The aggregate comparator checks four bootstrap tables, **not** all paper numbers, figure panels, or methods.

The SEER input and all derived patient-level outputs remain restricted. The applicable SEER agreement controls access and use. No trained model or model parameter is included in this repository.
