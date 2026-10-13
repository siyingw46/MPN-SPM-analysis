"""Hard validation and descriptive QA for the production patient/ML tables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


from runtime_paths import work_root
ROOT = work_root()
ANALYSIS_VERSION = os.environ.get("MPN_ANALYSIS_VERSION", "v0.4")
V04 = ANALYSIS_VERSION == "v0.4"
MASTER = ROOT / ("data/processed/v0.4/patient_master_v0.4_2000_2023.csv" if V04 else "data/processed/patient_master_2000_2023.csv")
ML = ROOT / ("data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv" if V04 else "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv")
RUN_ID = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_{ANALYSIS_VERSION}_dataset_validation")
RUN_DIR = ROOT / "logs/runs" / RUN_ID


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    master = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    ml = pd.read_csv(ML, encoding="utf-8-sig", low_memory=False)

    checks = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("master_patient_unique", not master["patient_id"].duplicated().any(), f"duplicates={master['patient_id'].duplicated().sum()}")
    check("ml_patient_unique", not ml["patient_id"].duplicated().any(), f"duplicates={ml['patient_id'].duplicated().sum()}")
    check("ml_subset_of_master", set(ml["patient_id"]).issubset(set(master["patient_id"])), "All ML patients must exist in master")
    check("ml_year_range", ml["diagnosis_year"].between(2004, 2023).all(), f"range={ml['diagnosis_year'].min()}-{ml['diagnosis_year'].max()}")
    check("ml_time_minimum", (ml["time_months"] >= 1).all(), f"min={ml['time_months'].min()}")
    check("event_domain", set(ml["event_type"].unique()).issubset({0, 1, 2, 3}), str(sorted(ml['event_type'].unique())))
    check("event_spm_consistent", (ml["event_spm"] == (ml["event_type"] == 1).astype(int)).all(), "event_spm must match event_type=1")
    check("event_death_consistent", (ml["event_death"] == (ml["event_type"] == 2).astype(int)).all(), "event_death must match event_type=2")
    if V04:
        check("event_hematologic_consistent", (ml["event_hematologic"] == (ml["event_type"] == 3).astype(int)).all(), "event_hematologic must match event_type=3")
    else:
        check("event_aml_consistent", (ml["event_aml"] == (ml["event_type"] == 3).astype(int)).all(), "event_aml must match event_type=3")
    check("split_domain", set(ml["data_split"].unique()) == {"development", "test"}, str(ml['data_split'].value_counts().to_dict()))
    development_fraction = (ml["data_split"] == "development").mean()
    check("split_fraction", abs(development_fraction - 0.70) < 0.005, f"development_fraction={development_fraction:.6f}")
    check("no_post_outcome_predictors", not any(name in ml.columns for name in ["survival_status", "total_tumor_count", "first_nonaml_spm_histology", "reporting_source"]), str(list(ml.columns)))
    check("age_domain", ml["age_numeric"].between(18, 85).all(), f"range={ml['age_numeric'].min()}-{ml['age_numeric'].max()}")
    check("age_topcode_consistency", (ml["age_85plus"] == (ml["age_numeric"] == 85).astype(int)).all(), "85+ indicator must match top-coded age")
    check("chemotherapy_domain", set(ml["chemotherapy"].unique()).issubset({0, 1}), str(sorted(ml['chemotherapy'].unique())))

    event_counts = ml.groupby(["data_split", "event_type"], dropna=False).size().reset_index(name="n")
    subtype_counts = ml.groupby(["data_split", "mpn_subtype"], dropna=False).size().reset_index(name="n")
    missingness = ml.isna().sum().reset_index()
    missingness.columns = ["variable", "n_missing"]
    missingness["pct_missing"] = missingness["n_missing"] / len(ml) * 100

    pd.DataFrame(checks).to_csv(RUN_DIR / "validation_checks.csv", index=False, encoding="utf-8-sig")
    event_counts.to_csv(RUN_DIR / "event_counts_by_split.csv", index=False, encoding="utf-8-sig")
    subtype_counts.to_csv(RUN_DIR / "subtype_counts_by_split.csv", index=False, encoding="utf-8-sig")
    missingness.to_csv(RUN_DIR / "ml_missingness.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_id": RUN_ID,
        "status": "PASS",
        "master_rows": int(len(master)),
        "master_columns": int(master.shape[1]),
        "ml_rows": int(len(ml)),
        "ml_columns": int(ml.shape[1]),
        "development_rows": int((ml["data_split"] == "development").sum()),
        "test_rows": int((ml["data_split"] == "test").sum()),
        "event_counts": {str(k): int(v) for k, v in ml["event_type"].value_counts().sort_index().items()},
        "time_range_months": [int(ml["time_months"].min()), int(ml["time_months"].max())],
        "master_sha256": sha256(MASTER),
        "ml_sha256": sha256(ML),
    }
    (RUN_DIR / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
