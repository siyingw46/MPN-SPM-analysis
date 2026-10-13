"""Create prespecified no-chemotherapy and temporal v0.4 analysis datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


from runtime_paths import work_root
ROOT = work_root()
MAIN = ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"
MASTER = ROOT / "data/processed/v0.4/patient_master_v0.4_2000_2023.csv"
OUT = ROOT / "data/processed/v0.4"
SEED = 20260723


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    year = result["diagnosis_year"].astype(int)
    result["diagnosis_era"] = np.select(
        [year <= 2008, year <= 2013, year <= 2018],
        ["2004_2008", "2009_2013", "2014_2018"],
        default="2019_2023",
    )
    rng = np.random.default_rng(SEED)
    result["data_split"] = ""
    for _, indices in result.groupby(["event_type", "mpn_subtype", "diagnosis_era"], dropna=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        rng.shuffle(idx)
        n_dev = int(round(len(idx) * 0.70))
        if len(idx) >= 2:
            n_dev = min(max(n_dev, 1), len(idx) - 1)
        result.loc[idx[:n_dev], "data_split"] = "development"
        result.loc[idx[n_dev:], "data_split"] = "test"
    result["split_seed"] = SEED
    return result


def main() -> None:
    main = pd.read_csv(MAIN, encoding="utf-8-sig", low_memory=False)
    no_chemo_2004 = main.drop(columns=["chemotherapy", "chemotherapy_raw"])
    no_chemo_2004.to_csv(OUT / "ml_no_chemotherapy_v0.4_2004_2023.csv", index=False, encoding="utf-8-sig")

    master = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    eligible = master.loc[
        master["time_months"].notna()
        & master["time_months"].ge(1)
        & master["index_year"].between(2000, 2023)
        & master["baseline_malignancy_at_zero"].eq(0)
    ].copy()
    full = pd.DataFrame({
        "patient_id": eligible["patient_id"],
        "age_numeric": eligible["age_numeric"],
        "age_85plus": eligible["age_85plus"],
        "sex": eligible["index_Sex"],
        "race": eligible["index_Race recode (W, B, AI, API)"],
        "mpn_subtype": eligible["index_mpn_subtype"],
        "diagnosis_year": eligible["index_year"],
        "rural_urban": eligible["index_Rural-Urban Continuum Code"],
        "median_household_income": eligible["index_Median household income inflation adj to 2024"],
        "time_months": eligible["time_months"],
        "time_years": eligible["time_years"],
        "event_type": eligible["event_type"],
        "event_spm": eligible["event_spm"],
        "event_hematologic": eligible["event_hematologic"],
        "event_death": eligible["event_death"],
    })
    full = assign_split(full.reset_index(drop=True))
    full.to_csv(OUT / "ml_no_chemotherapy_v0.4_2000_2023.csv", index=False, encoding="utf-8-sig")
    temporal = main.loc[main["diagnosis_year"].between(2004, 2018)].copy()
    temporal["data_split"] = np.where(temporal["diagnosis_year"].between(2004, 2015), "development", "test")
    temporal.to_csv(OUT / "ml_temporal_v0.4_2004_2018.csv", index=False, encoding="utf-8-sig")
    summary = {
        "no_chemotherapy_2004_2023_rows": len(no_chemo_2004),
        "no_chemotherapy_2000_2023_rows": len(full),
        "temporal_2004_2015_training_rows": int((temporal["data_split"] == "development").sum()),
        "temporal_2016_2018_test_rows": int((temporal["data_split"] == "test").sum()),
    }
    (OUT / "sensitivity_dataset_summary_v0.4.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
