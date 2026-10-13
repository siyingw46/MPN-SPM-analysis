"""Prespecified sensitivity analyses for the locked XGBoost-Cox primary model."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from comprisk import score_cr

from runtime_paths import work_root
ROOT = work_root()
OUT = ROOT / "deliverables" / "v0.4" / "tables" / "source_data"
MODEL_ROOT = ROOT / "output" / "models" / "model_development_v0.4"
TIMES = np.arange(12.0, 61.0, 1.0)
PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20}


def evaluate(name: str, path: Path, omit_chemotherapy: bool) -> pd.DataFrame:
    os.environ["MPN_ANALYSIS_VERSION"] = "v0.4"
    os.environ["MPN_OMIT_CHEMOTHERAPY"] = "1" if omit_chemotherapy else "0"
    import modeling_core
    importlib.reload(modeling_core)
    nested = importlib.import_module("11_v03_nested_oof")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    train = data.loc[data.data_split.eq("development")].copy().reset_index(drop=True)
    test = data.loc[data.data_split.eq("test")].copy().reset_index(drop=True)
    prep = modeling_core.make_preprocessor()
    x_train = np.asarray(prep.fit_transform(train[modeling_core.PREDICTOR_COLUMNS]), dtype=float)
    x_test = np.asarray(prep.transform(test[modeling_core.PREDICTOR_COLUMNS]), dtype=float)
    pred, _ = nested.fit_predict("xgboost_cox", PARAMS, x_train, train, x_test, TIMES, 20260723, outer_fit=True)
    result = score_cr({"xgboost_cox": pred}, test.time_months.to_numpy(float), test.event_type.to_numpy(int), TIMES, cause=1, metrics=("auc", "brier"))
    row = {"analysis": name, "n_train": len(train), "n_test": len(test),
           "events_test": int(test.event_type.eq(1).sum()),
           "iAUC": float(result.iauc.iloc[0].iAUC), "IBS": float(result.ibs.iloc[0].IBS)}
    for horizon in (12, 36, 60):
        row[f"AUC_{horizon}m"] = float(result.auc.loc[np.isclose(result.auc.times, horizon), "AUC"].iloc[0])
        row[f"Brier_{horizon}m"] = float(result.brier.loc[np.isclose(result.brier.times, horizon), "Brier"].iloc[0])
    return pd.DataFrame([row])


def subtype_performance() -> pd.DataFrame:
    with np.load(MODEL_ROOT / "final_test_evaluation" / "test_all_models_cif.npz", allow_pickle=False) as saved:
        pred = saved["xgboost_cox"].astype(float)
        time = saved["time_months"].astype(float)
        event = saved["event_type"].astype(int)
    data = pd.read_csv(ROOT / "data" / "processed" / "v0.4" / "ml_competing_risk_v0.4_2004_2023.csv", encoding="utf-8-sig")
    test = data.loc[data.data_split.eq("test")].reset_index(drop=True)
    rows = []
    for subtype in ("PV", "ET", "PMF"):
        mask = test.mpn_subtype.eq(subtype).to_numpy()
        score = score_cr({"xgboost_cox": pred[mask]}, time[mask], event[mask], TIMES, cause=1, metrics=("auc", "brier"))
        rows.append({"analysis": f"locked_global_model_{subtype}", "n_test": int(mask.sum()), "events_test": int((event[mask] == 1).sum()),
                     "iAUC": float(score.iauc.iloc[0].iAUC), "IBS": float(score.ibs.iloc[0].IBS)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate("no_chemotherapy_2004_2023", ROOT / "data/processed/v0.4/ml_no_chemotherapy_v0.4_2004_2023.csv", True),
        evaluate("no_chemotherapy_2000_2023", ROOT / "data/processed/v0.4/ml_no_chemotherapy_v0.4_2000_2023.csv", True),
        evaluate("temporal_2004_2015_to_2016_2018", ROOT / "data/processed/v0.4/ml_temporal_v0.4_2004_2018.csv", False),
    ]
    pd.concat(rows, ignore_index=True).to_csv(OUT / "Table_S3_primary_model_sensitivity.csv", index=False, encoding="utf-8-sig")
    subtype_performance().to_csv(OUT / "Table_S4_locked_model_by_MPN_subtype.csv", index=False, encoding="utf-8-sig")
    print("PASS")

if __name__ == "__main__":
    main()
