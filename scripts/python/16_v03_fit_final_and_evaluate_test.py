"""Fit all v0.3 base families on development and evaluate the locked test set.

Development decisions are frozen before test scoring:
* primary model: XGBoost-Cox;
* secondary sensitivity candidate: regularized ridge meta;
* all other base families: prespecified comparators.

The script creates point-estimate tables only.  Patient-level bootstrap CIs are
handled separately after this output passes audit.
"""

from __future__ import annotations

import importlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from comprisk import score_cr
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from modeling_core import PREDICTOR_COLUMNS, make_preprocessor
from modeling_v03_ensemble import (
    fit_nonnegative_simplex_stack,
    fit_ridge_meta,
    predict_ridge_meta,
    weighted_average,
)


nested = importlib.import_module("11_v03_nested_oof")
base_evaluation = importlib.import_module("13_v03_aggregate_base_oof")
ensemble_build = importlib.import_module("14_v03_nested_ensembles")

from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
BASE_FOLD_DIR = MODEL_ROOT / "nested_oof_folds"
BASE_OOF_DIR = MODEL_ROOT / "base_oof_evaluation"
ENSEMBLE_OOF_DIR = MODEL_ROOT / "ensemble_oof_evaluation"
OUTPUT_DIR = MODEL_ROOT / "final_test_evaluation"
FAMILIES = list(nested.ALL_FAMILIES)
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
REPORT_TIMES = np.array([12.0, 36.0, 60.0])
SEED = int(os.environ.get("MPN_RANDOM_SEED", "20260723"))


def canonical(parameters: dict) -> str:
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def select_final_parameters(family: str) -> tuple[dict, dict]:
    rows = []
    for path in sorted((BASE_FOLD_DIR / family).glob("repeat_*_fold_*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "parameters": metadata["selected_parameters"],
                "key": canonical(metadata["selected_parameters"]),
                "outer_ibs": float(metadata["outer_ibs"]),
            }
        )
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 audited folds for {family}, found {len(rows)}")
    counts = Counter(row["key"] for row in rows)
    maximum = max(counts.values())
    tied = [key for key, count in counts.items() if count == maximum]
    if len(tied) > 1:
        tied.sort(
            key=lambda key: np.mean(
                [row["outer_ibs"] for row in rows if row["key"] == key]
            )
        )
    selected_key = tied[0]
    selected = next(row["parameters"] for row in rows if row["key"] == selected_key)
    details = {
        "rule": "most frequent outer-fold selection; tie broken by mean outer IBS",
        "selection_counts": dict(counts),
        "selected_parameters": selected,
        "selected_count": counts[selected_key],
    }
    return selected, details


def base_fold_ibs() -> dict[str, list[float]]:
    result = {}
    for family in FAMILIES:
        values = []
        for path in sorted((BASE_FOLD_DIR / family).glob("repeat_*_fold_*.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            values.append(float(metadata["outer_ibs"]))
        if len(values) != 15:
            raise RuntimeError(f"Missing fold IBS values for {family}")
        result[family] = values
    return result


def strata(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["event_type"].astype(str)
        + "|"
        + frame["mpn_subtype"].astype(str)
        + "|"
        + frame["diagnosis_era"].astype(str)
    ).to_numpy()


def main_comparison(
    predictions: dict[str, np.ndarray],
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    test_time = test["time_months"].to_numpy(dtype=float)
    test_event = test["event_type"].to_numpy(dtype=int)
    scores = score_cr(
        predictions,
        test_time,
        test_event,
        EVAL_TIMES,
        cause=1,
        metrics=("auc", "brier"),
        calibration_at=REPORT_TIMES,
        calibration_n_bins=10,
    )
    development_target = Surv.from_arrays(
        event=development["event_type"].to_numpy(dtype=int) == 1,
        time=development["time_months"].to_numpy(dtype=float),
    )
    test_target = Surv.from_arrays(event=test_event == 1, time=test_time)
    position_60 = int(np.where(EVAL_TIMES == 60.0)[0][0])
    c_rows = []
    for model, prediction in predictions.items():
        risk = prediction[:, position_60]
        c_rows.append(
            {
                "model": model,
                "harrell_c_cause_specific_60m": float(
                    concordance_index_censored(
                        test_target["event"], test_target["time"], risk
                    )[0]
                ),
                "uno_c_cause_specific_60m": float(
                    concordance_index_ipcw(
                        development_target, test_target, risk, tau=60.0
                    )[0]
                ),
            }
        )
    c_table = pd.DataFrame(c_rows)
    main = scores.iauc[["model", "iAUC"]].merge(
        scores.ibs[["model", "IBS"]], on="model", how="outer"
    ).merge(c_table, on="model", how="outer")
    for horizon in REPORT_TIMES:
        auc_at = scores.auc.loc[
            np.isclose(scores.auc["times"], horizon), ["model", "AUC"]
        ].rename(columns={"AUC": f"AUC_{int(horizon)}m"})
        brier_at = scores.brier.loc[
            np.isclose(scores.brier["times"], horizon), ["model", "Brier"]
        ].rename(columns={"Brier": f"Brier_{int(horizon)}m"})
        main = main.merge(auc_at, on="model", how="outer").merge(
            brier_at, on="model", how="outer"
        )
    main["development_role_locked_before_test"] = main["model"].map(
        {"xgboost_cox": "primary"}
    ).fillna("comparator")
    role_order = {"primary": 0, "comparator": 1}
    main["role_order"] = main["development_role_locked_before_test"].map(role_order)
    main = main.sort_values(["role_order", "IBS"]).drop(columns="role_order").reset_index(drop=True)
    return main, scores, c_table


def main() -> None:
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")].copy().reset_index(drop=True)
    test = frame.loc[frame["data_split"].eq("test")].copy().reset_index(drop=True)
    del frame
    preprocessor = make_preprocessor()
    x_development = np.asarray(
        preprocessor.fit_transform(development[PREDICTOR_COLUMNS]), dtype=np.float64
    )
    x_test = np.asarray(
        preprocessor.transform(test[PREDICTOR_COLUMNS]), dtype=np.float64
    )

    final_parameters = {}
    parameter_details = {}
    base_predictions = {}
    fit_checks = {}
    for family_index, family in enumerate(FAMILIES):
        parameters, details = select_final_parameters(family)
        final_parameters[family] = parameters
        parameter_details[family] = details
        prediction, checks = nested.fit_predict(
            family,
            parameters,
            x_development,
            development,
            x_test,
            EVAL_TIMES,
            SEED + family_index * 100,
            outer_fit=True,
        )
        base_predictions[family] = prediction
        fit_checks[family] = checks
        print(f"final fit {family}: PASS", flush=True)

    iauc_table = pd.read_csv(BASE_OOF_DIR / "base_oof_iauc.csv")
    development_iauc = {
        str(row.model): float(row.iAUC) for row in iauc_table.itertuples(index=False)
    }
    eligible, eligibility = ensemble_build.select_eligible_models(
        base_fold_ibs(), development_iauc
    )
    with np.load(
        BASE_OOF_DIR / "development_repeated_oof_cif.npz", allow_pickle=False
    ) as saved:
        development_oof_stack = np.stack(
            [saved[family].astype(float) for family in eligible], axis=2
        )
    meta_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_assignment = np.zeros(len(development), dtype=np.int8)
    for fold, (_, validation_index) in enumerate(
        meta_splitter.split(np.zeros(len(development)), strata(development)), start=1
    ):
        fold_assignment[validation_index] = fold
    retention = pd.read_csv(ENSEMBLE_OOF_DIR / "table_ensemble_retention_decision.csv")
    advancing_ensembles = retention.loc[
        retention["advance_to_final_test"].astype(bool), "ensemble"
    ].astype(str).tolist()
    equal_weights = np.repeat(1.0 / len(eligible), len(eligible))
    fold_ibs = base_fold_ibs()
    mean_ibs = {family: float(np.mean(values)) for family, values in fold_ibs.items()}
    inverse_ibs = np.asarray(
        [1.0 / (mean_ibs[family] ** 2) for family in eligible], dtype=float
    )
    performance_weights = inverse_ibs / inverse_ibs.sum()
    stack_weights, stack_optimization = fit_nonnegative_simplex_stack(
        development_oof_stack,
        development["time_months"].to_numpy(dtype=float),
        development["event_type"].to_numpy(dtype=int),
        EVAL_TIMES,
    )
    ridge_coefficients, ridge_alpha, ridge_cv = fit_ridge_meta(
        development_oof_stack,
        development["time_months"].to_numpy(dtype=float),
        development["event_type"].to_numpy(dtype=int),
        EVAL_TIMES,
        fold_assignment,
    )
    test_stack = np.stack([base_predictions[family] for family in eligible], axis=2)
    ensemble_predictions = {
        "equal_weight_voting": weighted_average(test_stack, equal_weights),
        "performance_weighted_voting": weighted_average(test_stack, performance_weights),
        "nonnegative_linear_stacking": weighted_average(test_stack, stack_weights),
        "regularized_ridge_meta": predict_ridge_meta(test_stack, ridge_coefficients),
    }
    predictions = {
        **base_predictions,
        **{
            name: prediction
            for name, prediction in ensemble_predictions.items()
            if name in advancing_ensembles
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table, scores, c_table = main_comparison(predictions, development, test)
    scores.auc.to_csv(OUTPUT_DIR / "test_auc_long.csv", index=False, encoding="utf-8-sig")
    scores.brier.to_csv(
        OUTPUT_DIR / "test_brier_long.csv", index=False, encoding="utf-8-sig"
    )
    scores.iauc.to_csv(OUTPUT_DIR / "test_iauc.csv", index=False, encoding="utf-8-sig")
    scores.ibs.to_csv(OUTPUT_DIR / "test_ibs.csv", index=False, encoding="utf-8-sig")
    scores.calibration.to_csv(
        OUTPUT_DIR / "test_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )
    c_table.to_csv(OUTPUT_DIR / "test_c_indices.csv", index=False, encoding="utf-8-sig")
    calibration = base_evaluation.calibration_summary(
        predictions,
        test["time_months"].to_numpy(dtype=float),
        test["event_type"].to_numpy(dtype=int),
    )
    calibration.to_csv(
        OUTPUT_DIR / "test_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    dca = base_evaluation.dca_table(
        predictions,
        test["time_months"].to_numpy(dtype=float),
        test["event_type"].to_numpy(dtype=int),
    )
    dca.to_csv(OUTPUT_DIR / "test_dca_long.csv", index=False, encoding="utf-8-sig")
    main_table.to_csv(
        OUTPUT_DIR / "table_test_main_comparison.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        OUTPUT_DIR / "test_all_models_cif.npz",
        patient_id=test["patient_id"].to_numpy(),
        time_months=test["time_months"].to_numpy(dtype=float),
        event_type=test["event_type"].to_numpy(dtype=int),
        eval_times=EVAL_TIMES,
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    summary = {
        "status": "PASS",
        "development_rows": int(len(development)),
        "test_rows": int(len(test)),
        "selection_locked_before_test": {
            "primary": "xgboost_cox",
            "reason": (
                "XGBoost was designated as the prespecified primary base model before "
                "independent-test scoring; ensemble candidates advance only after passing "
                "the predeclared development nested-OOF retention rule."
            ),
        },
        "final_parameters": parameter_details,
        "fit_checks": fit_checks,
        "ensembles": {
            "development_retention_table": str(
                ENSEMBLE_OOF_DIR / "table_ensemble_retention_decision.csv"
            ),
            "advanced_to_final_test": advancing_ensembles,
            "eligible_members": eligible,
            "eligibility": eligibility,
            "equal_weight_voting": dict(zip(eligible, map(float, equal_weights))),
            "performance_weighted_voting": dict(
                zip(eligible, map(float, performance_weights))
            ),
            "nonnegative_linear_stacking": {
                "weights": dict(zip(eligible, map(float, stack_weights))),
                "optimization": stack_optimization,
            },
            "regularized_ridge_meta": {
                "selected_alpha": ridge_alpha,
                "intercept": float(ridge_coefficients[0]),
                "coefficients": dict(zip(eligible, map(float, ridge_coefficients[1:]))),
                "alpha_cv": ridge_cv,
            },
        },
        "bootstrap_completed": False,
        "figures_completed": False,
    }
    (OUTPUT_DIR / "final_test_point_estimate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(main_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
