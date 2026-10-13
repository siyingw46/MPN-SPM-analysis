"""Audit, aggregate, and compare nested ensemble OOF predictions.

The script combines ensemble and base-model OOF predictions on the development
partition only.  It applies the predeclared rule that an ensemble advances only
if its IBS is no worse than the best base model while iAUC is not more than
0.005 lower.  No figures and no independent-test evaluation are produced.
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
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from modeling_v03_ensemble import validate_target_cif


evaluation = importlib.import_module("13_v03_aggregate_base_oof")

from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
BASE_EVALUATION_DIR = MODEL_ROOT / "base_oof_evaluation"
ENSEMBLE_FOLD_DIR = MODEL_ROOT / "nested_ensemble_folds"
OUTPUT_DIR = MODEL_ROOT / "ensemble_oof_evaluation"
SIGNATURE = os.environ.get("MPN_ENSEMBLE_SIGNATURE", "v0.3_nested_ensemble_inner3_20260719")
FAMILIES = list(evaluation.FAMILIES)
ENSEMBLES = [
    "equal_weight_voting",
    "performance_weighted_voting",
    "nonnegative_linear_stacking",
    "regularized_ridge_meta",
]
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
REPORT_TIMES = np.array([12.0, 36.0, 60.0])
EXPECTED_REPEATS = 3
EXPECTED_FOLDS = 5
NUMERICAL_IBS_TOLERANCE = 1e-10


def main_comparison(
    predictions: dict[str, np.ndarray],
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    scores = score_cr(
        predictions,
        time,
        event,
        EVAL_TIMES,
        cause=1,
        metrics=("auc", "brier"),
        calibration_at=REPORT_TIMES,
        calibration_n_bins=10,
    )
    target_outcome = Surv.from_arrays(event=event == 1, time=time)
    position_60 = int(np.where(EVAL_TIMES == 60.0)[0][0])
    c_rows = []
    for model, prediction in predictions.items():
        risk = prediction[:, position_60]
        c_rows.append(
            {
                "model": model,
                "harrell_c_cause_specific_60m": float(
                    concordance_index_censored(
                        target_outcome["event"], target_outcome["time"], risk
                    )[0]
                ),
                "uno_c_cause_specific_60m": float(
                    concordance_index_ipcw(
                        target_outcome, target_outcome, risk, tau=60.0
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
    main["model_class"] = np.where(main["model"].isin(ENSEMBLES), "ensemble", "base")
    main = main.sort_values(["IBS", "iAUC"], ascending=[True, False]).reset_index(drop=True)
    main.insert(0, "development_rank_by_ibs", np.arange(1, len(main) + 1))
    return main, scores, c_table


def main() -> None:
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")].copy().reset_index(drop=True)
    del frame
    n = len(development)
    development_ids = development["patient_id"].astype(str).to_numpy()
    accumulators = {
        ensemble: np.zeros((n, len(EVAL_TIMES)), dtype=np.float64)
        for ensemble in ENSEMBLES
    }
    counts = np.zeros(n, dtype=np.int8)
    fold_rows = []
    weight_rows = []
    inclusion_counter: Counter[str] = Counter()
    problems = []

    for repeat in range(1, EXPECTED_REPEATS + 1):
        for fold in range(1, EXPECTED_FOLDS + 1):
            path = ENSEMBLE_FOLD_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}.npz"
            json_path = path.with_suffix(".json")
            if not path.exists() or not json_path.exists():
                problems.append(f"missing checkpoint or metadata: repeat={repeat}, fold={fold}")
                continue
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != SIGNATURE:
                    problems.append(f"signature mismatch: {path.name}")
                validation_index = saved["validation_index"].astype(int)
                patient_id = saved["patient_id"].astype(str)
                eligible = saved["eligible_models"].astype(str).tolist()
                if not np.array_equal(patient_id, development_ids[validation_index]):
                    problems.append(f"patient/index mismatch: {path.name}")
                counts[validation_index] += 1
                for ensemble in ENSEMBLES:
                    prediction = saved[ensemble].astype(float)
                    check = validate_target_cif(prediction)
                    if check["status"] != "PASS":
                        problems.append(f"invalid CIF {ensemble}: {path.name}: {check}")
                    accumulators[ensemble][validation_index] += prediction
            if metadata.get("test_partition_used") is not False:
                problems.append(f"test-use flag is not false: {json_path.name}")
            inclusion_counter.update(eligible)
            for ensemble in ENSEMBLES:
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "ensemble": ensemble,
                        "outer_ibs": float(metadata["outer_ibs"][ensemble]),
                        "eligible_models": "|".join(eligible),
                    }
                )
            parameter_block = metadata["ensemble_parameters"]
            method_weights = {
                "equal_weight_voting": parameter_block["equal_weight_voting"],
                "performance_weighted_voting": parameter_block[
                    "performance_weighted_voting"
                ],
                "nonnegative_linear_stacking": parameter_block[
                    "nonnegative_linear_stacking"
                ]["weights"],
                "regularized_ridge_meta": parameter_block["regularized_ridge_meta"][
                    "coefficients"
                ],
            }
            for method, weights in method_weights.items():
                for family, weight in weights.items():
                    weight_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "ensemble": method,
                            "base_model": family,
                            "weight_or_coefficient": float(weight),
                            "ridge_intercept": (
                                float(
                                    parameter_block["regularized_ridge_meta"]["intercept"]
                                )
                                if method == "regularized_ridge_meta"
                                else np.nan
                            ),
                            "ridge_alpha": (
                                float(
                                    parameter_block["regularized_ridge_meta"][
                                        "selected_alpha"
                                    ]
                                )
                                if method == "regularized_ridge_meta"
                                else np.nan
                            ),
                        }
                    )

    if problems:
        raise RuntimeError("Ensemble audit failed:\n" + "\n".join(problems))
    if not np.all(counts == EXPECTED_REPEATS):
        raise AssertionError(
            f"Ensemble OOF coverage failed: {np.unique(counts, return_counts=True)}"
        )
    ensemble_predictions = {
        ensemble: accumulators[ensemble] / counts[:, None] for ensemble in ENSEMBLES
    }
    with np.load(
        BASE_EVALUATION_DIR / "development_repeated_oof_cif.npz", allow_pickle=False
    ) as saved:
        if not np.array_equal(saved["patient_id"].astype(str), development_ids):
            raise AssertionError("Base OOF and development patient order do not match")
        base_predictions = {family: saved[family].astype(float) for family in FAMILIES}
    all_predictions = {**base_predictions, **ensemble_predictions}
    time = development["time_months"].to_numpy(dtype=float)
    event = development["event_type"].to_numpy(dtype=int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main, scores, c_table = main_comparison(all_predictions, time, event)
    best_base = main.loc[main["model_class"].eq("base")].sort_values("IBS").iloc[0]
    retention = []
    for ensemble in ENSEMBLES:
        row = main.loc[main["model"].eq(ensemble)].iloc[0]
        ibs_noninferior = bool(
            row["IBS"] <= best_base["IBS"] + NUMERICAL_IBS_TOLERANCE
        )
        iauc_noninferior = bool(row["iAUC"] >= best_base["iAUC"] - 0.005)
        retention.append(
            {
                "ensemble": ensemble,
                "best_base_model": str(best_base["model"]),
                "ensemble_ibs": float(row["IBS"]),
                "best_base_ibs": float(best_base["IBS"]),
                "ibs_difference_vs_best_base": float(row["IBS"] - best_base["IBS"]),
                "ensemble_iauc": float(row["iAUC"]),
                "best_base_iauc": float(best_base["iAUC"]),
                "iauc_difference_vs_best_base": float(row["iAUC"] - best_base["iAUC"]),
                "ibs_no_worse_than_best_base": ibs_noninferior,
                "iauc_not_lower_by_more_than_0.005": iauc_noninferior,
                "advance_to_final_test": bool(ibs_noninferior and iauc_noninferior),
            }
        )
    retention_table = pd.DataFrame(retention)
    retention_map = dict(
        zip(retention_table["ensemble"], retention_table["advance_to_final_test"])
    )
    main["advance_to_final_test"] = main.apply(
        lambda row: retention_map.get(row["model"], np.nan), axis=1
    )

    scores.auc.to_csv(OUTPUT_DIR / "all_oof_auc_long.csv", index=False, encoding="utf-8-sig")
    scores.brier.to_csv(
        OUTPUT_DIR / "all_oof_brier_long.csv", index=False, encoding="utf-8-sig"
    )
    scores.iauc.to_csv(OUTPUT_DIR / "all_oof_iauc.csv", index=False, encoding="utf-8-sig")
    scores.ibs.to_csv(OUTPUT_DIR / "all_oof_ibs.csv", index=False, encoding="utf-8-sig")
    scores.calibration.to_csv(
        OUTPUT_DIR / "all_oof_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )
    c_table.to_csv(OUTPUT_DIR / "all_oof_c_indices.csv", index=False, encoding="utf-8-sig")
    calibration = evaluation.calibration_summary(all_predictions, time, event)
    calibration.to_csv(
        OUTPUT_DIR / "all_oof_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    dca = evaluation.dca_table(all_predictions, time, event)
    dca.to_csv(OUTPUT_DIR / "all_oof_dca_long.csv", index=False, encoding="utf-8-sig")
    main.to_csv(
        OUTPUT_DIR / "table_all_models_main_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retention_table.to_csv(
        OUTPUT_DIR / "table_ensemble_retention_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "ensemble_outer_fold_ibs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(weight_rows).to_csv(
        OUTPUT_DIR / "ensemble_weights_and_coefficients_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    inclusion = pd.DataFrame(
        [
            {
                "base_model": family,
                "selected_outer_folds": int(inclusion_counter[family]),
                "selection_frequency": float(inclusion_counter[family] / 15.0),
            }
            for family in FAMILIES
        ]
    ).sort_values(["selection_frequency", "base_model"], ascending=[False, True])
    inclusion.to_csv(
        OUTPUT_DIR / "ensemble_base_model_inclusion_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        OUTPUT_DIR / "development_all_models_repeated_oof_cif.npz",
        patient_id=development["patient_id"].to_numpy(),
        time_months=time,
        event_type=event,
        eval_times=EVAL_TIMES,
        **{name: value.astype(np.float32) for name, value in all_predictions.items()},
    )
    summary = {
        "status": "PASS",
        "partition": "development only",
        "test_partition_used": False,
        "ensemble_folds_audited": 15,
        "patient_prediction_count": 3,
        "best_base_model": str(best_base["model"]),
        "best_base_ibs": float(best_base["IBS"]),
        "retention_rule": (
            "advance if development nested-OOF IBS <= best base IBS (with 1e-10 "
            "floating-point tolerance) and iAUC is not more than 0.005 lower than "
            "best base iAUC"
        ),
        "ensembles_advancing": retention_table.loc[
            retention_table["advance_to_final_test"], "ensemble"
        ].tolist(),
        "ensembles_removed": retention_table.loc[
            ~retention_table["advance_to_final_test"], "ensemble"
        ].tolist(),
    }
    (OUTPUT_DIR / "ensemble_oof_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(main.to_string(index=False), flush=True)
    print(retention_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
