"""Aggregate repeated nested-CV predictions and evaluate all base models.

Only the frozen development partition is used.  Every patient contributes the
mean of three genuinely out-of-fold predictions (one from each outer repeat).
This script creates model-comparison data tables, but deliberately creates no
figures and performs no independent-test evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from comprisk import score_cr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import (
    CensoringDistributionEstimator,
    cumulative_incidence_competing_risks,
)
from sksurv.util import Surv


from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
OOF_DIR = MODEL_ROOT / "nested_oof_folds"
OUTPUT_DIR = MODEL_ROOT / "base_oof_evaluation"
AUDIT_PATH = MODEL_ROOT / "audit/oof_audit_summary.json"
FAMILIES = [
    "cs_cox",
    "fine_gray",
    "elastic_net_cox",
    "linear_survival_svm",
    "kernel_survival_svm",
    "rsf",
    "extra_survival_trees",
    "xgboost_cox",
    "deephit_cr",
]
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
REPORT_TIMES = np.array([12.0, 36.0, 60.0])
EXPECTED_REPEATS = 3
EXPECTED_FOLDS = 5
SIGNATURE = os.environ.get("MPN_RUN_SIGNATURE", "v0.3_nested5x3_inner3_20260718")


def step_value(times: np.ndarray, values: np.ndarray, horizon: float) -> float:
    position = np.searchsorted(times, horizon, side="right") - 1
    return 0.0 if position < 0 else float(values[position])


def observed_cif(event: np.ndarray, time: np.ndarray, horizons: np.ndarray) -> dict[float, float]:
    unique_times, cif = cumulative_incidence_competing_risks(event, time)
    return {
        float(horizon): step_value(unique_times, cif[1], float(horizon))
        for horizon in horizons
    }


def ipcw_binary_at_horizon(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    censoring: CensoringDistributionEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return known-status mask, cause-1 indicator, and IPCW Brier weights."""

    before = time <= horizon
    censored_before = before & (event == 0)
    known = ~censored_before
    target = (before & (event == 1)).astype(int)
    weights = np.zeros(len(time), dtype=float)

    observed_before = before & (event != 0)
    if np.any(observed_before):
        query = np.nextafter(time[observed_before], -np.inf)
        g_event = np.maximum(censoring.predict_proba(query), 1e-6)
        weights[observed_before] = 1.0 / g_event
    after = time > horizon
    if np.any(after):
        g_horizon = max(float(censoring.predict_proba(np.asarray([horizon]))[0]), 1e-6)
        weights[after] = 1.0 / g_horizon
    return known, target, weights


def calibration_summary(
    predictions: dict[str, np.ndarray],
    time: np.ndarray,
    event: np.ndarray,
) -> pd.DataFrame:
    any_event_outcome = Surv.from_arrays(event=event != 0, time=time)
    censoring = CensoringDistributionEstimator().fit(any_event_outcome)
    aj = observed_cif(event, time, REPORT_TIMES)
    rows = []
    for model, prediction in predictions.items():
        for horizon in REPORT_TIMES:
            position = int(np.where(EVAL_TIMES == horizon)[0][0])
            probability = np.clip(prediction[:, position], 1e-6, 1.0 - 1e-6)
            known, target, weights = ipcw_binary_at_horizon(
                time, event, float(horizon), censoring
            )
            known_probability = probability[known]
            known_target = target[known]
            known_weights = weights[known]
            logit_probability = np.log(known_probability / (1.0 - known_probability))

            logistic = LogisticRegression(
                C=np.inf,
                solver="lbfgs",
                max_iter=1000,
                tol=1e-9,
            )
            logistic.fit(
                logit_probability[:, None],
                known_target,
                sample_weight=known_weights,
            )
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic.fit(known_probability, known_target, sample_weight=known_weights)
            calibrated = isotonic.predict(known_probability)
            absolute_error = np.abs(calibrated - known_probability)
            normalized_weight = known_weights / known_weights.sum()
            mean_prediction = float(np.mean(probability))
            observed = aj[float(horizon)]
            rows.append(
                {
                    "model": model,
                    "times": float(horizon),
                    "calibration_intercept": float(logistic.intercept_[0]),
                    "calibration_slope": float(logistic.coef_[0, 0]),
                    "mean_predicted_risk": mean_prediction,
                    "observed_aj_cif": observed,
                    "observed_expected_ratio": observed / mean_prediction,
                    "ici_isotonic": float(np.sum(normalized_weight * absolute_error)),
                    "e50_isotonic": float(
                        np.quantile(absolute_error, 0.50, method="linear")
                    ),
                    "e90_isotonic": float(
                        np.quantile(absolute_error, 0.90, method="linear")
                    ),
                    "known_status_rows": int(np.sum(known)),
                }
            )
    return pd.DataFrame(rows)


def dca_table(
    predictions: dict[str, np.ndarray],
    time: np.ndarray,
    event: np.ndarray,
) -> pd.DataFrame:
    thresholds = np.arange(0.005, 0.201, 0.005)
    any_event_outcome = Surv.from_arrays(event=event != 0, time=time)
    censoring = CensoringDistributionEstimator().fit(any_event_outcome)
    aj = observed_cif(event, time, REPORT_TIMES)
    rows = []
    n = len(time)
    for horizon in REPORT_TIMES:
        position = int(np.where(EVAL_TIMES == horizon)[0][0])
        known, target, weights = ipcw_binary_at_horizon(
            time, event, float(horizon), censoring
        )
        for threshold in thresholds:
            odds = threshold / (1.0 - threshold)
            rows.append(
                {
                    "model": "treat_none",
                    "times": float(horizon),
                    "threshold": float(threshold),
                    "net_benefit": 0.0,
                }
            )
            rows.append(
                {
                    "model": "treat_all",
                    "times": float(horizon),
                    "threshold": float(threshold),
                    "net_benefit": float(
                        aj[float(horizon)] - (1.0 - aj[float(horizon)]) * odds
                    ),
                }
            )
            for model, prediction in predictions.items():
                positive = prediction[:, position] >= threshold
                true_positive = np.sum(weights * known * positive * (target == 1)) / n
                false_positive = np.sum(weights * known * positive * (target == 0)) / n
                rows.append(
                    {
                        "model": model,
                        "times": float(horizon),
                        "threshold": float(threshold),
                        "net_benefit": float(true_positive - false_positive * odds),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit_by_family = {row["family"]: row for row in audit}
    incomplete = [
        family
        for family in FAMILIES
        if family not in audit_by_family or audit_by_family[family]["status"] != "PASS"
    ]
    if incomplete:
        raise RuntimeError(
            f"Base OOF audit must pass for all nine families before aggregation: {incomplete}"
        )

    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")].copy().reset_index(drop=True)
    del frame
    n = len(development)
    accumulators = {
        family: np.zeros((n, len(EVAL_TIMES)), dtype=np.float64) for family in FAMILIES
    }
    counts = {family: np.zeros(n, dtype=np.int8) for family in FAMILIES}

    for family in FAMILIES:
        for repeat in range(1, EXPECTED_REPEATS + 1):
            for fold in range(1, EXPECTED_FOLDS + 1):
                path = OOF_DIR / family / f"repeat_{repeat:02d}_fold_{fold:02d}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["signature"].item()) != SIGNATURE:
                        raise AssertionError(f"signature mismatch: {path}")
                    validation_index = saved["validation_index"].astype(int)
                    prediction = saved["target_cif"].astype(float)
                accumulators[family][validation_index] += prediction
                counts[family][validation_index] += 1
        if not np.all(counts[family] == EXPECTED_REPEATS):
            unique = np.unique(counts[family], return_counts=True)
            raise AssertionError(f"OOF repeat counts failed for {family}: {unique}")

    predictions = {
        family: accumulators[family] / counts[family][:, None] for family in FAMILIES
    }
    time = development["time_months"].to_numpy(dtype=float)
    event = development["event_type"].to_numpy(dtype=int)
    target_outcome = Surv.from_arrays(event=event == 1, time=time)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    scores.auc.to_csv(OUTPUT_DIR / "base_oof_auc_long.csv", index=False, encoding="utf-8-sig")
    scores.brier.to_csv(
        OUTPUT_DIR / "base_oof_brier_long.csv", index=False, encoding="utf-8-sig"
    )
    scores.iauc.to_csv(OUTPUT_DIR / "base_oof_iauc.csv", index=False, encoding="utf-8-sig")
    scores.ibs.to_csv(OUTPUT_DIR / "base_oof_ibs.csv", index=False, encoding="utf-8-sig")
    scores.calibration.to_csv(
        OUTPUT_DIR / "base_oof_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )

    calibration = calibration_summary(predictions, time, event)
    calibration.to_csv(
        OUTPUT_DIR / "base_oof_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    dca = dca_table(predictions, time, event)
    dca.to_csv(OUTPUT_DIR / "base_oof_dca_long.csv", index=False, encoding="utf-8-sig")

    c_rows = []
    position_60 = int(np.where(EVAL_TIMES == 60.0)[0][0])
    for family, prediction in predictions.items():
        risk = prediction[:, position_60]
        harrell = concordance_index_censored(
            target_outcome["event"], target_outcome["time"], risk
        )[0]
        uno = concordance_index_ipcw(
            target_outcome, target_outcome, risk, tau=60.0
        )[0]
        c_rows.append(
            {
                "model": family,
                "harrell_c_cause_specific_60m": float(harrell),
                "uno_c_cause_specific_60m": float(uno),
            }
        )
    c_table = pd.DataFrame(c_rows)
    c_table.to_csv(OUTPUT_DIR / "base_oof_c_indices.csv", index=False, encoding="utf-8-sig")

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
    main = main.sort_values(["IBS", "iAUC"], ascending=[True, False]).reset_index(drop=True)
    main.insert(0, "development_rank_by_ibs", np.arange(1, len(main) + 1))
    main.to_csv(OUTPUT_DIR / "table_base_model_main_comparison.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        OUTPUT_DIR / "development_repeated_oof_cif.npz",
        patient_id=development["patient_id"].to_numpy(),
        time_months=time,
        event_type=event,
        eval_times=EVAL_TIMES,
        **{family: prediction.astype(np.float32) for family, prediction in predictions.items()},
    )
    compact = development[
        ["patient_id", "time_months", "event_type", "mpn_subtype"]
    ].copy()
    for family, prediction in predictions.items():
        for horizon in REPORT_TIMES:
            position = int(np.where(EVAL_TIMES == horizon)[0][0])
            compact[f"{family}_{int(horizon)}m"] = prediction[:, position]
    compact.to_csv(
        OUTPUT_DIR / "development_base_oof_predictions_1_3_5y.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "status": "PASS",
        "partition": "development only",
        "test_partition_used": False,
        "patients": n,
        "outer_cross_validation": "5 folds x 3 repeats",
        "prediction_per_patient_per_model": EXPECTED_REPEATS,
        "primary_metric": "competing-risk IPCW IBS, 12-60 months",
        "model_order_by_ibs": main["model"].tolist(),
    }
    (OUTPUT_DIR / "base_oof_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(main.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
