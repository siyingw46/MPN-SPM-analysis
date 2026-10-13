"""Nested repeated-CV OOF predictions for the nine v0.3 model families.

Each family/fold is an independent checkpoint. Hyperparameters are selected by
3-fold CV inside every outer training fold; the outer design is 5 folds repeated
3 times. The frozen test partition is never scored by this script.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from comprisk import score_cr
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

from modeling_core import (
    CAUSE_LABELS,
    CifPrediction,
    PREDICTOR_COLUMNS,
    make_cause_outcome,
    make_preprocessor,
    validate_cif_prediction,
)
from modeling_v03_core import (
    DeepHitCompetingRiskEstimator,
    FineGrayCifEstimator,
    RiskScoreCoxCalibratedModel,
    XGBoostCoxRiskEstimator,
    predict_cif_from_cause_models,
)


from runtime_paths import work_root
ROOT = work_root()
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
OUTPUT_DIR = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
OOF_DIR = OUTPUT_DIR / "nested_oof_folds"
SEED = int(os.environ.get("MPN_RANDOM_SEED", "20260723"))
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
TUNE_TIMES = np.array([12.0, 24.0, 36.0, 48.0, 60.0])
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
SIGNATURE = os.environ.get("MPN_RUN_SIGNATURE", "v0.3_nested5x3_inner3_20260718")
ALL_FAMILIES = [
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
BASE_ELASTIC_ALPHA = {1: 0.0004981579528423184, 2: 0.002612079608919077, 3: 0.0000690124700268599}


def strata(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["event_type"].astype(str)
        + "|"
        + frame["mpn_subtype"].astype(str)
        + "|"
        + frame["diagnosis_era"].astype(str)
    ).to_numpy()


def candidates_for(family: str) -> list[dict[str, Any]]:
    if family in {"cs_cox", "fine_gray"}:
        return [{}]
    if family == "elastic_net_cox":
        return [
            {"l1_ratio": l1_ratio, "alpha_scale": alpha_scale}
            for l1_ratio in [0.5, 0.9]
            for alpha_scale in [0.5, 1.0, 2.0]
        ]
    if family == "linear_survival_svm":
        return [{"alpha": value} for value in [0.01, 0.1, 1.0]]
    if family == "kernel_survival_svm":
        return [
            {"alpha": 0.1, "gamma": gamma, "n_components": 96}
            for gamma in [0.015, 0.03, 0.06]
        ]
    if family == "rsf":
        return [
            {"min_samples_leaf": 30, "max_features": "sqrt"},
            {"min_samples_leaf": 60, "max_features": "sqrt"},
            {"min_samples_leaf": 30, "max_features": 0.5},
        ]
    if family == "extra_survival_trees":
        return [
            {"min_samples_leaf": 30, "max_features": "sqrt"},
            {"min_samples_leaf": 60, "max_features": "sqrt"},
            {"min_samples_leaf": 30, "max_features": 0.5},
        ]
    if family == "xgboost_cox":
        return [
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 10},
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20},
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 30},
        ]
    if family == "deephit_cr":
        return [
            {
                "num_durations": 24,
                "hidden_nodes": [32, 16],
                "dropout": 0.1,
                "alpha": 0.2,
                "sigma": 0.1,
                "learning_rate": 0.003,
                "inner_epochs": 8,
                "outer_epochs": 16,
            },
            {
                "num_durations": 36,
                "hidden_nodes": [64, 32],
                "dropout": 0.2,
                "alpha": 0.2,
                "sigma": 0.1,
                "learning_rate": 0.001,
                "inner_epochs": 12,
                "outer_epochs": 20,
            },
        ]
    raise ValueError(f"Unknown family: {family}")


def cause_model(family: str, parameters: dict, cause: int, seed: int, outer_fit: bool):
    if family == "cs_cox":
        return CoxPHSurvivalAnalysis(alpha=1e-6, ties="breslow", n_iter=200)
    if family == "elastic_net_cox":
        return CoxnetSurvivalAnalysis(
            alphas=[BASE_ELASTIC_ALPHA[cause] * float(parameters["alpha_scale"])],
            l1_ratio=float(parameters["l1_ratio"]),
            fit_baseline_model=True,
            max_iter=100000,
            tol=1e-7,
        )
    if family == "linear_survival_svm":
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "svm",
                    FastSurvivalSVM(
                        alpha=float(parameters["alpha"]),
                        rank_ratio=1.0,
                        max_iter=20,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return RiskScoreCoxCalibratedModel(base)
    if family == "kernel_survival_svm":
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "kernel",
                    Nystroem(
                        kernel="rbf",
                        gamma=float(parameters["gamma"]),
                        n_components=int(parameters["n_components"]),
                        random_state=seed,
                    ),
                ),
                (
                    "svm",
                    FastSurvivalSVM(
                        alpha=float(parameters["alpha"]),
                        rank_ratio=1.0,
                        max_iter=20,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return RiskScoreCoxCalibratedModel(base)
    if family in {"rsf", "extra_survival_trees"}:
        estimator_class = RandomSurvivalForest if family == "rsf" else ExtraSurvivalTrees
        common = dict(
            n_estimators=150 if outer_fit else 60,
            min_samples_split=10,
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=parameters["max_features"],
            max_samples=0.632,
            n_jobs=-1,
            random_state=seed,
        )
        if family == "extra_survival_trees":
            common["bootstrap"] = True
        return estimator_class(**common)
    if family == "xgboost_cox":
        base = XGBoostCoxRiskEstimator(
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            min_child_weight=float(parameters["min_child_weight"]),
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        return RiskScoreCoxCalibratedModel(base)
    raise ValueError(f"No cause-specific factory for {family}")


def fit_predict(
    family: str,
    parameters: dict,
    x_train: np.ndarray,
    train_frame: pd.DataFrame,
    x_validation: np.ndarray,
    horizons: np.ndarray,
    seed: int,
    outer_fit: bool,
) -> tuple[np.ndarray, dict]:
    if family == "fine_gray":
        model = FineGrayCifEstimator(cause=1, max_iter=50)
        model.fit(
            x_train,
            time=train_frame["time_months"].to_numpy(dtype=float),
            event=train_frame["event_type"].to_numpy(dtype=int),
        )
        prediction = model.predict_cumulative_incidence(x_validation, times=horizons)
        checks = {
            "status": "PASS",
            "fit_method": model.fit_method_,
            "selected_columns": int(len(model.selected_columns_)),
        }
        return prediction, checks
    if family == "deephit_cr":
        model = DeepHitCompetingRiskEstimator(
            num_durations=int(parameters["num_durations"]),
            hidden_nodes=tuple(parameters["hidden_nodes"]),
            dropout=float(parameters["dropout"]),
            alpha=float(parameters["alpha"]),
            sigma=float(parameters["sigma"]),
            learning_rate=float(parameters["learning_rate"]),
            batch_size=256,
            epochs=int(parameters["outer_epochs"] if outer_fit else parameters["inner_epochs"]),
            random_state=seed,
        )
        model.fit(
            x_train,
            train_frame["time_months"].to_numpy(dtype=float),
            train_frame["event_type"].to_numpy(dtype=int),
        )
        cif = model.predict_all_cause_cif(x_validation, horizons)
        stacked = np.stack([cif[cause] for cause in CAUSE_LABELS], axis=0)
        full = CifPrediction(
            horizons=horizons,
            cif=cif,
            survival=1.0 - stacked.sum(axis=0),
            maximum_total_hazard_increment=0.0,
        )
        checks = validate_cif_prediction(full)
        if checks["status"] != "PASS":
            raise AssertionError(checks)
        return cif[1], checks

    models = {}
    optimizer_results = {}
    for cause in CAUSE_LABELS:
        model = cause_model(family, parameters, cause, seed + cause, outer_fit)
        model.fit(x_train, make_cause_outcome(train_frame, cause))
        models[cause] = model
        if family in {"linear_survival_svm", "kernel_survival_svm"}:
            result = model.risk_estimator_.named_steps["svm"].optimizer_result_
            optimizer_results[str(cause)] = {
                "success": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
    full = predict_cif_from_cause_models(models, x_validation, horizons)
    checks = validate_cif_prediction(full)
    if optimizer_results:
        checks["optimizer_results"] = optimizer_results
    if checks["status"] != "PASS":
        raise AssertionError(checks)
    return full.cif[1], checks


def ibs_for(prediction: np.ndarray, validation_frame: pd.DataFrame, times: np.ndarray) -> float:
    scored = score_cr(
        {"candidate": prediction},
        validation_frame["time_months"].to_numpy(dtype=float),
        validation_frame["event_type"].to_numpy(dtype=int),
        times,
        cause=1,
        metrics=("brier",),
    )
    return float(scored.ibs.loc[0, "IBS"])


def checkpoint_path(family: str, repeat: int, fold: int) -> Path:
    return OOF_DIR / family / f"repeat_{repeat:02d}_fold_{fold:02d}.npz"


def run_family_fold(
    family: str,
    repeat: int,
    fold: int,
    outer_train_index: np.ndarray,
    outer_validation_index: np.ndarray,
    development: pd.DataFrame,
) -> None:
    path = checkpoint_path(family, repeat, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            valid_signature = "signature" in saved.files and str(saved["signature"].item()) == SIGNATURE
            valid_indices = np.array_equal(saved["validation_index"], outer_validation_index)
        if valid_signature and valid_indices:
            print(f"{family} repeat={repeat} fold={fold}: checkpoint reused", flush=True)
            return

    started = time.perf_counter()
    outer_train = development.iloc[outer_train_index].copy()
    outer_validation = development.iloc[outer_validation_index].copy()
    candidate_list = candidates_for(family)
    candidate_results = []

    if len(candidate_list) == 1:
        selected = candidate_list[0]
    else:
        inner_splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + repeat * 100 + fold)
        inner_splits = list(
            inner_splitter.split(np.zeros(len(outer_train)), strata(outer_train))
        )
        prepared = []
        for inner_train_pos, inner_validation_pos in inner_splits:
            inner_train = outer_train.iloc[inner_train_pos].copy()
            inner_validation = outer_train.iloc[inner_validation_pos].copy()
            preprocessor = make_preprocessor()
            x_inner_train = np.asarray(
                preprocessor.fit_transform(inner_train[PREDICTOR_COLUMNS]), dtype=np.float64
            )
            x_inner_validation = np.asarray(
                preprocessor.transform(inner_validation[PREDICTOR_COLUMNS]), dtype=np.float64
            )
            prepared.append((inner_train, inner_validation, x_inner_train, x_inner_validation))

        for candidate_index, parameters in enumerate(candidate_list):
            fold_losses = []
            fold_fit_checks = []
            failure = None
            candidate_started = time.perf_counter()
            for inner_fold, (inner_train, inner_validation, x_inner_train, x_inner_validation) in enumerate(prepared, start=1):
                try:
                    prediction, fit_checks = fit_predict(
                        family,
                        parameters,
                        x_inner_train,
                        inner_train,
                        x_inner_validation,
                        TUNE_TIMES,
                        SEED + repeat * 10000 + fold * 1000 + candidate_index * 10 + inner_fold,
                        outer_fit=False,
                    )
                    fold_losses.append(ibs_for(prediction, inner_validation, TUNE_TIMES))
                    fold_fit_checks.append(fit_checks)
                except Exception as error:  # recorded; all-candidate failure remains fatal
                    failure = f"{type(error).__name__}: {error}"
                    fold_losses = []
                    fold_fit_checks = []
                    break
            mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
            candidate_results.append(
                {
                    "candidate_index": candidate_index,
                    "parameters": parameters,
                    "inner_ibs": fold_losses,
                    "inner_fit_checks": fold_fit_checks,
                    "mean_inner_ibs": mean_loss,
                    "failure": failure,
                    "seconds": float(time.perf_counter() - candidate_started),
                }
            )
            print(
                f"{family} repeat={repeat} fold={fold} candidate={candidate_index} "
                f"IBS={mean_loss:.6f}",
                flush=True,
            )
        finite = [row for row in candidate_results if np.isfinite(row["mean_inner_ibs"])]
        if not finite:
            raise RuntimeError(f"All candidates failed for {family}, repeat={repeat}, fold={fold}")
        best = min(finite, key=lambda row: row["mean_inner_ibs"])
        selected = best["parameters"]

    preprocessor = make_preprocessor()
    x_outer_train = np.asarray(
        preprocessor.fit_transform(outer_train[PREDICTOR_COLUMNS]), dtype=np.float64
    )
    x_outer_validation = np.asarray(
        preprocessor.transform(outer_validation[PREDICTOR_COLUMNS]), dtype=np.float64
    )
    prediction, cif_checks = fit_predict(
        family,
        selected,
        x_outer_train,
        outer_train,
        x_outer_validation,
        EVAL_TIMES,
        SEED + repeat * 10000 + fold * 100,
        outer_fit=True,
    )
    outer_ibs = ibs_for(prediction, outer_validation, EVAL_TIMES)
    np.savez_compressed(
        path,
        signature=np.asarray(SIGNATURE),
        family=np.asarray(family),
        repeat=np.asarray(repeat),
        fold=np.asarray(fold),
        validation_index=outer_validation_index,
        patient_id=outer_validation["patient_id"].to_numpy(),
        eval_times=EVAL_TIMES,
        target_cif=prediction.astype(np.float32),
    )
    summary = {
        "signature": SIGNATURE,
        "family": family,
        "repeat": repeat,
        "fold": fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_validation_rows": int(len(outer_validation)),
        "selected_parameters": selected,
        "candidate_results": candidate_results,
        "outer_ibs": outer_ibs,
        "cif_validation": cif_checks,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_partition_used": False,
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{family} repeat={repeat} fold={fold}: saved | outer IBS={outer_ibs:.6f} | "
        f"{summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        default=",".join(ALL_FAMILIES),
        help="Comma-separated model family ids",
    )
    parser.add_argument("--max-outer-splits", type=int, default=15)
    parser.add_argument(
        "--outer-split-start",
        type=int,
        default=0,
        help="Zero-based outer split offset for non-overlapping checkpoint-safe workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = sorted(set(families) - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")].copy().reset_index(drop=True)
    del frame
    outer_splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=SEED
    )
    outer_splits = list(outer_splitter.split(np.zeros(len(development)), strata(development)))
    if args.outer_split_start < 0:
        raise ValueError("--outer-split-start must be non-negative")
    selected_splits = outer_splits[
        args.outer_split_start : args.outer_split_start + int(args.max_outer_splits)
    ]
    for local_split_number, (outer_train_index, outer_validation_index) in enumerate(selected_splits):
        split_number = args.outer_split_start + local_split_number
        repeat = split_number // OUTER_FOLDS + 1
        fold = split_number % OUTER_FOLDS + 1
        for family in families:
            run_family_fold(
                family,
                repeat,
                fold,
                outer_train_index,
                outer_validation_index,
                development,
            )


if __name__ == "__main__":
    main()
