"""Build four leakage-safe ensembles inside each outer validation fold.

Base models are screened and ensemble parameters are estimated exclusively from
three-fold OOF predictions within the outer training partition.  The frozen
test partition is never loaded or scored.
"""

from __future__ import annotations

import argparse
import os
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from comprisk import score_cr
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from modeling_core import PREDICTOR_COLUMNS, make_preprocessor
from modeling_v03_ensemble import (
    fit_nonnegative_simplex_stack,
    fit_ridge_meta,
    predict_ridge_meta,
    validate_target_cif,
    weighted_average,
)


nested = importlib.import_module("11_v03_nested_oof")

from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
BASE_DIR = MODEL_ROOT / "nested_oof_folds"
OUTPUT_DIR = MODEL_ROOT / "nested_ensemble_folds"
SEED = int(os.environ.get("MPN_RANDOM_SEED", "20260723"))
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
SIGNATURE = os.environ.get("MPN_ENSEMBLE_SIGNATURE", "v0.3_nested_ensemble_inner3_20260719")
# Ensemble candidates are the stable development-stage finalists: conventional
# and regularized Cox models plus the three best-performing tree/boosting
# families.  SVM variants remain prespecified stand-alone comparators but are
# excluded here because of convergence warnings; DeepHit is excluded because of
# materially inferior nested-OOF IBS.  The independent test set is never used.
FAMILIES = [
    "cs_cox",
    "elastic_net_cox",
    "rsf",
    "extra_survival_trees",
    "xgboost_cox",
]
ENSEMBLES = [
    "equal_weight_voting",
    "performance_weighted_voting",
    "nonnegative_linear_stacking",
    "regularized_ridge_meta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-outer-splits", type=int, default=15)
    parser.add_argument("--outer-split-start", type=int, default=0)
    return parser.parse_args()


def strata(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["event_type"].astype(str)
        + "|"
        + frame["mpn_subtype"].astype(str)
        + "|"
        + frame["diagnosis_era"].astype(str)
    ).to_numpy()


def checkpoint_path(repeat: int, fold: int) -> Path:
    return OUTPUT_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}.npz"


def selected_parameters(family: str, repeat: int, fold: int) -> dict:
    path = BASE_DIR / family / f"repeat_{repeat:02d}_fold_{fold:02d}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("test_partition_used") is not False:
        raise AssertionError(f"Unexpected test-use flag in {path}")
    return metadata["selected_parameters"]


def outer_base_predictions(
    repeat: int, fold: int, expected_validation_index: np.ndarray
) -> dict[str, np.ndarray]:
    predictions = {}
    for family in FAMILIES:
        path = BASE_DIR / family / f"repeat_{repeat:02d}_fold_{fold:02d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["validation_index"], expected_validation_index):
                raise AssertionError(f"Base outer indices do not match: {path}")
            predictions[family] = saved["target_cif"].astype(float)
    return predictions


def select_eligible_models(
    fold_ibs: dict[str, list[float]], iauc: dict[str, float]
) -> tuple[list[str], dict]:
    mean_ibs = {family: float(np.mean(values)) for family, values in fold_ibs.items()}
    best_family = min(FAMILIES, key=lambda family: mean_ibs[family])
    best_values = np.asarray(fold_ibs[best_family], dtype=float)
    best_se = float(np.std(best_values, ddof=1) / np.sqrt(len(best_values)))
    ibs_threshold = mean_ibs[best_family] + best_se
    iauc_threshold = max(iauc.values()) - 0.02
    eligible = [
        family
        for family in FAMILIES
        if mean_ibs[family] <= ibs_threshold and iauc[family] >= iauc_threshold
    ]
    ranked = sorted(FAMILIES, key=lambda family: (mean_ibs[family], -iauc[family]))
    if len(eligible) < 3:
        eligible = ranked[:3]
    eligible = sorted(eligible, key=lambda family: (mean_ibs[family], -iauc[family]))[:6]
    return eligible, {
        "best_inner_ibs_family": best_family,
        "best_inner_ibs": mean_ibs[best_family],
        "best_inner_ibs_standard_error": best_se,
        "one_standard_error_ibs_threshold": ibs_threshold,
        "maximum_inner_iauc": max(iauc.values()),
        "iauc_eligibility_threshold": iauc_threshold,
        "minimum_models": 3,
        "maximum_models": 6,
        "mean_inner_ibs": mean_ibs,
        "inner_iauc": iauc,
        "eligible_models": eligible,
    }


def run_fold(
    repeat: int,
    fold: int,
    outer_train_index: np.ndarray,
    outer_validation_index: np.ndarray,
    development: pd.DataFrame,
) -> None:
    path = checkpoint_path(repeat, fold)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            valid = (
                "signature" in saved.files
                and str(saved["signature"].item()) == SIGNATURE
                and np.array_equal(saved["validation_index"], outer_validation_index)
            )
        if valid:
            print(f"ensemble repeat={repeat} fold={fold}: checkpoint reused", flush=True)
            return

    started = time.perf_counter()
    outer_train = development.iloc[outer_train_index].copy().reset_index(drop=True)
    outer_validation = development.iloc[outer_validation_index].copy()
    parameters = {
        family: selected_parameters(family, repeat, fold) for family in FAMILIES
    }
    inner_splitter = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + repeat * 100 + fold,
    )
    inner_splits = list(
        inner_splitter.split(np.zeros(len(outer_train)), strata(outer_train))
    )
    inner_predictions = {
        family: np.zeros((len(outer_train), len(EVAL_TIMES)), dtype=np.float64)
        for family in FAMILIES
    }
    fold_assignment = np.zeros(len(outer_train), dtype=np.int8)
    fold_ibs = {family: [] for family in FAMILIES}
    fit_checks: dict[str, list[dict]] = {family: [] for family in FAMILIES}

    for inner_fold, (inner_train_pos, inner_validation_pos) in enumerate(
        inner_splits, start=1
    ):
        fold_assignment[inner_validation_pos] = inner_fold
        inner_train = outer_train.iloc[inner_train_pos].copy()
        inner_validation = outer_train.iloc[inner_validation_pos].copy()
        preprocessor = make_preprocessor()
        x_inner_train = np.asarray(
            preprocessor.fit_transform(inner_train[PREDICTOR_COLUMNS]), dtype=np.float64
        )
        x_inner_validation = np.asarray(
            preprocessor.transform(inner_validation[PREDICTOR_COLUMNS]), dtype=np.float64
        )
        for family_index, family in enumerate(FAMILIES):
            prediction, checks = nested.fit_predict(
                family,
                parameters[family],
                x_inner_train,
                inner_train,
                x_inner_validation,
                EVAL_TIMES,
                SEED
                + repeat * 100000
                + fold * 10000
                + inner_fold * 100
                + family_index,
                outer_fit=False,
            )
            inner_predictions[family][inner_validation_pos] = prediction
            fold_ibs[family].append(
                nested.ibs_for(prediction, inner_validation, EVAL_TIMES)
            )
            fit_checks[family].append(checks)
            print(
                f"ensemble repeat={repeat} fold={fold} inner={inner_fold} "
                f"family={family} IBS={fold_ibs[family][-1]:.6f}",
                flush=True,
            )

    inner_score = score_cr(
        inner_predictions,
        outer_train["time_months"].to_numpy(dtype=float),
        outer_train["event_type"].to_numpy(dtype=int),
        EVAL_TIMES,
        cause=1,
        metrics=("auc",),
    )
    iauc = {
        str(row.model): float(row.iAUC) for row in inner_score.iauc.itertuples(index=False)
    }
    eligible, eligibility = select_eligible_models(fold_ibs, iauc)
    inner_stack = np.stack([inner_predictions[family] for family in eligible], axis=2)
    outer_predictions = outer_base_predictions(
        repeat, fold, outer_validation_index
    )
    outer_stack = np.stack([outer_predictions[family] for family in eligible], axis=2)
    inner_time = outer_train["time_months"].to_numpy(dtype=float)
    inner_event = outer_train["event_type"].to_numpy(dtype=int)

    equal_weights = np.repeat(1.0 / len(eligible), len(eligible))
    inverse_ibs = np.asarray(
        [1.0 / (eligibility["mean_inner_ibs"][family] ** 2) for family in eligible]
    )
    performance_weights = inverse_ibs / inverse_ibs.sum()
    stack_weights, stack_optimization = fit_nonnegative_simplex_stack(
        inner_stack, inner_time, inner_event, EVAL_TIMES
    )
    ridge_coefficients, ridge_alpha, ridge_cv = fit_ridge_meta(
        inner_stack,
        inner_time,
        inner_event,
        EVAL_TIMES,
        fold_assignment,
    )
    ensemble_predictions = {
        "equal_weight_voting": weighted_average(outer_stack, equal_weights),
        "performance_weighted_voting": weighted_average(
            outer_stack, performance_weights
        ),
        "nonnegative_linear_stacking": weighted_average(outer_stack, stack_weights),
        "regularized_ridge_meta": predict_ridge_meta(
            outer_stack, ridge_coefficients
        ),
    }
    validations = {
        name: validate_target_cif(prediction)
        for name, prediction in ensemble_predictions.items()
    }
    failed = [name for name, check in validations.items() if check["status"] != "PASS"]
    if failed:
        raise AssertionError(f"Ensemble CIF validation failed: {failed}")
    outer_ibs = {
        name: nested.ibs_for(prediction, outer_validation, EVAL_TIMES)
        for name, prediction in ensemble_predictions.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(SIGNATURE),
        repeat=np.asarray(repeat),
        fold=np.asarray(fold),
        validation_index=outer_validation_index,
        patient_id=outer_validation["patient_id"].to_numpy(),
        eval_times=EVAL_TIMES,
        eligible_models=np.asarray(eligible),
        **{
            name: prediction.astype(np.float32)
            for name, prediction in ensemble_predictions.items()
        },
    )
    summary = {
        "signature": SIGNATURE,
        "repeat": repeat,
        "fold": fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_validation_rows": int(len(outer_validation)),
        "test_partition_used": False,
        "base_selected_parameters": parameters,
        "base_inner_fold_ibs": fold_ibs,
        "base_inner_fit_checks": fit_checks,
        "eligibility": eligibility,
        "ensemble_parameters": {
            "equal_weight_voting": dict(zip(eligible, map(float, equal_weights))),
            "performance_weighted_voting": dict(
                zip(eligible, map(float, performance_weights))
            ),
            "nonnegative_linear_stacking": {
                "weights": dict(zip(eligible, map(float, stack_weights))),
                "optimization": stack_optimization,
            },
            "regularized_ridge_meta": {
                "intercept": float(ridge_coefficients[0]),
                "coefficients": dict(
                    zip(eligible, map(float, ridge_coefficients[1:]))
                ),
                "selected_alpha": ridge_alpha,
                "alpha_cv": ridge_cv,
                "postprocessing": "clip to [0,1], then cumulative maximum over time",
            },
        },
        "outer_ibs": outer_ibs,
        "cif_validation": validations,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"ensemble repeat={repeat} fold={fold}: saved | eligible={eligible} | "
        f"IBS={outer_ibs} | {summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")].copy().reset_index(drop=True)
    del frame
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )
    outer_splits = list(splitter.split(np.zeros(len(development)), strata(development)))
    if args.outer_split_start < 0:
        raise ValueError("--outer-split-start must be non-negative")
    selected_splits = outer_splits[
        args.outer_split_start : args.outer_split_start + int(args.max_outer_splits)
    ]
    for local_split_number, (outer_train_index, outer_validation_index) in enumerate(selected_splits):
        split_number = args.outer_split_start + local_split_number
        repeat = split_number // OUTER_FOLDS + 1
        fold = split_number % OUTER_FOLDS + 1
        run_fold(
            repeat,
            fold,
            outer_train_index,
            outer_validation_index,
            development,
        )


if __name__ == "__main__":
    main()
