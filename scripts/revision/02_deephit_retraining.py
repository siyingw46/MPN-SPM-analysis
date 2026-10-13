"""Expanded nested-CV DeepHit training with development-only early stopping."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["MPN_ANALYSIS_VERSION"] = "v0.4"

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from comprisk import score_cr
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

import sys

from runtime_paths import work_root
ROOT = work_root()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from modeling_core import PREDICTOR_COLUMNS, make_preprocessor  # noqa: E402
from modeling_v03_core import ReshapeDeepHitOutput  # noqa: E402


DATA = ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"
OUT_ROOT = ROOT / "revision/05_deephit_retraining"
FOLD_DIR = OUT_ROOT / "nested_folds"
SEED = 20260723
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
TUNE_TIMES = np.array([12.0, 24.0, 36.0, 48.0, 60.0])
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
REPORT_TIMES = (12.0, 36.0, 60.0)
MAX_EPOCHS = 200
PATIENCE = 15
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 1024
SIGNATURE = "v3_deephit_nested5x3_inner3_earlystop_20260917"

# Permit several non-overlapping fold workers to run safely on one workstation.
# The default is deliberately conservative; callers may override it per worker.
torch.set_num_threads(int(os.environ.get("MPN_TORCH_THREADS", "2")))
torch.set_num_interop_threads(1)

CANDIDATES = [
    {
        "id": "DH24_small_lr3e-3",
        "num_durations": 24,
        "hidden_nodes": [32, 16],
        "dropout": 0.1,
        "learning_rate": 0.003,
        "alpha": 0.2,
        "sigma": 0.1,
    },
    {
        "id": "DH36_medium_lr1e-3",
        "num_durations": 36,
        "hidden_nodes": [64, 32],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "alpha": 0.2,
        "sigma": 0.1,
    },
    {
        "id": "DH60_medium_lr5e-4",
        "num_durations": 60,
        "hidden_nodes": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.0005,
        "alpha": 0.2,
        "sigma": 0.1,
    },
    {
        "id": "DH36_large_lr1e-3",
        "num_durations": 36,
        "hidden_nodes": [128, 64],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "alpha": 0.2,
        "sigma": 0.1,
    },
]


def strata(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.event_type.astype(str)
        + "|"
        + frame.mpn_subtype.astype(str)
        + "|"
        + frame.diagnosis_era.astype(str)
    ).to_numpy()


def trap_mean(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.trapezoid(values[valid], times[valid]) / (times[valid][-1] - times[valid][0]))


def ibs(prediction: np.ndarray, frame: pd.DataFrame, times: np.ndarray) -> float:
    scored = score_cr(
        {"deephit": prediction},
        frame.time_months.to_numpy(float),
        frame.event_type.to_numpy(int),
        times,
        cause=1,
        metrics=("brier",),
    )
    return trap_mean(times, scored.brier.Brier.to_numpy(float))


def build_model(n_features: int, parameters: dict, duration_index: np.ndarray) -> DeepHit:
    num_risks = 3
    base = tt.practical.MLPVanilla(
        n_features,
        list(parameters["hidden_nodes"]),
        num_risks * int(parameters["num_durations"]),
        batch_norm=True,
        dropout=float(parameters["dropout"]),
    )
    network = torch.nn.Sequential(
        base, ReshapeDeepHitOutput.make(num_risks, int(parameters["num_durations"]))
    )
    model = DeepHit(
        network,
        tt.optim.Adam,
        alpha=float(parameters["alpha"]),
        sigma=float(parameters["sigma"]),
        duration_index=duration_index,
    )
    model.optimizer.set_lr(float(parameters["learning_rate"]))
    return model


def fit_predict(
    x_train: np.ndarray,
    train: pd.DataFrame,
    x_prediction: np.ndarray,
    horizons: np.ndarray,
    parameters: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_prediction = np.asarray(x_prediction, dtype=np.float32)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
    )
    fit_pos, early_pos = next(splitter.split(np.zeros(len(train)), train.event_type.to_numpy(int)))
    fit_frame = train.iloc[fit_pos]
    early_frame = train.iloc[early_pos]
    labtrans = LabTransDiscreteTime(int(parameters["num_durations"]))
    y_fit = labtrans.fit_transform(
        fit_frame.time_months.to_numpy(np.float32), fit_frame.event_type.to_numpy(np.int64)
    )
    y_early = labtrans.transform(
        early_frame.time_months.to_numpy(np.float32), early_frame.event_type.to_numpy(np.int64)
    )
    y_fit = (y_fit[0], y_fit[1].astype(np.int64))
    y_early = (y_early[0], y_early[1].astype(np.int64))
    model = build_model(x_train.shape[1], parameters, labtrans.cuts)
    callback = tt.callbacks.EarlyStopping(
        metric="loss",
        dataset="val",
        patience=PATIENCE,
        min_delta=1e-5,
        checkpoint_model=True,
        load_best=True,
        rm_file=True,
    )
    log = model.fit(
        x_train[fit_pos],
        y_fit,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[callback],
        verbose=False,
        val_data=(x_train[early_pos], y_early),
        val_batch_size=4096,
    )
    history = log.to_pandas()
    raw = np.asarray(model.predict_cif(x_prediction, numpy=True, to_cpu=True), dtype=float)
    positions = np.searchsorted(np.asarray(labtrans.cuts, float), horizons, side="right") - 1
    positions = np.clip(positions, 0, len(labtrans.cuts) - 1)
    target = raw[0, positions, :].T
    metadata = {
        "epochs_completed": int(len(history)),
        "best_validation_loss": float(history["val_loss"].min()),
        "final_validation_loss": float(history["val_loss"].iloc[-1]),
        "fit_rows": int(len(fit_pos)),
        "early_stopping_rows": int(len(early_pos)),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
    }
    return target, metadata


def checkpoint_paths(repeat: int, fold: int) -> tuple[Path, Path]:
    stem = FOLD_DIR / f"repeat_{repeat:02d}_fold_{fold:02d}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def run_fold(
    development: pd.DataFrame,
    repeat: int,
    fold: int,
    outer_train_index: np.ndarray,
    outer_validation_index: np.ndarray,
) -> None:
    npz_path, json_path = checkpoint_paths(repeat, fold)
    if npz_path.exists() and json_path.exists():
        with np.load(npz_path, allow_pickle=False) as saved:
            if (
                str(saved["signature"].item()) == SIGNATURE
                and np.array_equal(saved["validation_index"], outer_validation_index)
            ):
                print(f"repeat={repeat} fold={fold}: checkpoint reused", flush=True)
                return
    started = time.perf_counter()
    outer_train = development.iloc[outer_train_index].copy().reset_index(drop=True)
    outer_validation = development.iloc[outer_validation_index].copy().reset_index(drop=True)
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + repeat * 100 + fold,
    )
    prepared = []
    for inner_train_pos, inner_val_pos in inner.split(np.zeros(len(outer_train)), strata(outer_train)):
        inner_train = outer_train.iloc[inner_train_pos].copy().reset_index(drop=True)
        inner_val = outer_train.iloc[inner_val_pos].copy().reset_index(drop=True)
        preprocessor = make_preprocessor()
        x_train = np.asarray(preprocessor.fit_transform(inner_train[PREDICTOR_COLUMNS]), dtype=float)
        x_val = np.asarray(preprocessor.transform(inner_val[PREDICTOR_COLUMNS]), dtype=float)
        prepared.append((inner_train, inner_val, x_train, x_val))
    candidate_results = []
    for candidate_index, parameters in enumerate(CANDIDATES):
        losses, fit_logs = [], []
        candidate_started = time.perf_counter()
        for inner_fold, (inner_train, inner_val, x_train, x_val) in enumerate(prepared, start=1):
            prediction, log = fit_predict(
                x_train,
                inner_train,
                x_val,
                TUNE_TIMES,
                parameters,
                SEED + repeat * 10000 + fold * 1000 + candidate_index * 10 + inner_fold,
            )
            losses.append(ibs(prediction, inner_val, TUNE_TIMES))
            fit_logs.append(log)
        candidate_results.append(
            {
                "candidate": parameters,
                "inner_ibs": losses,
                "mean_inner_ibs": float(np.mean(losses)),
                "fit_logs": fit_logs,
                "seconds": float(time.perf_counter() - candidate_started),
            }
        )
        print(
            f"repeat={repeat} fold={fold} candidate={parameters['id']} "
            f"IBS={np.mean(losses):.6f}",
            flush=True,
        )
    selected_result = min(candidate_results, key=lambda row: row["mean_inner_ibs"])
    selected = selected_result["candidate"]
    preprocessor = make_preprocessor()
    x_outer_train = np.asarray(
        preprocessor.fit_transform(outer_train[PREDICTOR_COLUMNS]), dtype=float
    )
    x_outer_validation = np.asarray(
        preprocessor.transform(outer_validation[PREDICTOR_COLUMNS]), dtype=float
    )
    prediction, outer_log = fit_predict(
        x_outer_train,
        outer_train,
        x_outer_validation,
        EVAL_TIMES,
        selected,
        SEED + repeat * 10000 + fold * 100,
    )
    outer_ibs = ibs(prediction, outer_validation, EVAL_TIMES)
    np.savez_compressed(
        npz_path,
        signature=np.asarray(SIGNATURE),
        repeat=np.asarray(repeat),
        fold=np.asarray(fold),
        validation_index=outer_validation_index,
        patient_id=outer_validation.patient_id.to_numpy(),
        eval_times=EVAL_TIMES,
        target_cif=prediction.astype(np.float32),
    )
    summary = {
        "signature": SIGNATURE,
        "repeat": repeat,
        "fold": fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_validation_rows": int(len(outer_validation)),
        "candidate_results": candidate_results,
        "selected_candidate": selected,
        "outer_fit_log": outer_log,
        "outer_ibs": outer_ibs,
        "elapsed_seconds": float(time.perf_counter() - started),
        "locked_test_used": False,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"repeat={repeat} fold={fold}: saved; selected={selected['id']} "
        f"outer IBS={outer_ibs:.6f}; {summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


def evaluate(development: pd.DataFrame, test: pd.DataFrame, prediction: np.ndarray) -> dict:
    scores = score_cr(
        {"deephit_cr_retrained": prediction},
        test.time_months.to_numpy(float),
        test.event_type.to_numpy(int),
        EVAL_TIMES,
        cause=1,
        metrics=("auc", "brier"),
    )
    auc = scores.auc.set_index("times").AUC
    brier = scores.brier.set_index("times").Brier
    risk60 = prediction[:, -1]
    dev_target = Surv.from_arrays(development.event_type.to_numpy(int) == 1, development.time_months.to_numpy(float))
    test_target = Surv.from_arrays(test.event_type.to_numpy(int) == 1, test.time_months.to_numpy(float))
    row = {
        "iAUC": trap_mean(EVAL_TIMES, scores.auc.AUC.to_numpy(float)),
        "IBS": trap_mean(EVAL_TIMES, scores.brier.Brier.to_numpy(float)),
        "harrell_c_cause_specific_60m": float(
            concordance_index_censored(test.event_type.to_numpy(int) == 1, test.time_months.to_numpy(float), risk60)[0]
        ),
        "uno_c_cause_specific_60m": float(concordance_index_ipcw(dev_target, test_target, risk60, tau=60.0)[0]),
    }
    for horizon in REPORT_TIMES:
        row[f"AUC_{int(horizon)}m"] = float(auc.loc[horizon])
        row[f"Brier_{int(horizon)}m"] = float(brier.loc[horizon])
    return row


def aggregate_and_fit_final(development: pd.DataFrame, test: pd.DataFrame) -> None:
    repeated = np.full((OUTER_REPEATS, len(development), len(EVAL_TIMES)), np.nan, dtype=np.float32)
    metadata = []
    for repeat in range(1, OUTER_REPEATS + 1):
        for fold in range(1, OUTER_FOLDS + 1):
            npz_path, json_path = checkpoint_paths(repeat, fold)
            with np.load(npz_path, allow_pickle=False) as saved:
                repeated[repeat - 1, saved["validation_index"].astype(int)] = saved["target_cif"]
            metadata.append(json.loads(json_path.read_text(encoding="utf-8")))
    if np.isnan(repeated).any():
        raise RuntimeError("Incomplete DeepHit OOF matrix")
    oof = repeated.mean(axis=0)
    oof_metrics = evaluate(development, development, oof)

    selected_ids = [row["selected_candidate"]["id"] for row in metadata]
    counts = Counter(selected_ids)
    outer_ibs_by_id = defaultdict(list)
    for row in metadata:
        outer_ibs_by_id[row["selected_candidate"]["id"]].append(row["outer_ibs"])
    selected_id = min(
        counts,
        key=lambda key: (-counts[key], float(np.mean(outer_ibs_by_id[key])), key),
    )
    selected = next(candidate for candidate in CANDIDATES if candidate["id"] == selected_id)

    preprocessor = make_preprocessor()
    x_dev = np.asarray(preprocessor.fit_transform(development[PREDICTOR_COLUMNS]), dtype=float)
    x_test = np.asarray(preprocessor.transform(test[PREDICTOR_COLUMNS]), dtype=float)
    test_prediction, final_log = fit_predict(
        x_dev,
        development,
        x_test,
        EVAL_TIMES,
        selected,
        SEED + 999999,
    )
    test_metrics = evaluate(development, test, test_prediction)
    original = pd.read_csv(
        ROOT / "output/models/model_development_v0.4/final_test_evaluation/table_test_main_comparison.csv"
    )
    old = original.loc[original.model.eq("deephit_cr")].iloc[0].to_dict()
    comparison = pd.DataFrame(
        [
            {"model_version": "Original low-budget DeepHit", **{k: old[k] for k in test_metrics}},
            {"model_version": "Retuned DeepHit with early stopping", **test_metrics},
        ]
    )
    comparison.to_csv(OUT_ROOT / "Table_DeepHit_performance_comparison.csv", index=False, encoding="utf-8-sig")

    fold_rows, candidate_rows = [], []
    for row in metadata:
        fold_rows.append(
            {
                "repeat": row["repeat"],
                "fold": row["fold"],
                "selected_candidate": row["selected_candidate"]["id"],
                "outer_ibs": row["outer_ibs"],
                "outer_epochs_completed": row["outer_fit_log"]["epochs_completed"],
                "outer_best_validation_loss": row["outer_fit_log"]["best_validation_loss"],
            }
        )
        for candidate in row["candidate_results"]:
            candidate_rows.append(
                {
                    "repeat": row["repeat"],
                    "fold": row["fold"],
                    "candidate": candidate["candidate"]["id"],
                    "mean_inner_ibs": candidate["mean_inner_ibs"],
                    "mean_epochs_completed": float(np.mean([x["epochs_completed"] for x in candidate["fit_logs"]])),
                    "seconds": candidate["seconds"],
                }
            )
    pd.DataFrame(fold_rows).to_csv(
        OUT_ROOT / "Table_DeepHit_outer_fold_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(candidate_rows).to_csv(
        OUT_ROOT / "Table_DeepHit_inner_candidate_results.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        OUT_ROOT / "deephit_retrained_predictions.npz",
        development_patient_id=development.patient_id.to_numpy(),
        development_oof_cif=oof,
        test_patient_id=test.patient_id.to_numpy(),
        test_cif=test_prediction.astype(np.float32),
        eval_times=EVAL_TIMES,
    )

    selected_inner_ibs = []
    for row in metadata:
        selected_inner_ibs.append(
            next(
                x["mean_inner_ibs"]
                for x in row["candidate_results"]
                if x["candidate"]["id"] == row["selected_candidate"]["id"]
            )
        )
    eligibility = {
        "mean_selected_inner_ibs": float(np.mean(selected_inner_ibs)),
        "inner_ibs_one_se_threshold": 0.036703154425060135,
        "development_oof_iAUC": float(oof_metrics["iAUC"]),
        "development_iAUC_threshold": 0.6449798403404313,
    }
    eligibility["eligible_for_ensemble"] = bool(
        eligibility["mean_selected_inner_ibs"] <= eligibility["inner_ibs_one_se_threshold"]
        and eligibility["development_oof_iAUC"] >= eligibility["development_iAUC_threshold"]
    )
    summary = {
        "status": "PASS",
        "signature": SIGNATURE,
        "candidates": CANDIDATES,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "selected_candidate_rule": "most frequent outer-fold selection; tie broken by mean outer IBS",
        "selection_counts": dict(counts),
        "selected_final_candidate": selected,
        "final_training_log": final_log,
        "development_oof_metrics": oof_metrics,
        "locked_test_metrics": test_metrics,
        "original_locked_test_metrics": {k: old[k] for k in test_metrics},
        "ensemble_eligibility": eligibility,
        "locked_test_used_for_parameter_selection": False,
    }
    (OUT_ROOT / "deephit_retraining_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-split-start", type=int, default=0)
    parser.add_argument("--max-outer-splits", type=int, default=15)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame.data_split.eq("development")].copy().reset_index(drop=True)
    test = frame.loc[frame.data_split.eq("test")].copy().reset_index(drop=True)
    if not args.aggregate_only:
        splitter = RepeatedStratifiedKFold(
            n_splits=OUTER_FOLDS,
            n_repeats=OUTER_REPEATS,
            random_state=SEED,
        )
        splits = list(splitter.split(np.zeros(len(development)), strata(development)))
        subset = splits[args.outer_split_start : args.outer_split_start + args.max_outer_splits]
        for local_index, (outer_train, outer_validation) in enumerate(subset):
            split_index = args.outer_split_start + local_index
            repeat = split_index // OUTER_FOLDS + 1
            fold = split_index % OUTER_FOLDS + 1
            run_fold(development, repeat, fold, outer_train, outer_validation)
    if (not args.skip_aggregate) and all(
        checkpoint_paths(r, f)[0].exists() for r in range(1, 4) for f in range(1, 6)
    ):
        aggregate_and_fit_final(development, test)
    else:
        print("DeepHit fold run complete; waiting for all 15 checkpoints before aggregation", flush=True)


if __name__ == "__main__":
    main()
