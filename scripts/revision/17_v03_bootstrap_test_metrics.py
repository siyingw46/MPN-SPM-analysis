"""Patient-level bootstrap inference for the locked v0.3 test predictions.

The models and their roles were frozen before test evaluation.  This script
does not refit or tune any model.  It resamples test patients, recomputes the
competing-risk metrics (including the censoring distribution), and uses the
same bootstrap samples for all paired primary-model comparisons.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from comprisk.evaluation import _km_censoring_cr, _per_time_auc_brier


from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
INPUT_DIR = MODEL_ROOT / "final_test_evaluation"
PREDICTION_PATH = INPUT_DIR / "test_all_models_cif.npz"
SEED = int(os.environ.get("MPN_RANDOM_SEED", "20260723"))
N_BOOTSTRAP = int(os.environ.get("MPN_BOOTSTRAP_REPLICATES", "5000"))
BOOTSTRAP_TAG = f"bootstrap{N_BOOTSTRAP}"
CONFIDENCE_LEVEL = 0.95
CAUSE = 1
PRIMARY = "xgboost_cox"
REPORT_TIMES = (12.0, 36.0, 60.0)
BOOTSTRAP_BATCH = int(os.environ.get("MPN_BOOTSTRAP_BATCH", "25"))
N_JOBS = int(os.environ.get("MPN_BOOTSTRAP_N_JOBS", "-1"))
INCLUDE_C_INDICES = os.environ.get("MPN_BOOTSTRAP_INCLUDE_C_INDICES", "0") == "1"


def trap_mean(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapezoid(values[valid], times[valid]) / (times[valid][-1] - times[valid][0]))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust a one-dimensional collection while retaining input order."""
    result = np.full(len(p_values), np.nan, dtype=float)
    valid = np.where(np.isfinite(p_values))[0]
    if not len(valid):
        return result
    order = valid[np.argsort(p_values[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(p_values[index]))
        result[index] = min(running, 1.0)
    return result


def bootstrap_p_value(samples: np.ndarray) -> float:
    """Two-sided sign-based bootstrap P value with a finite-sample correction."""
    values = samples[np.isfinite(samples)]
    if not len(values):
        return float("nan")
    nonpositive = (np.count_nonzero(values <= 0.0) + 1.0) / (len(values) + 1.0)
    nonnegative = (np.count_nonzero(values >= 0.0) + 1.0) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(nonpositive, nonnegative)))


def one_bootstrap(
    seed: int,
    predictions: list[np.ndarray],
    test_time: np.ndarray,
    test_event: np.ndarray,
    eval_times: np.ndarray,
    development_target: np.ndarray,
    include_c_indices: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(test_time)
    index = rng.integers(0, n, size=n)
    time_b = test_time[index]
    event_b = test_event[index]
    censoring_time, censoring_survival = _km_censoring_cr(time_b, event_b)
    scores = np.full((len(predictions), len(eval_times), 2), np.nan, dtype=float)
    c_indices = np.full((len(predictions), 2), np.nan, dtype=float)
    target_b = Surv.from_arrays(event_b == CAUSE, time_b) if include_c_indices else None
    for model_index, prediction in enumerate(predictions):
        prediction_b = prediction[index]
        auc, brier = _per_time_auc_brier(
            prediction_b,
            time_b,
            event_b,
            eval_times,
            censoring_time,
            censoring_survival,
            CAUSE,
        )
        scores[model_index, :, 0] = auc
        scores[model_index, :, 1] = brier
        if include_c_indices:
            risk_60 = prediction_b[:, -1]
            c_indices[model_index, 0] = concordance_index_censored(
                event_b == CAUSE, time_b, risk_60
            )[0]
            c_indices[model_index, 1] = concordance_index_ipcw(
                development_target, target_b, risk_60, tau=60.0
            )[0]
    return scores, c_indices


def percentile_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - CONFIDENCE_LEVEL
    with np.errstate(invalid="ignore"):
        lower = np.nanpercentile(values, 100.0 * alpha / 2.0, axis=0)
        upper = np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper


def main() -> None:
    started = time.perf_counter()
    with np.load(PREDICTION_PATH, allow_pickle=False) as saved:
        eval_times = saved["eval_times"].astype(float)
        test_time = saved["time_months"].astype(float)
        test_event = saved["event_type"].astype(int)
        metadata_keys = {"patient_id", "time_months", "event_type", "eval_times"}
        model_names = [key for key in saved.files if key not in metadata_keys]
        prediction_arrays = [saved[model].astype(float) for model in model_names]

    if PRIMARY not in model_names:
        raise RuntimeError(f"Locked primary model {PRIMARY!r} is absent from predictions")
    if not np.array_equal(eval_times, np.arange(12.0, 61.0, 1.0)):
        raise RuntimeError("Unexpected evaluation grid")

    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    development = frame.loc[frame["data_split"].eq("development")]
    development_target = Surv.from_arrays(
        development["event_type"].to_numpy(dtype=int) == CAUSE,
        development["time_months"].to_numpy(dtype=float),
    )
    del frame, development

    if BOOTSTRAP_BATCH < 1:
        raise ValueError("MPN_BOOTSTRAP_BATCH must be positive")
    checkpoint_path = INPUT_DIR / f"test_{BOOTSTRAP_TAG}_checkpoint.npz"
    seeds = np.random.SeedSequence(SEED).generate_state(N_BOOTSTRAP)
    score_shape = (N_BOOTSTRAP, len(model_names), len(eval_times), 2)
    c_shape = (N_BOOTSTRAP, len(model_names), 2)
    completed = np.zeros(N_BOOTSTRAP, dtype=bool)
    bootstrap_scores = np.full(score_shape, np.nan, dtype=float)
    bootstrap_c = np.full(c_shape, np.nan, dtype=float)
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            compatible = (
                np.array_equal(checkpoint["model_names"], np.asarray(model_names))
                and np.array_equal(checkpoint["eval_times"], eval_times)
                and int(checkpoint["seed"]) == SEED
                and checkpoint["scores"].shape == score_shape
                and checkpoint["c_indices"].shape == c_shape
                and "include_c_indices" in checkpoint.files
                and bool(int(checkpoint["include_c_indices"])) == INCLUDE_C_INDICES
            )
            if compatible:
                completed = checkpoint["completed"].astype(bool)
                bootstrap_scores = checkpoint["scores"].astype(float)
                bootstrap_c = checkpoint["c_indices"].astype(float)
                print(f"Bootstrap checkpoint reused: {int(completed.sum())}/{N_BOOTSTRAP}", flush=True)
            else:
                print("Bootstrap checkpoint incompatible; starting a new deterministic run", flush=True)

    pending = np.flatnonzero(~completed)
    for batch_start in range(0, len(pending), BOOTSTRAP_BATCH):
        batch_indices = pending[batch_start : batch_start + BOOTSTRAP_BATCH]
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(one_bootstrap)(
                int(seeds[index]),
                prediction_arrays,
                test_time,
                test_event,
                eval_times,
                development_target,
                INCLUDE_C_INDICES,
            )
            for index in batch_indices
        )
        for index, result in zip(batch_indices, results):
            bootstrap_scores[index] = result[0]
            bootstrap_c[index] = result[1]
            completed[index] = True
        del results
        np.savez_compressed(
            checkpoint_path,
            model_names=np.asarray(model_names),
            eval_times=eval_times,
            seed=np.asarray(SEED),
            completed=completed,
            scores=bootstrap_scores,
            c_indices=bootstrap_c,
            include_c_indices=np.asarray(int(INCLUDE_C_INDICES)),
        )
        print(f"Bootstrap checkpoint saved: {int(completed.sum())}/{N_BOOTSTRAP}", flush=True)

    point_main = pd.read_csv(INPUT_DIR / "table_test_main_comparison.csv")
    point_main = point_main.set_index("model")
    auc_lower, auc_upper = percentile_interval(bootstrap_scores[..., 0])
    brier_lower, brier_upper = percentile_interval(bootstrap_scores[..., 1])
    c_lower, c_upper = percentile_interval(bootstrap_c)
    iauc_bootstrap = np.asarray(
        [
            [trap_mean(eval_times, bootstrap_scores[b, m, :, 0]) for m in range(len(model_names))]
            for b in range(N_BOOTSTRAP)
        ],
        dtype=float,
    )
    ibs_bootstrap = np.asarray(
        [
            [trap_mean(eval_times, bootstrap_scores[b, m, :, 1]) for m in range(len(model_names))]
            for b in range(N_BOOTSTRAP)
        ],
        dtype=float,
    )
    iauc_lower, iauc_upper = percentile_interval(iauc_bootstrap)
    ibs_lower, ibs_upper = percentile_interval(ibs_bootstrap)

    auc_rows: list[dict] = []
    brier_rows: list[dict] = []
    c_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model_index, model in enumerate(model_names):
        for time_index, horizon in enumerate(eval_times):
            auc_rows.append(
                {
                    "model": model,
                    "times": float(horizon),
                    "AUC": float(point_main.loc[model, f"AUC_{int(horizon)}m"])
                    if horizon in REPORT_TIMES
                    else float("nan"),
                    "lower": float(auc_lower[model_index, time_index]),
                    "upper": float(auc_upper[model_index, time_index]),
                }
            )
            brier_rows.append(
                {
                    "model": model,
                    "times": float(horizon),
                    "Brier": float(point_main.loc[model, f"Brier_{int(horizon)}m"])
                    if horizon in REPORT_TIMES
                    else float("nan"),
                    "lower": float(brier_lower[model_index, time_index]),
                    "upper": float(brier_upper[model_index, time_index]),
                }
            )
        c_rows.extend(
            [
                {
                    "model": model,
                    "metric": "harrell_c_cause_specific_60m",
                    "estimate": float(point_main.loc[model, "harrell_c_cause_specific_60m"]),
                    "lower": float(c_lower[model_index, 0]),
                    "upper": float(c_upper[model_index, 0]),
                },
                {
                    "model": model,
                    "metric": "uno_c_cause_specific_60m",
                    "estimate": float(point_main.loc[model, "uno_c_cause_specific_60m"]),
                    "lower": float(c_lower[model_index, 1]),
                    "upper": float(c_upper[model_index, 1]),
                },
            ]
        )
        row = {
            "model": model,
            "development_role_locked_before_test": point_main.loc[
                model, "development_role_locked_before_test"
            ],
            "iAUC": float(point_main.loc[model, "iAUC"]),
            "iAUC_lower": float(iauc_lower[model_index]),
            "iAUC_upper": float(iauc_upper[model_index]),
            "IBS": float(point_main.loc[model, "IBS"]),
            "IBS_lower": float(ibs_lower[model_index]),
            "IBS_upper": float(ibs_upper[model_index]),
        }
        for horizon in REPORT_TIMES:
            time_index = int(np.where(np.isclose(eval_times, horizon))[0][0])
            row.update(
                {
                    f"AUC_{int(horizon)}m": float(point_main.loc[model, f"AUC_{int(horizon)}m"]),
                    f"AUC_{int(horizon)}m_lower": float(auc_lower[model_index, time_index]),
                    f"AUC_{int(horizon)}m_upper": float(auc_upper[model_index, time_index]),
                    f"Brier_{int(horizon)}m": float(point_main.loc[model, f"Brier_{int(horizon)}m"]),
                    f"Brier_{int(horizon)}m_lower": float(brier_lower[model_index, time_index]),
                    f"Brier_{int(horizon)}m_upper": float(brier_upper[model_index, time_index]),
                }
            )
        summary_rows.append(row)

    paired_rows: list[dict] = []
    primary_index = model_names.index(PRIMARY)
    comparison_metrics = [
        ("iAUC", None, iauc_bootstrap, "higher_is_better"),
        ("IBS", None, ibs_bootstrap, "lower_is_better"),
    ]
    if INCLUDE_C_INDICES:
        comparison_metrics.extend(
            [
                ("Harrell_C_60m", None, bootstrap_c[:, :, 0], "higher_is_better"),
                ("Uno_C_60m", None, bootstrap_c[:, :, 1], "higher_is_better"),
            ]
        )
    for horizon in REPORT_TIMES:
        time_index = int(np.where(np.isclose(eval_times, horizon))[0][0])
        comparison_metrics.extend(
            [
                (f"AUC_{int(horizon)}m", horizon, bootstrap_scores[:, :, time_index, 0], "higher_is_better"),
                (f"Brier_{int(horizon)}m", horizon, bootstrap_scores[:, :, time_index, 1], "lower_is_better"),
            ]
        )
    for metric, horizon, samples, direction in comparison_metrics:
        metric_rows = []
        for comparator_index, comparator in enumerate(model_names):
            if comparator == PRIMARY:
                continue
            difference = samples[:, primary_index] - samples[:, comparator_index]
            lower, upper = percentile_interval(difference)
            metric_rows.append(
                {
                    "primary_model": PRIMARY,
                    "comparator_model": comparator,
                    "metric": metric,
                    "times": horizon,
                    "difference_definition": "primary_minus_comparator",
                    "direction": direction,
                    "estimate": float(np.nanmean(difference)),
                    "lower": float(lower),
                    "upper": float(upper),
                    "p_bootstrap_unadjusted": bootstrap_p_value(difference),
                }
            )
        adjusted = holm_adjust(
            np.asarray([row["p_bootstrap_unadjusted"] for row in metric_rows], dtype=float)
        )
        for row, p_adjusted in zip(metric_rows, adjusted):
            row["p_holm_within_metric"] = float(p_adjusted)
            paired_rows.append(row)

    pd.DataFrame(auc_rows).to_csv(
        INPUT_DIR / f"test_auc_{BOOTSTRAP_TAG}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(brier_rows).to_csv(
        INPUT_DIR / f"test_brier_{BOOTSTRAP_TAG}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(c_rows).to_csv(
        INPUT_DIR / f"test_c_indices_{BOOTSTRAP_TAG}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).sort_values("iAUC", ascending=False).to_csv(
        INPUT_DIR / f"table_test_main_comparison_{BOOTSTRAP_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(paired_rows).to_csv(
        INPUT_DIR / f"test_primary_paired_differences_{BOOTSTRAP_TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        INPUT_DIR / f"test_{BOOTSTRAP_TAG}_replicates.npz",
        model_names=np.asarray(model_names),
        eval_times=eval_times,
        auc=bootstrap_scores[..., 0],
        brier=bootstrap_scores[..., 1],
        harrell_c=bootstrap_c[:, :, 0],
        uno_c=bootstrap_c[:, :, 1],
        iauc=iauc_bootstrap,
        ibs=ibs_bootstrap,
    )
    summary = {
        "status": "PASS",
        "bootstrap_replicates": N_BOOTSTRAP,
        "confidence_level": CONFIDENCE_LEVEL,
        "seed": SEED,
        "test_rows": int(len(test_time)),
        "models": model_names,
        "primary_model_locked_before_test": PRIMARY,
        "paired_difference_definition": "primary minus comparator; Holm adjustment within each metric/time point",
        "c_indices_bootstrapped": INCLUDE_C_INDICES,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (INPUT_DIR / f"test_{BOOTSTRAP_TAG}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
