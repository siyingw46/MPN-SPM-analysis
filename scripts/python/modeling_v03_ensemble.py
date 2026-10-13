"""Leakage-safe ensemble utilities for competing-risk CIF predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


def censoring_estimator(time: np.ndarray, event: np.ndarray) -> CensoringDistributionEstimator:
    outcome = Surv.from_arrays(event=np.asarray(event) != 0, time=np.asarray(time, dtype=float))
    return CensoringDistributionEstimator().fit(outcome)


def ipcw_binary_at_horizon(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    censoring: CensoringDistributionEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    before = time <= horizon
    known = ~(before & (event == 0))
    target = (before & (event == 1)).astype(float)
    weights = np.zeros(len(time), dtype=float)
    observed_before = before & (event != 0)
    if np.any(observed_before):
        query = np.nextafter(time[observed_before], -np.inf)
        weights[observed_before] = 1.0 / np.maximum(
            censoring.predict_proba(query), 1e-6
        )
    after = time > horizon
    if np.any(after):
        g_horizon = max(float(censoring.predict_proba(np.asarray([horizon]))[0]), 1e-6)
        weights[after] = 1.0 / g_horizon
    return known, target, weights


def quadratic_terms(
    prediction_stack: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizons: Sequence[float],
    patient_mask: np.ndarray | None = None,
    include_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized X'WX and X'Wy for pooled IPCW Brier loss."""

    prediction_stack = np.asarray(prediction_stack, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    horizons = np.asarray(horizons, dtype=float)
    n, n_times, n_models = prediction_stack.shape
    if n != len(time) or n_times != len(horizons):
        raise ValueError("Prediction stack and outcome/time dimensions do not agree")
    if patient_mask is None:
        patient_mask = np.ones(n, dtype=bool)
    else:
        patient_mask = np.asarray(patient_mask, dtype=bool)
    width = n_models + int(include_intercept)
    gram = np.zeros((width, width), dtype=float)
    target_cross = np.zeros(width, dtype=float)
    weight_sum = 0.0
    censoring = censoring_estimator(time[patient_mask], event[patient_mask])

    for position, horizon in enumerate(horizons):
        known_sub, target_sub, weights_sub = ipcw_binary_at_horizon(
            time[patient_mask], event[patient_mask], float(horizon), censoring
        )
        base = prediction_stack[patient_mask, position, :]
        if include_intercept:
            design = np.column_stack([np.ones(len(base)), base])
        else:
            design = base
        selected = known_sub & (weights_sub > 0.0)
        x = design[selected]
        y = target_sub[selected]
        w = weights_sub[selected]
        gram += x.T @ (w[:, None] * x)
        target_cross += x.T @ (w * y)
        weight_sum += float(np.sum(w))
    if weight_sum <= 0.0:
        raise RuntimeError("No positive IPCW weight available for ensemble fitting")
    return gram / weight_sum, target_cross / weight_sum, weight_sum


def quadratic_loss(
    coefficients: np.ndarray,
    prediction_stack: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizons: Sequence[float],
    patient_mask: np.ndarray | None = None,
    include_intercept: bool = False,
) -> float:
    gram, target_cross, _ = quadratic_terms(
        prediction_stack,
        time,
        event,
        horizons,
        patient_mask=patient_mask,
        include_intercept=include_intercept,
    )
    coefficients = np.asarray(coefficients, dtype=float)
    return float(coefficients @ gram @ coefficients - 2.0 * coefficients @ target_cross)


def fit_nonnegative_simplex_stack(
    prediction_stack: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizons: Sequence[float],
) -> tuple[np.ndarray, dict]:
    gram, target_cross, _ = quadratic_terms(
        prediction_stack, time, event, horizons, include_intercept=False
    )
    n_models = prediction_stack.shape[2]

    def objective(weights: np.ndarray) -> float:
        return float(weights @ gram @ weights - 2.0 * weights @ target_cross)

    optimization = minimize(
        objective,
        x0=np.repeat(1.0 / n_models, n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not optimization.success:
        raise RuntimeError(f"Nonnegative stacking failed: {optimization.message}")
    weights = np.maximum(np.asarray(optimization.x, dtype=float), 0.0)
    weights /= weights.sum()
    return weights, {
        "success": bool(optimization.success),
        "message": str(optimization.message),
        "iterations": int(optimization.nit),
        "objective": float(optimization.fun),
    }


def fit_ridge_meta(
    prediction_stack: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizons: Sequence[float],
    fold_assignment: np.ndarray,
    alphas: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
) -> tuple[np.ndarray, float, list[dict]]:
    """Select and fit an IPCW-Brier ridge meta-model with an intercept."""

    fold_assignment = np.asarray(fold_assignment, dtype=int)
    n_models = prediction_stack.shape[2]
    penalty_mask = np.diag(np.r_[0.0, np.ones(n_models)])
    cv_rows = []
    for alpha in alphas:
        losses = []
        for fold in sorted(np.unique(fold_assignment)):
            train_mask = fold_assignment != fold
            validation_mask = fold_assignment == fold
            gram, target_cross, _ = quadratic_terms(
                prediction_stack,
                time,
                event,
                horizons,
                patient_mask=train_mask,
                include_intercept=True,
            )
            coefficients = np.linalg.pinv(gram + float(alpha) * penalty_mask) @ target_cross
            losses.append(
                quadratic_loss(
                    coefficients,
                    prediction_stack,
                    time,
                    event,
                    horizons,
                    patient_mask=validation_mask,
                    include_intercept=True,
                )
            )
        cv_rows.append(
            {
                "alpha": float(alpha),
                "fold_losses_without_constant": [float(value) for value in losses],
                "mean_loss_without_constant": float(np.mean(losses)),
            }
        )
    best = min(cv_rows, key=lambda row: row["mean_loss_without_constant"])
    selected_alpha = float(best["alpha"])
    gram, target_cross, _ = quadratic_terms(
        prediction_stack, time, event, horizons, include_intercept=True
    )
    coefficients = np.linalg.pinv(gram + selected_alpha * penalty_mask) @ target_cross
    return np.asarray(coefficients, dtype=float), selected_alpha, cv_rows


def weighted_average(prediction_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ntk,k->nt", np.asarray(prediction_stack), np.asarray(weights))


def predict_ridge_meta(prediction_stack: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    raw = coefficients[0] + np.einsum(
        "ntk,k->nt", np.asarray(prediction_stack), coefficients[1:]
    )
    clipped = np.clip(raw, 0.0, 1.0)
    return np.maximum.accumulate(clipped, axis=1)


def validate_target_cif(prediction: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    finite = bool(np.isfinite(prediction).all())
    minimum = float(np.nanmin(prediction))
    maximum = float(np.nanmax(prediction))
    violations = int(np.sum(np.diff(prediction, axis=1) < -1e-6))
    status = (
        "PASS"
        if finite and minimum >= -1e-6 and maximum <= 1.0 + 1e-6 and violations == 0
        else "FAIL"
    )
    return {
        "status": status,
        "finite": finite,
        "minimum_probability": minimum,
        "maximum_probability": maximum,
        "monotonicity_violation_count": violations,
    }
