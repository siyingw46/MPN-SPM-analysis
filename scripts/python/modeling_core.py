"""Shared utilities for competing-risk prediction models.

The project fits one survival model per cause and combines their predicted
cause-specific cumulative hazards on a common time grid.  The combination is
the covariate-specific product-integral analogue of Aalen-Johansen.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv


CONTINUOUS_FEATURES = ["age_numeric", "diagnosis_year"]
BINARY_FEATURES = ["age_85plus"] + ([] if os.environ.get("MPN_OMIT_CHEMOTHERAPY") == "1" else ["chemotherapy"])
CATEGORICAL_FEATURES = [
    "sex",
    "race",
    "mpn_subtype",
    "rural_urban",
    "median_household_income",
]
PREDICTOR_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
CAUSE_LABELS = {1: "nonhematologic_spm" if os.environ.get("MPN_ANALYSIS_VERSION") == "v0.4" else "non_aml_spm", 2: "death", 3: "hematologic_malignancy" if os.environ.get("MPN_ANALYSIS_VERSION") == "v0.4" else "aml"}


@dataclass(frozen=True)
class CifPrediction:
    """CIF predictions and diagnostics from cause-specific models."""

    horizons: np.ndarray
    cif: dict[int, np.ndarray]
    survival: np.ndarray
    maximum_total_hazard_increment: float


def make_preprocessor() -> ColumnTransformer:
    """Create the locked development-only feature transformation."""

    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_cause_outcome(frame: pd.DataFrame, cause: int) -> np.ndarray:
    """Encode a cause-specific endpoint; competing events are censored."""

    return Surv.from_arrays(
        event=frame["event_type"].to_numpy(dtype=int) == cause,
        time=frame["time_months"].to_numpy(dtype=float),
    )


def make_pipeline(model: Any) -> Pipeline:
    """Build a leakage-safe preprocessing/model pipeline."""

    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def _cumulative_hazard_on_grid(model: Any, features: Any, grid: np.ndarray) -> np.ndarray:
    """Evaluate a fitted scikit-survival cumulative hazard on an arbitrary grid."""

    raw = np.asarray(model.predict_cumulative_hazard_function(features, return_array=True), dtype=float)
    model_times = np.asarray(model.unique_times_, dtype=float)
    positions = np.searchsorted(model_times, grid, side="right") - 1
    result = np.zeros((raw.shape[0], grid.size), dtype=float)
    valid = positions >= 0
    if np.any(valid):
        result[:, valid] = raw[:, positions[valid]]
    return result


def predict_cif_from_cause_specific_pipelines(
    pipelines: Mapping[int, Pipeline],
    frame: pd.DataFrame,
    horizons: Sequence[float],
) -> CifPrediction:
    """Combine cause-specific cumulative hazards into absolute CIFs.

    At each common event time, the previous all-cause survival is multiplied
    by each cause-specific hazard increment.  Survival is then updated by the
    product integral ``S(t)=S(t-)*(1-sum(dH_k(t)))``.  This makes the three
    CIFs mutually coherent rather than three unrelated binary probabilities.
    """

    requested_horizons = np.asarray(horizons, dtype=float)
    if requested_horizons.ndim != 1 or requested_horizons.size == 0:
        raise ValueError("horizons must be a non-empty one-dimensional sequence")
    if np.any(requested_horizons <= 0) or np.any(np.diff(requested_horizons) <= 0):
        raise ValueError("horizons must be strictly increasing and positive")
    if set(pipelines) != set(CAUSE_LABELS):
        raise ValueError(f"Expected fitted pipelines for causes {sorted(CAUSE_LABELS)}")

    maximum_horizon = float(requested_horizons.max())
    event_times = []
    transformed_by_cause: dict[int, Any] = {}
    model_by_cause: dict[int, Any] = {}
    for cause, pipeline in pipelines.items():
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed_by_cause[cause] = preprocessor.transform(frame[PREDICTOR_COLUMNS])
        model_by_cause[cause] = model
        times = np.asarray(model.unique_times_, dtype=float)
        event_times.append(times[(times > 0) & (times <= maximum_horizon)])

    grid = np.unique(np.concatenate([*event_times, requested_horizons]))
    cumulative_hazards = {}
    for cause in sorted(CAUSE_LABELS):
        cumulative_hazards[cause] = _cumulative_hazard_on_grid(
            model_by_cause[cause], transformed_by_cause[cause], grid
        )

    sample_count = len(frame)
    survival = np.ones(sample_count, dtype=float)
    cumulative_incidence = {cause: np.zeros(sample_count, dtype=float) for cause in CAUSE_LABELS}
    horizon_cif = {
        cause: np.zeros((sample_count, requested_horizons.size), dtype=float) for cause in CAUSE_LABELS
    }
    horizon_survival = np.zeros((sample_count, requested_horizons.size), dtype=float)
    previous_hazard = {cause: np.zeros(sample_count, dtype=float) for cause in CAUSE_LABELS}
    horizon_lookup = {float(value): index for index, value in enumerate(requested_horizons)}
    maximum_total_increment = 0.0

    for grid_index, time_value in enumerate(grid):
        increments = {}
        for cause in sorted(CAUSE_LABELS):
            current = cumulative_hazards[cause][:, grid_index]
            increment = np.maximum(current - previous_hazard[cause], 0.0)
            increments[cause] = increment
            previous_hazard[cause] = current
        total_increment = sum(increments.values())
        maximum_total_increment = max(maximum_total_increment, float(np.max(total_increment)))
        if np.any(total_increment > 1.0 + 1e-10):
            raise ValueError(
                "A predicted all-cause hazard increment exceeded 1; "
                "the Aalen-Johansen product integral is invalid for this fit."
            )
        for cause in sorted(CAUSE_LABELS):
            cumulative_incidence[cause] += survival * increments[cause]
        survival *= np.maximum(1.0 - total_increment, 0.0)

        horizon_index = horizon_lookup.get(float(time_value))
        if horizon_index is not None:
            for cause in sorted(CAUSE_LABELS):
                horizon_cif[cause][:, horizon_index] = cumulative_incidence[cause]
            horizon_survival[:, horizon_index] = survival

    return CifPrediction(
        horizons=requested_horizons,
        cif=horizon_cif,
        survival=horizon_survival,
        maximum_total_hazard_increment=maximum_total_increment,
    )


def validate_cif_prediction(prediction: CifPrediction, tolerance: float = 1e-8) -> dict[str, Any]:
    """Return machine-readable probability and monotonicity checks."""

    stacked = np.stack([prediction.cif[cause] for cause in sorted(CAUSE_LABELS)], axis=0)
    total_cif = stacked.sum(axis=0)
    minimum_probability = float(min(stacked.min(), prediction.survival.min()))
    maximum_probability = float(max(stacked.max(), prediction.survival.max()))
    monotonic_violations = {
        CAUSE_LABELS[cause]: int(np.sum(np.any(np.diff(prediction.cif[cause], axis=1) < -tolerance, axis=1)))
        for cause in sorted(CAUSE_LABELS)
    }
    coherence_error = np.abs(total_cif + prediction.survival - 1.0)
    result = {
        "status": "PASS",
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "maximum_sum_of_cifs": float(total_cif.max()),
        "sum_of_cifs_above_one_count": int(np.sum(np.any(total_cif > 1.0 + tolerance, axis=1))),
        "monotonicity_violation_count_by_cause": monotonic_violations,
        "maximum_probability_coherence_error": float(coherence_error.max()),
        "maximum_total_hazard_increment": prediction.maximum_total_hazard_increment,
    }
    if (
        minimum_probability < -tolerance
        or maximum_probability > 1.0 + tolerance
        or result["sum_of_cifs_above_one_count"] > 0
        or any(monotonic_violations.values())
        or result["maximum_probability_coherence_error"] > tolerance
    ):
        result["status"] = "FAIL"
    return result
