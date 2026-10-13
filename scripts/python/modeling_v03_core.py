"""Model adapters for the expanded v0.3 competing-risk model registry."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from scipy.linalg import qr
from sksurv.linear_model import CoxPHSurvivalAnalysis

from modeling_core import CAUSE_LABELS, CifPrediction


class FineGrayCifEstimator(BaseEstimator):
    """Fine-Gray estimator with fold-specific rank-deficiency protection."""

    def __init__(self, cause: int = 1, max_iter: int = 50, fallback_ridge: float = 1e-6):
        self.cause = cause
        self.max_iter = max_iter
        self.fallback_ridge = fallback_ridge

    def fit(self, features: Any, time: np.ndarray, event: np.ndarray):
        from comprisk import FineGrayRegression, PenalizedFineGrayRegression

        x = np.asarray(features, dtype=float)
        _, upper, pivots = qr(x, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(upper))
        tolerance = np.finfo(float).eps * max(x.shape) * (diagonal.max() if diagonal.size else 0.0)
        rank = int(np.sum(diagonal > tolerance))
        self.selected_columns_ = np.sort(np.asarray(pivots[:rank], dtype=int))
        if self.selected_columns_.size == 0:
            raise ValueError("Fine-Gray design matrix has zero numerical rank")
        selected = x[:, self.selected_columns_]
        self.fit_method_ = "unpenalized_qr_full_rank"
        try:
            self.model_ = FineGrayRegression(cause=int(self.cause), max_iter=int(self.max_iter))
            self.model_.fit(selected, time=np.asarray(time, dtype=float), event=np.asarray(event, dtype=int))
        except RuntimeError as error:
            if "singular" not in str(error).lower():
                raise
            self.fit_method_ = "ridge_fallback_after_qr"
            self.model_ = PenalizedFineGrayRegression(
                penalty="ridge",
                lambdas=[float(self.fallback_ridge)],
                standardize=True,
                cause=int(self.cause),
                max_iter=1000,
                tol=1e-5,
            )
            self.model_.fit(selected, time=np.asarray(time, dtype=float), event=np.asarray(event, dtype=int))
        return self

    def predict_cumulative_incidence(self, features: Any, times: Sequence[float]) -> np.ndarray:
        selected = np.asarray(features, dtype=float)[:, self.selected_columns_]
        return np.asarray(self.model_.predict_cumulative_incidence(selected, times=np.asarray(times)), dtype=float)


class RiskScoreCoxCalibratedModel(BaseEstimator):
    """Convert a survival risk-ranking estimator into a cumulative-hazard model.

    The base estimator is fitted on a cause-specific endpoint. Its training risk
    score is standardized and entered into a one-variable Cox model, which
    supplies the fold-specific baseline cumulative hazard. This adapter is used
    for models such as survival SVM and XGBoost-Cox that otherwise provide only
    a risk score rather than a calibrated probability curve.
    """

    def __init__(self, risk_estimator: Any, calibration_alpha: float = 1e-4):
        self.risk_estimator = risk_estimator
        self.calibration_alpha = calibration_alpha

    def fit(self, features: Any, outcome: np.ndarray):
        self.risk_estimator_ = clone(self.risk_estimator)
        self.risk_estimator_.fit(features, outcome)
        training_score = np.asarray(self.risk_estimator_.predict(features), dtype=float).reshape(-1)
        self.score_mean_ = float(np.mean(training_score))
        self.score_scale_ = float(np.std(training_score))
        if not np.isfinite(self.score_scale_) or self.score_scale_ < 1e-12:
            self.score_scale_ = 1.0
        standardized = self._standardize(training_score)
        self.calibrator_ = CoxPHSurvivalAnalysis(
            alpha=float(self.calibration_alpha), ties="breslow", n_iter=200
        )
        self.calibrator_.fit(standardized[:, None], outcome)
        self.unique_times_ = np.asarray(self.calibrator_.unique_times_, dtype=float)
        return self

    def _standardize(self, score: np.ndarray) -> np.ndarray:
        return (np.asarray(score, dtype=float) - self.score_mean_) / self.score_scale_

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(self.risk_estimator_.predict(features), dtype=float).reshape(-1)

    def predict_cumulative_hazard_function(self, features: Any, return_array: bool = False):
        score = self._standardize(self.predict(features))
        return self.calibrator_.predict_cumulative_hazard_function(
            score[:, None], return_array=return_array
        )

    def predict_survival_function(self, features: Any, return_array: bool = False):
        score = self._standardize(self.predict(features))
        return self.calibrator_.predict_survival_function(score[:, None], return_array=return_array)


class XGBoostCoxRiskEstimator(BaseEstimator):
    """Scikit-learn compatible XGBoost ``survival:cox`` risk estimator."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_child_weight: float = 5.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 20260718,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, features: Any, outcome: np.ndarray):
        from xgboost import XGBRegressor

        event_field, time_field = outcome.dtype.names
        time = np.asarray(outcome[time_field], dtype=float)
        event = np.asarray(outcome[event_field], dtype=bool)
        signed_time = np.where(event, time, -time)
        self.model_ = XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            tree_method="hist",
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_child_weight=float(self.min_child_weight),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            reg_alpha=float(self.reg_alpha),
            reg_lambda=float(self.reg_lambda),
            n_jobs=int(self.n_jobs),
            random_state=int(self.random_state),
        )
        self.model_.fit(features, signed_time, verbose=False)
        return self

    def predict(self, features: Any) -> np.ndarray:
        # XGBoost returns exp(margin) for survival:cox.  Log restores the
        # unconstrained risk score used by the fold-specific Cox calibrator.
        hazard_ratio = np.asarray(self.model_.predict(features), dtype=float)
        return np.log(np.clip(hazard_ratio, 1e-12, 1e12))


def _cumulative_hazard_on_grid(model: Any, features: Any, grid: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_cumulative_hazard_function(features, return_array=True), dtype=float)
    model_times = np.asarray(model.unique_times_, dtype=float)
    positions = np.searchsorted(model_times, grid, side="right") - 1
    result = np.zeros((raw.shape[0], grid.size), dtype=float)
    valid = positions >= 0
    if np.any(valid):
        result[:, valid] = raw[:, positions[valid]]
    return result


def predict_cif_from_cause_models(
    models: Mapping[int, Any], features: Any, horizons: Sequence[float]
) -> CifPrediction:
    """Combine three fitted cause-specific models into coherent CIFs."""

    requested_horizons = np.asarray(horizons, dtype=float)
    if set(models) != set(CAUSE_LABELS):
        raise ValueError(f"Expected models for causes {sorted(CAUSE_LABELS)}")
    if np.any(requested_horizons <= 0) or np.any(np.diff(requested_horizons) <= 0):
        raise ValueError("horizons must be positive and strictly increasing")
    maximum_horizon = float(requested_horizons.max())
    grid = np.unique(
        np.concatenate(
            [
                *[
                    np.asarray(model.unique_times_, dtype=float)[
                        (np.asarray(model.unique_times_, dtype=float) > 0)
                        & (np.asarray(model.unique_times_, dtype=float) <= maximum_horizon)
                    ]
                    for model in models.values()
                ],
                requested_horizons,
            ]
        )
    )
    cumulative_hazards = {
        cause: _cumulative_hazard_on_grid(model, features, grid) for cause, model in models.items()
    }
    sample_count = cumulative_hazards[1].shape[0]
    survival = np.ones(sample_count, dtype=float)
    cif = {cause: np.zeros(sample_count, dtype=float) for cause in CAUSE_LABELS}
    horizon_cif = {
        cause: np.zeros((sample_count, requested_horizons.size), dtype=float) for cause in CAUSE_LABELS
    }
    horizon_survival = np.zeros((sample_count, requested_horizons.size), dtype=float)
    previous = {cause: np.zeros(sample_count, dtype=float) for cause in CAUSE_LABELS}
    lookup = {float(value): index for index, value in enumerate(requested_horizons)}
    maximum_increment = 0.0
    for time_index, time_value in enumerate(grid):
        increments = {}
        for cause in CAUSE_LABELS:
            current = cumulative_hazards[cause][:, time_index]
            increments[cause] = np.maximum(current - previous[cause], 0.0)
            previous[cause] = current
        total_increment = sum(increments.values())
        maximum_increment = max(maximum_increment, float(np.max(total_increment)))
        if np.any(total_increment > 1.0 + 1e-10):
            raise ValueError("Predicted all-cause hazard increment exceeded one")
        for cause in CAUSE_LABELS:
            cif[cause] += survival * increments[cause]
        survival *= np.maximum(1.0 - total_increment, 0.0)
        horizon_index = lookup.get(float(time_value))
        if horizon_index is not None:
            for cause in CAUSE_LABELS:
                horizon_cif[cause][:, horizon_index] = cif[cause]
            horizon_survival[:, horizon_index] = survival
    return CifPrediction(
        horizons=requested_horizons,
        cif=horizon_cif,
        survival=horizon_survival,
        maximum_total_hazard_increment=maximum_increment,
    )


class ReshapeDeepHitOutput:
    """Factory for a torch module that reshapes logits by risk and duration."""

    @staticmethod
    def make(num_risks: int, num_durations: int):
        import torch

        class _Reshape(torch.nn.Module):
            def forward(self, values):
                return values.view(-1, num_risks, num_durations)

        return _Reshape()


class DeepHitCompetingRiskEstimator:
    """Small reproducible wrapper around pycox's multi-risk DeepHit model."""

    def __init__(
        self,
        num_durations: int = 24,
        hidden_nodes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        alpha: float = 0.2,
        sigma: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        random_state: int = 20260718,
    ):
        self.num_durations = num_durations
        self.hidden_nodes = hidden_nodes
        self.dropout = dropout
        self.alpha = alpha
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, features: np.ndarray, time: np.ndarray, event: np.ndarray):
        import torch
        import torchtuples as tt
        from pycox.models import DeepHit
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        x = np.asarray(features, dtype=np.float32)
        time = np.asarray(time, dtype=np.float32)
        event = np.asarray(event, dtype=np.int64)
        self.labtrans_ = LabTransDiscreteTime(int(self.num_durations))
        duration_index, transformed_event = self.labtrans_.fit_transform(time, event)
        target = (duration_index, transformed_event.astype(np.int64))
        num_risks = int(np.max(event))
        self.num_risks_ = num_risks
        base = tt.practical.MLPVanilla(
            x.shape[1],
            list(self.hidden_nodes),
            num_risks * int(self.num_durations),
            batch_norm=True,
            dropout=float(self.dropout),
        )
        network = torch.nn.Sequential(
            base, ReshapeDeepHitOutput.make(num_risks, int(self.num_durations))
        )
        self.model_ = DeepHit(
            network,
            tt.optim.Adam,
            alpha=float(self.alpha),
            sigma=float(self.sigma),
            duration_index=self.labtrans_.cuts,
        )
        self.model_.optimizer.set_lr(float(self.learning_rate))
        self.training_log_ = self.model_.fit(
            x,
            target,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            verbose=False,
        )
        return self

    def predict_all_cause_cif(
        self, features: np.ndarray, horizons: Sequence[float]
    ) -> dict[int, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        raw = np.asarray(self.model_.predict_cif(x, numpy=True, to_cpu=True), dtype=float)
        cuts = np.asarray(self.labtrans_.cuts, dtype=float)
        horizons_array = np.asarray(horizons, dtype=float)
        positions = np.searchsorted(cuts, horizons_array, side="right") - 1
        positions = np.clip(positions, 0, len(cuts) - 1)
        # pycox returns [risk, duration, patient].
        return {
            cause: raw[cause - 1, positions, :].T for cause in range(1, self.num_risks_ + 1)
        }
