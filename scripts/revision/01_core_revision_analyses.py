"""Reviewer-requested event sensitivity, explainability, and exclusion analyses.

Outputs are written to ``revision`` without changing the
locked v0.4 data or the V2 manuscript.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["MPN_ANALYSIS_VERSION"] = "v0.4"

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from comprisk import score_cr
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import cumulative_incidence_competing_risks
from sksurv.util import Surv

from runtime_paths import work_root
ROOT = work_root()
DATA = ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"
MASTER = ROOT / "data/processed/v0.4/patient_master_v0.4_2000_2023.csv"
OUT = ROOT / "revision"
EVENT_OUT = OUT / "01_event_sensitivity"
EXPLAIN_OUT = OUT / "02_model_explainability"
EXCLUDED_OUT = OUT / "03_excluded_population"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from modeling_core import (  # noqa: E402
    CAUSE_LABELS,
    PREDICTOR_COLUMNS,
    make_cause_outcome,
    make_preprocessor,
)
from modeling_v03_core import (  # noqa: E402
    RiskScoreCoxCalibratedModel,
    XGBoostCoxRiskEstimator,
    predict_cif_from_cause_models,
)


SEED = 20260723
EVAL_TIMES = np.arange(12.0, 61.0, 1.0)
REPORT_TIMES = (12.0, 36.0, 60.0)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 20.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

DISPLAY = {
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "mpn_subtype": "MPN subtype",
    "diagnosis_year": "Diagnosis year",
    "rural_urban": "Rural–urban category",
    "median_household_income": "Household income",
    "chemotherapy": "Registry-recorded chemotherapy",
}
GROUP_COLUMNS = {
    "age": ["age_numeric", "age_85plus"],
    "sex": ["sex"],
    "race": ["race"],
    "mpn_subtype": ["mpn_subtype"],
    "diagnosis_year": ["diagnosis_year"],
    "rural_urban": ["rural_urban"],
    "median_household_income": ["median_household_income"],
    "chemotherapy": ["chemotherapy"],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def trap_mean(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.trapezoid(values[valid], times[valid]) / (times[valid][-1] - times[valid][0]))


def fit_xgboost_competing_risk(
    development: pd.DataFrame, test: pd.DataFrame
) -> tuple[object, dict[int, object], np.ndarray, dict]:
    preprocessor = make_preprocessor()
    x_dev = np.asarray(preprocessor.fit_transform(development[PREDICTOR_COLUMNS]), dtype=float)
    x_test = np.asarray(preprocessor.transform(test[PREDICTOR_COLUMNS]), dtype=float)
    models: dict[int, object] = {}
    for cause in CAUSE_LABELS:
        risk = XGBoostCoxRiskEstimator(
            **XGB_PARAMS,
            n_jobs=-1,
            random_state=SEED + cause,
        )
        model = RiskScoreCoxCalibratedModel(risk)
        model.fit(x_dev, make_cause_outcome(development, cause))
        models[cause] = model
    full = predict_cif_from_cause_models(models, x_test, EVAL_TIMES)
    pred = full.cif[1]
    metrics = evaluate_prediction(development, test, pred)
    return preprocessor, models, pred, metrics


def predict_target_cif(preprocessor: object, models: dict[int, object], frame: pd.DataFrame) -> np.ndarray:
    x = np.asarray(preprocessor.transform(frame[PREDICTOR_COLUMNS]), dtype=float)
    return predict_cif_from_cause_models(models, x, EVAL_TIMES).cif[1]


def evaluate_prediction(development: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> dict:
    scores = score_cr(
        {"model": pred},
        test["time_months"].to_numpy(float),
        test["event_type"].to_numpy(int),
        EVAL_TIMES,
        cause=1,
        metrics=("auc", "brier"),
    )
    auc = scores.auc.set_index("times")["AUC"]
    brier = scores.brier.set_index("times")["Brier"]
    risk60 = pred[:, -1]
    test_target = Surv.from_arrays(test.event_type.to_numpy(int) == 1, test.time_months.to_numpy(float))
    dev_target = Surv.from_arrays(
        development.event_type.to_numpy(int) == 1,
        development.time_months.to_numpy(float),
    )
    out = {
        "iAUC": trap_mean(EVAL_TIMES, scores.auc["AUC"].to_numpy(float)),
        "IBS": trap_mean(EVAL_TIMES, scores.brier["Brier"].to_numpy(float)),
        "Harrell_C_60m": float(
            concordance_index_censored(test.event_type.to_numpy(int) == 1, test.time_months.to_numpy(float), risk60)[0]
        ),
        "Uno_C_60m": float(concordance_index_ipcw(dev_target, test_target, risk60, tau=60.0)[0]),
    }
    for horizon in REPORT_TIMES:
        out[f"AUC_{int(horizon)}m"] = float(auc.loc[horizon])
        out[f"Brier_{int(horizon)}m"] = float(brier.loc[horizon])
    return out


def step_at(times: np.ndarray, values: np.ndarray, horizon: float) -> float:
    position = np.searchsorted(times, horizon, side="right") - 1
    return 0.0 if position < 0 else float(values[position])


def cif_summary(frame: pd.DataFrame, scenario: str) -> list[dict]:
    times, estimates, intervals = cumulative_incidence_competing_risks(
        frame.event_type.to_numpy(int),
        frame.time_months.to_numpy(float),
        conf_level=0.95,
        conf_type="log-log",
        var_type="Aalen",
    )
    rows = []
    for event in (1, 2, 3):
        for horizon in REPORT_TIMES:
            rows.append(
                {
                    "scenario": scenario,
                    "event_type": event,
                    "horizon_months": horizon,
                    "cif": step_at(times, estimates[event], horizon),
                    "ci_lower": step_at(times, intervals[event, 0], horizon),
                    "ci_upper": step_at(times, intervals[event, 1], horizon),
                }
            )
    return rows


def build_scenarios(ml: pd.DataFrame, master: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    final_master = master.loc[
        master.index_year.between(2004, 2023)
        & master.baseline_malignancy_at_zero.eq(0)
        & master.ml_time_valid.eq(1)
    ].copy()
    audit = final_master[[
        "patient_id",
        "index_Vital status recode (study cutoff used)",
        "index_survival_months_numeric",
    ]].merge(ml[["patient_id", "time_months", "event_type"]], on="patient_id", how="right")
    death_time = pd.to_numeric(audit.index_survival_months_numeric, errors="coerce")
    tie = (
        audit["index_Vital status recode (study cutoff used)"].eq("Dead")
        & audit.event_type.isin([1, 3])
        & pd.to_numeric(audit.time_months, errors="coerce").eq(death_time)
    )
    tie_ids = set(audit.loc[tie, "patient_id"])

    death_priority = ml.copy()
    mask = death_priority.patient_id.isin(tie_ids)
    death_priority.loc[mask, "event_type"] = 2
    death_priority.loc[mask, "event_spm"] = 0
    death_priority.loc[mask, "event_hematologic"] = 0
    death_priority.loc[mask, "event_death"] = 1

    exclude_ties = ml.loc[~ml.patient_id.isin(tie_ids)].copy()
    early_event = ml.event_type.ne(0) & ml.time_months.lt(3)
    exclude_early_events = ml.loc[~early_event].copy()

    changes = []
    for _, row in ml.loc[ml.patient_id.isin(tie_ids), ["patient_id", "time_months", "event_type"]].iterrows():
        changes.append(
            {
                "patient_id": row.patient_id,
                "scenario": "death_priority",
                "action": "reclassified_to_death",
                "original_event_type": int(row.event_type),
                "new_event_type": 2,
                "time_months": float(row.time_months),
            }
        )
        changes.append(
            {
                "patient_id": row.patient_id,
                "scenario": "exclude_same_month_tumour_death",
                "action": "excluded",
                "original_event_type": int(row.event_type),
                "new_event_type": np.nan,
                "time_months": float(row.time_months),
            }
        )
    for _, row in ml.loc[early_event, ["patient_id", "time_months", "event_type"]].iterrows():
        changes.append(
            {
                "patient_id": row.patient_id,
                "scenario": "exclude_month_1_2_events",
                "action": "excluded_event_within_months_1_2",
                "original_event_type": int(row.event_type),
                "new_event_type": np.nan,
                "time_months": float(row.time_months),
            }
        )
    return {
        "main_tumour_priority": ml.copy(),
        "death_priority": death_priority,
        "exclude_same_month_tumour_death": exclude_ties,
        "exclude_month_1_2_events": exclude_early_events,
    }, pd.DataFrame(changes)


def event_sensitivity(ml: pd.DataFrame, master: pd.DataFrame) -> tuple[object, dict[int, object]]:
    EVENT_OUT.mkdir(parents=True, exist_ok=True)
    scenarios, changes = build_scenarios(ml, master)
    changes.to_csv(EVENT_OUT / "event_sensitivity_patient_audit.csv", index=False, encoding="utf-8-sig")
    count_rows, cif_rows, metric_rows = [], [], []
    main_model = None
    for scenario, frame in scenarios.items():
        counts = frame.event_type.value_counts().reindex([0, 1, 2, 3], fill_value=0)
        count_rows.append(
            {
                "scenario": scenario,
                "n": len(frame),
                "event_0_censoring": int(counts[0]),
                "event_1_nonhematologic_spm": int(counts[1]),
                "event_2_death": int(counts[2]),
                "event_3_hematologic_mpn": int(counts[3]),
                "development_n": int(frame.data_split.eq("development").sum()),
                "test_n": int(frame.data_split.eq("test").sum()),
            }
        )
        cif_rows.extend(cif_summary(frame, scenario))
        development = frame.loc[frame.data_split.eq("development")].reset_index(drop=True)
        test = frame.loc[frame.data_split.eq("test")].reset_index(drop=True)
        preprocessor, models, prediction, metrics = fit_xgboost_competing_risk(development, test)
        metric_rows.append({"scenario": scenario, **metrics})
        np.savez_compressed(
            EVENT_OUT / f"xgboost_predictions_{scenario}.npz",
            patient_id=test.patient_id.to_numpy(),
            eval_times=EVAL_TIMES,
            target_cif=prediction.astype(np.float32),
            time_months=test.time_months.to_numpy(float),
            event_type=test.event_type.to_numpy(int),
        )
        if scenario == "main_tumour_priority":
            main_model = (preprocessor, models)
    counts = pd.DataFrame(count_rows)
    cifs = pd.DataFrame(cif_rows)
    metrics = pd.DataFrame(metric_rows)
    main_cif = cifs.loc[(cifs.scenario.eq("main_tumour_priority")) & cifs.event_type.eq(1), ["horizon_months", "cif"]]
    main_cif = dict(zip(main_cif.horizon_months, main_cif.cif))
    cifs["absolute_difference_from_main"] = cifs.apply(
        lambda row: row.cif - main_cif[row.horizon_months] if row.event_type == 1 else np.nan,
        axis=1,
    )
    main_metrics = metrics.loc[metrics.scenario.eq("main_tumour_priority")].iloc[0]
    for metric in ["iAUC", "IBS", "AUC_12m", "AUC_36m", "AUC_60m"]:
        metrics[f"delta_{metric}_from_main"] = metrics[metric] - float(main_metrics[metric])
    counts.to_csv(EVENT_OUT / "Table_event_sensitivity_cohort_counts.csv", index=False, encoding="utf-8-sig")
    cifs.to_csv(EVENT_OUT / "Table_event_sensitivity_CIF.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(EVENT_OUT / "Table_event_sensitivity_XGBoost_performance.csv", index=False, encoding="utf-8-sig")
    if main_model is None:
        raise RuntimeError("Main XGBoost model was not fitted")
    return main_model


def permutation_importance(
    ml: pd.DataFrame, preprocessor: object, models: dict[int, object], repeats: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    development = ml.loc[ml.data_split.eq("development")].reset_index(drop=True)
    test = ml.loc[ml.data_split.eq("test")].reset_index(drop=True)
    baseline_prediction = predict_target_cif(preprocessor, models, test)
    baseline = evaluate_prediction(development, test, baseline_prediction)
    rng = np.random.default_rng(SEED + 700)
    rows = []
    for group, columns in GROUP_COLUMNS.items():
        for repeat in range(1, repeats + 1):
            permuted = test.copy()
            order = rng.permutation(len(permuted))
            for column in columns:
                permuted[column] = test[column].to_numpy()[order]
            prediction = predict_target_cif(preprocessor, models, permuted)
            metrics = evaluate_prediction(development, test, prediction)
            rows.append(
                {
                    "variable": group,
                    "variable_label": DISPLAY[group],
                    "repeat": repeat,
                    "delta_iAUC": baseline["iAUC"] - metrics["iAUC"],
                    "delta_IBS": metrics["IBS"] - baseline["IBS"],
                    **{
                        f"delta_AUC_{int(h)}m": baseline[f"AUC_{int(h)}m"] - metrics[f"AUC_{int(h)}m"]
                        for h in REPORT_TIMES
                    },
                }
            )
        print(f"Permutation importance completed: {group}", flush=True)
    long = pd.DataFrame(rows)
    summary_rows = []
    for (variable, label), subset in long.groupby(["variable", "variable_label"], sort=False):
        row = {"variable": variable, "variable_label": label}
        for metric in ["delta_iAUC", "delta_IBS", "delta_AUC_12m", "delta_AUC_36m", "delta_AUC_60m"]:
            values = subset[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_lower"] = float(np.percentile(values, 2.5))
            row[f"{metric}_upper"] = float(np.percentile(values, 97.5))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("delta_iAUC_mean", ascending=False)
    return long, summary, baseline_prediction


def shap_analysis(
    ml: pd.DataFrame,
    preprocessor: object,
    models: dict[int, object],
    baseline_prediction: np.ndarray,
    sample_n: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = ml.loc[ml.data_split.eq("test")].reset_index(drop=True)
    rng = np.random.default_rng(SEED + 701)
    sample_index = np.sort(rng.choice(len(test), size=min(sample_n, len(test)), replace=False))
    sample = test.iloc[sample_index].copy().reset_index(drop=True)
    x_sample = np.asarray(preprocessor.transform(sample[PREDICTOR_COLUMNS]), dtype=float)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    xgb_model = models[1].risk_estimator_.model_
    contributions = xgb_model.get_booster().predict(xgb.DMatrix(x_sample), pred_contribs=True)
    shap_values = contributions[:, :-1]
    bias = contributions[:, -1]
    margin = xgb_model.get_booster().predict(xgb.DMatrix(x_sample), output_margin=True)
    maximum_additivity_error = float(np.max(np.abs(shap_values.sum(axis=1) + bias - margin)))

    def indices_for(group: str) -> np.ndarray:
        mapping = {
            "age": ["continuous__age_numeric", "binary__age_85plus"],
            "diagnosis_year": ["continuous__diagnosis_year"],
            "chemotherapy": ["binary__chemotherapy"],
            "sex": ["categorical__sex_"],
            "race": ["categorical__race_"],
            "mpn_subtype": ["categorical__mpn_subtype_"],
            "rural_urban": ["categorical__rural_urban_"],
            "median_household_income": ["categorical__median_household_income_"],
        }
        wanted = mapping[group]
        return np.asarray(
            [i for i, name in enumerate(feature_names) if any(name == key or name.startswith(key) for key in wanted)],
            dtype=int,
        )

    group_values = {}
    group_summary = []
    for group in GROUP_COLUMNS:
        idx = indices_for(group)
        if not len(idx):
            raise RuntimeError(f"No encoded columns mapped to {group}; available={feature_names.tolist()}")
        values = shap_values[:, idx].sum(axis=1)
        group_values[group] = values
        group_summary.append(
            {
                "variable": group,
                "variable_label": DISPLAY[group],
                "mean_absolute_SHAP": float(np.mean(np.abs(values))),
                "mean_SHAP": float(np.mean(values)),
                "encoded_columns": int(len(idx)),
            }
        )
    group_summary_frame = pd.DataFrame(group_summary).sort_values("mean_absolute_SHAP", ascending=False)
    encoded_summary = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "mean_absolute_SHAP": np.mean(np.abs(shap_values), axis=0),
            "mean_SHAP": np.mean(shap_values, axis=0),
        }
    ).sort_values("mean_absolute_SHAP", ascending=False)
    patient = pd.DataFrame({"patient_id": sample.patient_id, "tree_shap_bias": bias})
    for group, values in group_values.items():
        patient[f"SHAP_{group}"] = values
    patient["age_numeric"] = sample.age_numeric.to_numpy()
    patient["diagnosis_year"] = sample.diagnosis_year.to_numpy()
    patient["event_type"] = sample.event_type.to_numpy()
    patient["predicted_60m_CIF"] = baseline_prediction[sample_index, -1]
    metadata = {
        "sample_n": int(len(sample)),
        "random_seed": SEED + 701,
        "explained_output": "XGBoost target-cause log relative hazard margin before Cox calibration",
        "maximum_tree_shap_additivity_error": maximum_additivity_error,
    }
    (EXPLAIN_OUT / "SHAP_method_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return group_summary_frame, encoded_summary, patient


def explainability(ml: pd.DataFrame, preprocessor: object, models: dict[int, object]) -> None:
    EXPLAIN_OUT.mkdir(parents=True, exist_ok=True)
    long, permutation, baseline_prediction = permutation_importance(ml, preprocessor, models, repeats=50)
    long.to_csv(EXPLAIN_OUT / "permutation_importance_50_repeats_long.csv", index=False, encoding="utf-8-sig")
    permutation.to_csv(EXPLAIN_OUT / "Table_permutation_importance_summary.csv", index=False, encoding="utf-8-sig")
    shap_group, shap_encoded, shap_patient = shap_analysis(
        ml, preprocessor, models, baseline_prediction, sample_n=3000
    )
    shap_group.to_csv(EXPLAIN_OUT / "Table_grouped_SHAP_importance.csv", index=False, encoding="utf-8-sig")
    shap_encoded.to_csv(EXPLAIN_OUT / "Table_encoded_feature_SHAP_importance.csv", index=False, encoding="utf-8-sig")
    shap_patient.to_csv(EXPLAIN_OUT / "SHAP_patient_level_sample.csv", index=False, encoding="utf-8-sig")
    joblib.dump(
        {"preprocessor": preprocessor, "cause_specific_models": models, "eval_times": EVAL_TIMES},
        EXPLAIN_OUT / "locked_XGBoost_competing_risk_model.joblib",
        compress=3,
    )

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.1, 1.0, 1.25]})
    perm_plot = permutation.sort_values("delta_iAUC_mean")
    axes[0].errorbar(
        perm_plot.delta_iAUC_mean,
        np.arange(len(perm_plot)),
        xerr=np.vstack(
            [
                perm_plot.delta_iAUC_mean - perm_plot.delta_iAUC_lower,
                perm_plot.delta_iAUC_upper - perm_plot.delta_iAUC_mean,
            ]
        ),
        fmt="o",
        color="#235789",
        ecolor="#8FB3D9",
        capsize=2,
        markersize=4,
    )
    axes[0].axvline(0, color="#777777", lw=0.7, ls="--")
    axes[0].set_yticks(np.arange(len(perm_plot)), perm_plot.variable_label)
    axes[0].set_xlabel("Decrease in iAUC after permutation")
    axes[0].set_title("Final-CIF permutation importance")

    shap_plot = shap_group.sort_values("mean_absolute_SHAP")
    axes[1].barh(np.arange(len(shap_plot)), shap_plot.mean_absolute_SHAP, color="#D98C3F")
    axes[1].set_yticks(np.arange(len(shap_plot)), shap_plot.variable_label)
    axes[1].set_xlabel("Mean |grouped TreeSHAP|")
    axes[1].set_title("Target-cause XGBoost importance")

    age = shap_patient.sort_values("age_numeric")
    axes[2].scatter(age.age_numeric, age.SHAP_age, s=5, alpha=0.18, color="#235789", rasterized=True)
    age["age_bin"] = pd.qcut(age.age_numeric, q=12, duplicates="drop")
    smooth = age.groupby("age_bin", observed=True).agg(x=("age_numeric", "mean"), y=("SHAP_age", "mean"))
    axes[2].plot(smooth.x, smooth.y, color="#B23A48", lw=1.5)
    axes[2].axhline(0, color="#777777", lw=0.7, ls="--")
    axes[2].set_xlabel("Age at MPN diagnosis, years")
    axes[2].set_ylabel("Grouped TreeSHAP for age")
    axes[2].set_title("Age contribution to target log hazard")
    for label, ax in zip("abc", axes):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, EXPLAIN_OUT, "Figure_model_explainability")


def smd_continuous(reference: pd.Series, comparison: pd.Series) -> float:
    a, b = reference.dropna().astype(float), comparison.dropna().astype(float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def smd_binary(reference: pd.Series, comparison: pd.Series, level: str) -> float:
    p0 = float(reference.eq(level).mean())
    p1 = float(comparison.eq(level).mean())
    pooled = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2.0)
    return float((p1 - p0) / pooled) if pooled > 0 else 0.0




def main() -> None:
    started = time.perf_counter()
    for directory in [EVENT_OUT, EXPLAIN_OUT, EXCLUDED_OUT]:
        directory.mkdir(parents=True, exist_ok=True)
    ml = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    master = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    preprocessor, models = event_sensitivity(ml, master)
    explainability(ml, preprocessor, models)
    summary = {
        "status": "PASS",
        "main_rows": int(len(ml)),
        "event_sensitivity_scenarios": 4,
        "permutation_repeats_per_variable": 50,
        "shap_sample_n": 3000,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (OUT / "core_revision_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)



if __name__ == "__main__":
    main()
