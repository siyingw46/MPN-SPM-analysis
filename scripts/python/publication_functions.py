"""Build the publication-oriented v0.4 tables, figures, and source data.

This script does not refit or select a model. It consumes the audited locked
development/test predictions and descriptive outputs. The only derived
patient-level display is the risk-stratification panel: quartile cut points are
estimated from development OOF XGBoost-Cox 60-month risk and applied unchanged
to the locked test set.
"""

from __future__ import annotations

import json
import shutil
import math
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comprisk import gray_test
from sksurv.nonparametric import cumulative_incidence_competing_risks


from runtime_paths import work_root
ROOT = work_root()
DELIVERY = ROOT / "deliverables" / "v0.4"
MODEL = ROOT / "output" / "models" / "model_development_v0.4"
TEST = MODEL / "final_test_evaluation"
BASE = MODEL / "base_oof_evaluation"
ENSEMBLE = MODEL / "ensemble_oof_evaluation"
DATA = ROOT / "data" / "processed" / "v0.4" / "ml_competing_risk_v0.4_2004_2023.csv"
SOURCE_OLD = DELIVERY / "tables" / "source_data"

MAIN_FIG = DELIVERY / "figures" / "main"
SUPP_FIG = DELIVERY / "figures" / "supplementary"
FIG_SOURCE = DELIVERY / "figures" / "source_data_publication"
BUILD_FIG = DELIVERY / ".build_figures"
MAIN_TABLE = DELIVERY / "tables" / "main"
SUPP_TABLE = DELIVERY / "tables" / "supplementary"
QA_DIR = DELIVERY / "qa"

MODEL_LABELS = {
    "xgboost_cox": "XGBoost–Cox",
    "regularized_ridge_meta": "Ridge meta-ensemble",
    "rsf": "Random survival forest",
    "extra_survival_trees": "Extra survival trees",
    "kernel_survival_svm": "Kernel survival SVM",
    "elastic_net_cox": "Elastic-net Cox",
    "cs_cox": "Cause-specific Cox",
    "linear_survival_svm": "Linear survival SVM",
    "fine_gray": "Fine–Gray",
    "deephit_cr": "DeepHit competing risk",
}
MODEL_ORDER = [
    "xgboost_cox",
    "regularized_ridge_meta",
    "rsf",
    "extra_survival_trees",
    "kernel_survival_svm",
    "elastic_net_cox",
    "cs_cox",
    "linear_survival_svm",
    "fine_gray",
    "deephit_cr",
]
SELECTED_MODELS = [
    "xgboost_cox",
    "regularized_ridge_meta",
    "rsf",
    "cs_cox",
]
COLORS = {
    "xgboost_cox": "#005A9C",
    "regularized_ridge_meta": "#D97706",
    "rsf": "#008A8A",
    "cs_cox": "#6B7280",
    "PV": "#005A9C",
    "ET": "#008A8A",
    "PMF": "#7A5195",
}
LINESTYLES = {
    "xgboost_cox": "-",
    "regularized_ridge_meta": "--",
    "rsf": "-.",
    "cs_cox": ":",
}
HORIZONS = [12, 36, 60]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )




def save_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    word_asset_dir = BUILD_FIG / folder.name
    word_asset_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(word_asset_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def p_value_text(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "not assessed"
    if value == 0:
        return "P<1e-300"
    if value < 0.001:
        return f"P={value:.2e}"
    return f"P={value:.3f}"


def ci_text(estimate: float, lower: float, upper: float, digits: int = 3) -> str:
    return f"{estimate:.{digits}f} ({lower:.{digits}f}–{upper:.{digits}f})"


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, original in enumerate(order):
        running = max(running, min(1.0, values[original] * (len(values) - rank)))
        adjusted[original] = running
    return adjusted.tolist()


def step_at(times: np.ndarray, values: np.ndarray, horizon: float) -> float:
    position = np.searchsorted(times, horizon, side="right") - 1
    return 0.0 if position < 0 else float(values[position])










def figure1(data: pd.DataFrame) -> None:
    flow = pd.read_csv(SOURCE_OLD / "Figure_1_flow_source_data.csv", encoding="utf-8-sig")
    outcomes = pd.read_csv(SOURCE_OLD / "Figure_1_outcome_source_data.csv", encoding="utf-8-sig")
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for current in (ax, ax2):
        current.set_axis_off()
    ax.set(xlim=(0, 2.2), ylim=(-0.3, 5.5))
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    rows = flow.iloc[:5]
    for row, ypos in zip(rows.itertuples(index=False), y):
        ax.text(
            0.65,
            ypos,
            f"{row.step}\nn={row.n:,}",
            ha="center",
            va="center",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F5F8FC", edgecolor="#005A9C", linewidth=0.9),
        )
    for upper, lower in zip(y[:-1], y[1:]):
        ax.annotate("", xy=(0.65, lower + 0.30), xytext=(0.65, upper - 0.30), arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    exclusion_texts = [
        f"Excluded: diagnosis in 2000–2003\nn={flow.iloc[1].n-flow.iloc[2].n:,}",
        f"Excluded: synchronous tumour at month 0\nn={flow.iloc[2].n-flow.iloc[3].n:,}",
        f"Excluded: missing/invalid or <1-month time\nn={flow.iloc[3].n-flow.iloc[4].n:,}",
    ]
    for text, ypos in zip(exclusion_texts, y[2:5]):
        ax.text(1.50, ypos, text, ha="left", va="center", fontsize=5.7, color="#444444")
        ax.plot([1.04, 1.42], [ypos, ypos], color="#999999", lw=0.7)
    ax.text(
        0.65,
        0.0,
        "Primary machine-learning cohort\nn=85,680",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F1FB", edgecolor="#005A9C", linewidth=1.1),
    )
    ax.annotate("", xy=(0.65, 0.30), xytext=(0.65, 0.70), arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    panel_label(ax, "a")

    ax2.set(xlim=(0, 2), ylim=(-0.2, 5.4))
    ax2.text(1.0, 5.05, "Patient-level locked split", ha="center", va="center", fontsize=8, fontweight="bold")
    split = data.data_split.value_counts()
    for xpos, label, n, color in [
        (0.52, "Development\nnested cross-validation", int(split.development), "#008A8A"),
        (1.48, "Locked internal test", int(split.test), "#D97706"),
    ]:
        ax2.annotate("", xy=(xpos, 4.15), xytext=(1.0, 4.75), arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
        ax2.text(
            xpos,
            3.85,
            f"{label}\nn={n:,}",
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor=color, linewidth=1.0),
        )
    ax2.text(1.0, 3.05, "Mutually exclusive first outcome", ha="center", va="center", fontsize=7.6, fontweight="bold")
    outcome_colors = ["#737373", "#005A9C", "#4D4D4D", "#7A5195"]
    oy = [2.45, 1.75, 1.05, 0.35]
    for row, ypos, color in zip(outcomes.itertuples(index=False), oy, outcome_colors):
        short = str(row.event).replace("Haematological/MPN competing event", "Haematological/MPN competing event")
        ax2.text(
            1.0,
            ypos,
            f"{short}\nn={row.n:,}",
            ha="center",
            va="center",
            fontsize=6.6,
            color=color,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=color, linewidth=0.8),
        )
    panel_label(ax2, "b")
    fig.tight_layout()
    save_figure(fig, MAIN_FIG, "Figure_1_cohort_flow")
    pd.concat(
        [
            flow.assign(component="cohort flow"),
            outcomes.rename(columns={"event": "step"}).assign(component="outcome disposition"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(FIG_SOURCE / "Figure_1_source_data.csv", index=False, encoding="utf-8-sig")


def figure2(frames: dict[str, pd.DataFrame]) -> None:
    long = pd.read_csv(DELIVERY / "figures" / "source_data" / "Figure_2_and_S1_CIF_source_data.csv", encoding="utf-8-sig")
    sub = long[(long.group.isin(["PV", "ET", "PMF"])) & (long.event_type == 1)].copy()
    summary = frames["cif_summary"]
    summary = summary[(summary.group.isin(["PV", "ET", "PMF"])) & (summary.event_type == 1)].copy()
    tests = frames["gray"]

    fig = plt.figure(figsize=(7.2, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for subtype in ["PV", "ET", "PMF"]:
        s = sub[sub.group == subtype]
        ax1.step(s.time_months, s.cif * 100, where="post", color=COLORS[subtype], label=subtype)
        ax1.fill_between(s.time_months, s.ci_lower * 100, s.ci_upper * 100, step="post", color=COLORS[subtype], alpha=0.10)
    ax1.set(xlim=(0, 120), xlabel="Months since index MPN diagnosis", ylabel="Cumulative incidence of non-hematologic SPM (%)")
    ax1.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax1.legend(title="MPN subtype", loc="upper left")
    overall = tests.loc[tests.comparison.eq("PV vs ET vs PMF"), "p_raw"].iloc[0]
    pair_lines = []
    for row in tests.loc[~tests.comparison.eq("PV vs ET vs PMF")].itertuples(index=False):
        pair_lines.append(f"{row.comparison}, Holm: {p_value_text(row.p_holm)}")
    ax1.text(
        0.98,
        0.03,
        "Gray overall (PV vs ET vs PMF): " + p_value_text(overall) + "\n" + "\n".join(pair_lines),
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#BBBBBB"),
    )
    panel_label(ax1, "a")

    markers = {12: "o", 36: "s", 60: "D"}
    offsets = {12: -0.15, 36: 0.0, 60: 0.15}
    ybase = {"PV": 2, "ET": 1, "PMF": 0}
    for horizon in HORIZONS:
        part = summary[summary.horizon_months == horizon]
        for row in part.itertuples(index=False):
            yval = ybase[row.group] + offsets[horizon]
            ax2.errorbar(
                row.cif * 100,
                yval,
                xerr=[[max(0, row.cif - row.ci_lower) * 100], [(row.ci_upper - row.cif) * 100]],
                fmt=markers[horizon],
                color=COLORS[row.group],
                mec="white",
                mew=0.4,
                ms=4.5,
                capsize=2,
                lw=0.9,
                label=f"{horizon // 12} year" if row.group == "PV" else None,
            )
    ax2.set_yticks([0, 1, 2], ["PMF", "ET", "PV"])
    ax2.set_xlabel("Cumulative incidence (%, 95% CI)")
    ax2.grid(axis="x", color="#DDDDDD", lw=0.5)
    ax2.legend(title="Horizon", loc="lower right")
    panel_label(ax2, "b")
    fig.tight_layout()
    save_figure(fig, MAIN_FIG, "Figure_2_nonhematologic_SPM_by_MPN_subtype")
    pd.concat(
        [
            sub.assign(panel="a: CIF curve"),
            summary.assign(panel="b: landmark CIF"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(FIG_SOURCE / "Figure_2_source_data.csv", index=False, encoding="utf-8-sig")








def figure4(frames: dict[str, pd.DataFrame]) -> None:
    bins = frames["calibration_bins"]
    summary = frames["calibration"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, horizon, label in zip(axes, HORIZONS, ["a", "b", "c"]):
        part = bins[(bins.times == horizon) & bins.model.isin(["xgboost_cox", "regularized_ridge_meta"])]
        maximum = max(part.predicted_decile.max(), part.upper_ci.max()) * 1.08
        ax.plot([0, maximum], [0, maximum], color="#777777", ls=":", lw=1, label="Perfect calibration")
        for model in ["xgboost_cox", "regularized_ridge_meta"]:
            m = part[part.model == model].sort_values("predicted_decile")
            ax.errorbar(
                m.predicted_decile,
                m.observed_freq,
                yerr=[m.observed_freq - m.lower_ci, m.upper_ci - m.observed_freq],
                fmt="o-",
                color=COLORS[model],
                ms=3,
                lw=1.1,
                capsize=1.5,
                label=MODEL_LABELS[model],
            )
        notes = []
        for model in ["xgboost_cox", "regularized_ridge_meta"]:
            s = summary[(summary.model == model) & (summary.times == horizon)].iloc[0]
            notes.append(f"{MODEL_LABELS[model]}: O/E={s.observed_expected_ratio:.2f}, slope={s.calibration_slope:.2f}")
        ax.text(0.03, 0.97, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
        ax.set(
            xlim=(0, maximum),
            ylim=(0, maximum),
            xlabel="Mean predicted risk",
            ylabel="Observed Aalen–Johansen CIF" if horizon == 12 else "",
            title=f"{horizon // 12}-year calibration",
        )
        ax.grid(color="#E5E5E5", lw=0.45)
        panel_label(ax, label)
    axes[0].legend(loc="lower right", fontsize=5.6)
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, MAIN_FIG, "Figure_5_calibration")
    pd.concat(
        [
            bins[bins.model.isin(["xgboost_cox", "regularized_ridge_meta"])].assign(record_type="calibration bins"),
            summary[summary.model.isin(["xgboost_cox", "regularized_ridge_meta"])].assign(record_type="calibration summary"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(FIG_SOURCE / "Figure_5_source_data.csv", index=False, encoding="utf-8-sig")


def risk_stratification(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with np.load(BASE / "development_repeated_oof_cif.npz", allow_pickle=False) as dev_npz:
        dev_times = dev_npz["eval_times"].astype(float)
        dev_risk = dev_npz["xgboost_cox"][:, np.where(dev_times == 60)[0][0]].astype(float)
    with np.load(TEST / "test_all_models_cif.npz", allow_pickle=False) as test_npz:
        test_times = test_npz["eval_times"].astype(float)
        test_risk = test_npz["xgboost_cox"][:, np.where(test_times == 60)[0][0]].astype(float)
        patient_id = test_npz["patient_id"]
        time = test_npz["time_months"].astype(float)
        event = test_npz["event_type"].astype(int)
    cutpoints = np.quantile(dev_risk, [0.25, 0.50, 0.75])
    group_index = np.digitize(test_risk, cutpoints, right=True)
    labels = np.array(["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    risk_frame = pd.DataFrame(
        {
            "patient_id": patient_id,
            "time_months": time,
            "event_type": event,
            "predicted_risk_60m": test_risk,
            "risk_group": labels[group_index],
        }
    )
    curve_rows = []
    summary_rows = []
    for label in labels:
        subset = risk_frame[risk_frame.risk_group == label]
        times, estimates, intervals = cumulative_incidence_competing_risks(
            subset.event_type.to_numpy(int),
            subset.time_months.to_numpy(float),
            conf_level=0.95,
            conf_type="log-log",
            var_type="Aalen",
        )
        keep = times <= 60
        for idx in np.where(keep)[0]:
            curve_rows.append(
                {
                    "risk_group": label,
                    "time_months": float(times[idx]),
                    "cif": float(estimates[1, idx]),
                    "ci_lower": float(intervals[1, 0, idx]),
                    "ci_upper": float(intervals[1, 1, idx]),
                }
            )
        summary_rows.append(
            {
                "risk_group": label,
                "n_test": len(subset),
                "nonhematologic_SPM_events_total": int(subset.event_type.eq(1).sum()),
                "nonhematologic_SPM_events_by_60m": int(((subset.event_type == 1) & (subset.time_months <= 60)).sum()),
                "mean_predicted_risk_60m": float(subset.predicted_risk_60m.mean()),
                "observed_CIF_60m": step_at(times, estimates[1], 60),
                "development_cutpoint_lower": (
                    np.nan if label == labels[0] else float(cutpoints[list(labels).index(label) - 1])
                ),
                "development_cutpoint_upper": (
                    np.nan if label == labels[-1] else float(cutpoints[list(labels).index(label)])
                ),
            }
        )
    overall = gray_test(
        risk_frame.time_months.to_numpy(float),
        risk_frame.event_type.to_numpy(int),
        risk_frame.risk_group.to_numpy(str),
        cause=1,
    )
    pair_rows = [
        {
            "comparison": "Overall across Q1–Q4",
            "p_raw": float(overall.pvalue),
            "p_holm": np.nan,
            "test": "Gray test, rho=0",
        }
    ]
    raw = []
    pair_objects = []
    for left, right in combinations(labels, 2):
        subset = risk_frame[risk_frame.risk_group.isin([left, right])]
        result = gray_test(
            subset.time_months.to_numpy(float),
            subset.event_type.to_numpy(int),
            subset.risk_group.to_numpy(str),
            cause=1,
        )
        raw.append(float(result.pvalue))
        pair_objects.append((left, right, result))
    for (left, right, result), adjusted in zip(pair_objects, holm(raw)):
        pair_rows.append(
            {
                "comparison": f"{left} vs {right}",
                "p_raw": float(result.pvalue),
                "p_holm": adjusted,
                "test": "Pairwise Gray test, Holm adjusted",
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def dca_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dca = frames["dca"]
    rows = []
    for horizon in HORIZONS:
        h = dca[dca.times == horizon]
        reference = (
            h[h.model.isin(["treat_none", "treat_all"])]
            .pivot(index="threshold", columns="model", values="net_benefit")
            .max(axis=1)
        )
        for model in ["xgboost_cox", "regularized_ridge_meta"]:
            m = h[h.model == model].set_index("threshold").net_benefit
            delta = m - reference
            positive = delta[delta > 0]
            best_threshold = float(delta.idxmax())
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "horizon_months": horizon,
                    "maximum_incremental_net_benefit": float(delta.max()),
                    "threshold_at_maximum_incremental_net_benefit": best_threshold,
                    "minimum_threshold_with_positive_net_benefit_vs_both_references": float(positive.index.min()) if len(positive) else np.nan,
                    "maximum_threshold_with_positive_net_benefit_vs_both_references": float(positive.index.max()) if len(positive) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def figure5(frames: dict[str, pd.DataFrame], risk_curve: pd.DataFrame, risk_summary: pd.DataFrame, risk_tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    dca = frames["dca"]
    dca_ranges = {
        12: (0.5, 4.0, -0.006, 0.013),
        36: (0.5, 8.0, -0.006, 0.038),
        60: (0.5, 12.0, -0.006, 0.060),
    }
    for ax, horizon, label in zip(axes.flat[:3], HORIZONS, ["a", "b", "c"]):
        h = dca[dca.times == horizon]
        for model, color, linestyle in [
            ("treat_none", "#999999", "-"),
            ("treat_all", "#333333", ":"),
            ("xgboost_cox", COLORS["xgboost_cox"], "-"),
            ("regularized_ridge_meta", COLORS["regularized_ridge_meta"], "--"),
        ]:
            m = h[h.model == model]
            ax.plot(
                m.threshold * 100,
                m.net_benefit,
                color=color,
                ls=linestyle,
                label=MODEL_LABELS.get(model, model.replace("_", " ").title()),
            )
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set(
            xlim=dca_ranges[horizon][:2],
            ylim=dca_ranges[horizon][2:],
            xlabel="Risk threshold (%)",
            ylabel="Net benefit" if horizon in [12, 60] else "",
            title=f"{horizon // 12}-year decision curve",
        )
        ax.grid(axis="y", color="#E3E3E3", lw=0.45)
        panel_label(ax, label)
    axes[0, 0].legend(loc="best", fontsize=5.5)

    ax = axes[1, 1]
    group_colors = {
        "Q1 lowest": "#BFD7EA",
        "Q2": "#5AA9A8",
        "Q3": "#7A5195",
        "Q4 highest": "#C23B23",
    }
    for group in ["Q1 lowest", "Q2", "Q3", "Q4 highest"]:
        part = risk_curve[risk_curve.risk_group == group]
        row = risk_summary[risk_summary.risk_group == group].iloc[0]
        ax.step(
            part.time_months,
            part.cif * 100,
            where="post",
            color=group_colors[group],
            label=f"{group} (n={int(row.n_test):,})",
        )
    p_overall = risk_tests.loc[risk_tests.comparison.eq("Overall across Q1–Q4"), "p_raw"].iloc[0]
    ax.text(
        0.03,
        0.97,
        "Gray overall across four development-defined strata:\n" + p_value_text(float(p_overall)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB"),
    )
    ax.set(xlim=(0, 60), xlabel="Months since index MPN diagnosis", ylabel="Observed non-hematologic SPM CIF (%)", title="Locked-test risk strata")
    ax.grid(axis="y", color="#E3E3E3", lw=0.45)
    ax.legend(loc="lower right", fontsize=5.5)
    panel_label(ax, "d")
    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    save_figure(fig, MAIN_FIG, "Figure_6_decision_curves_and_risk_strata")

    dca[dca.model.isin(["treat_none", "treat_all", "xgboost_cox", "regularized_ridge_meta"])].to_csv(
        FIG_SOURCE / "Figure_6_DCA_source_data.csv", index=False, encoding="utf-8-sig"
    )
    risk_curve.to_csv(FIG_SOURCE / "Figure_6_risk_strata_CIF_source_data.csv", index=False, encoding="utf-8-sig")










