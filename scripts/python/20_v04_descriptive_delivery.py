"""Create v0.4 descriptive tables and publication figures before model reporting.

The endpoint is first non-hematologic malignant SPM. Death and a subsequent
malignant hematologic neoplasm (including registry-recorded MPN evolution) are
competing events.  This script never loads model predictions or the test set.
"""

from __future__ import annotations

import os
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
VERSION = os.environ.get("MPN_ANALYSIS_VERSION", "v0.4")
DATA_DIR = ROOT / "data" / "processed" / VERSION
MASTER_PATH = DATA_DIR / "patient_master_v0.4_2000_2023.csv"
ML_PATH = DATA_DIR / "ml_competing_risk_v0.4_2004_2023.csv"
RAW_PATH = ROOT / "data" / "raw" / "SEER_MPN_first_primary_person_selection_2000_2023.txt"
DELIVERY_DIR = ROOT / "deliverables" / VERSION
TABLE_DIR = DELIVERY_DIR / "tables" / "source_data"
FIGURE_DIR = DELIVERY_DIR / "figures"
FIGURE_SOURCE_DIR = DELIVERY_DIR / "figures" / "source_data"

EVENTS = {
    1: ("Non-haematological malignant SPM", "#0F4D92"),
    2: ("Death", "#4D4D4D"),
    3: ("Haematological/MPN competing event", "#B64342"),
}
SUBTYPE_COLORS = {"PV": "#0F4D92", "ET": "#42949E", "PMF": "#9A4D8E"}
HORIZONS = (12, 36, 60)


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, options in (
        ("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 220}),
    ):
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight", **options)
    plt.close(fig)


def step_at(times: np.ndarray, values: np.ndarray, horizon: float) -> float:
    position = np.searchsorted(times, horizon, side="right") - 1
    return 0.0 if position < 0 else float(values[position])


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, original in enumerate(order):
        running = max(running, min(1.0, values[original] * (len(values) - rank)))
        adjusted[original] = running
    return adjusted.tolist()


def p_text(value: float) -> str:
    return "P<0.001" if value < 0.001 else f"P={value:.3f}"


def cif(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cumulative_incidence_competing_risks(
        data["event_type"].to_numpy(dtype=int),
        data["time_months"].to_numpy(dtype=float),
        conf_level=0.95,
        conf_type="log-log",
        var_type="Aalen",
    )


def cif_source(data: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    times, estimates, intervals = cif(data)
    long_rows, summary_rows = [], []
    for event, (label, _) in EVENTS.items():
        for index, month in enumerate(times):
            long_rows.append({
                "group": group, "event_type": event, "event": label,
                "time_months": float(month), "cif": float(estimates[event, index]),
                "ci_lower": float(intervals[event, 0, index]),
                "ci_upper": float(intervals[event, 1, index]),
            })
        for horizon in HORIZONS:
            summary_rows.append({
                "group": group, "event_type": event, "event": label,
                "horizon_months": horizon,
                "cif": step_at(times, estimates[event], horizon),
                "ci_lower": step_at(times, intervals[event, 0], horizon),
                "ci_upper": step_at(times, intervals[event, 1], horizon),
            })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def make_flow(master: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    year = master.loc[master["ml_year_eligible"].eq(1)]
    no_zero = year.loc[year["baseline_malignancy_at_zero"].eq(0)]
    valid = no_zero.loc[no_zero["ml_time_valid"].eq(1)]
    raw_records = sum(1 for _ in RAW_PATH.open(encoding="utf-8-sig")) - 1
    return pd.DataFrame([
        {"step": "Tumour-level records in source export", "n": raw_records},
        {"step": "Unique persons with classic MPN as first primary", "n": len(master)},
        {"step": "Index MPN diagnosed in 2004–2023", "n": len(year)},
        {"step": "No synchronous malignant tumour at month 0", "n": len(no_zero)},
        {"step": "Follow-up/event time available and at least 1 month", "n": len(valid)},
        {"step": "Primary machine-learning cohort", "n": len(ml)},
    ])


def draw_flow(flow: pd.DataFrame, event_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_axis_off()
    y = np.linspace(0.92, 0.47, len(flow))
    for row, position in zip(flow.itertuples(index=False), y):
        ax.text(0.5, position, f"{row.step}\nn = {row.n:,}", ha="center", va="center",
                fontsize=8, bbox={"boxstyle": "round,pad=0.48", "facecolor": "#F5F7FA",
                                  "edgecolor": "#3775BA", "linewidth": 0.9}, zorder=3)
    for upper, lower in zip(y[:-1], y[1:]):
        ax.annotate("", xy=(0.5, lower + 0.035), xytext=(0.5, upper - 0.035),
                    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#767676"}, zorder=1)
    exclusions = [
        f"Excluded: diagnosis in 2000–2003, n={flow.iloc[1].n - flow.iloc[2].n:,}",
        f"Excluded: synchronous malignant tumour at month 0, n={flow.iloc[2].n - flow.iloc[3].n:,}",
        f"Excluded: follow-up/event time missing or <1 month, n={flow.iloc[3].n - flow.iloc[4].n:,}",
    ]
    for text, position in zip(exclusions, y[2:5]):
        ax.text(0.98, position + 0.038, text, ha="right", va="center", fontsize=6.4, color="#4D4D4D")
    ax.text(0.5, 0.32, "First post-index outcome disposition", ha="center", va="center",
            fontweight="bold", fontsize=8, zorder=3)
    x = [0.10, 0.36, 0.62, 0.88]
    colors = ["#767676", EVENTS[1][1], EVENTS[2][1], EVENTS[3][1]]
    for row, xpos, color in zip(event_counts.itertuples(index=False), x, colors):
        ax.annotate("", xy=(xpos, 0.19), xytext=(0.5, 0.29),
                    arrowprops={"arrowstyle": "->", "lw": 0.75, "color": "#767676"}, zorder=1)
        ax.text(xpos, 0.11, f"{row.event}\nn = {row.n:,}", ha="center", va="center", fontsize=6.4,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "linewidth": 1.0}, zorder=3)
    fig.tight_layout()
    save_figure(fig, "Figure_1_cohort_flow")


def draw_competing_cif(data: pd.DataFrame, overall_long: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    for event, (label, color) in EVENTS.items():
        subset = overall_long.loc[overall_long["event_type"].eq(event)]
        ax.step(subset["time_months"], subset["cif"] * 100, where="post", label=label, color=color, lw=1.7)
        ax.fill_between(subset["time_months"], subset["ci_lower"] * 100, subset["ci_upper"] * 100,
                        step="post", color=color, alpha=0.12)
    ax.set(xlim=(0, 120), ylim=(0, None), xlabel="Months since index MPN diagnosis",
           ylabel="Cumulative incidence (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.text(0.99, 0.03, "Aalen–Johansen estimates; 95% pointwise confidence intervals",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#4D4D4D")
    fig.tight_layout()
    save_figure(fig, "Figure_S1_all_competing_events")


def draw_subtype_cif(data: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    for subtype in ("PV", "ET", "PMF"):
        subset = data.loc[data["mpn_subtype"].eq(subtype)]
        times, estimates, _ = cif(subset)
        ax.step(times, estimates[1] * 100, where="post", color=SUBTYPE_COLORS[subtype], lw=1.8,
                label=f"{subtype} (n={len(subset):,})")
    pair_lines = [f"{r.comparison}: {p_text(r.p_holm)}" for r in tests.itertuples(index=False)
                  if r.comparison != "PV vs ET vs PMF"]
    overall = tests.loc[tests["comparison"].eq("PV vs ET vs PMF"), "p_raw"].iloc[0]
    ax.text(0.98, 0.97, "Gray's test for non-hematologic SPM\n" + f"Overall: {p_text(overall)}\n" + "\n".join(pair_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB"})
    ax.set(xlim=(0, 120), ylim=(0, None), xlabel="Months since index MPN diagnosis",
           ylabel="Non-hematologic malignant SPM cumulative incidence (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "Figure_2_nonhematologic_SPM_by_MPN_subtype")


def baseline_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = {"Overall": data, "Development": data.loc[data["data_split"].eq("development")],
               "Test": data.loc[data["data_split"].eq("test")]}
    rows = []
    for label, subset in columns.items():
        n = len(subset)
        rows.extend([
            {"Characteristic": "Patients", "Level": "", "Column": label, "Value": str(n)},
            {"Characteristic": "Age, years", "Level": "mean (SD)", "Column": label,
             "Value": f"{subset.age_numeric.mean():.1f} ({subset.age_numeric.std():.1f})"},
            {"Characteristic": "Age, years", "Level": "median [IQR]", "Column": label,
             "Value": f"{subset.age_numeric.median():.0f} [{subset.age_numeric.quantile(.25):.0f}–{subset.age_numeric.quantile(.75):.0f}]"},
        ])
        for variable, name in (("sex", "Sex"), ("race", "Race"), ("mpn_subtype", "MPN subtype"),
                               ("diagnosis_era", "Diagnosis era"), ("rural_urban", "Rural–urban category"),
                               ("median_household_income", "Median household income"),
                               ("chemotherapy_raw", "Initial chemotherapy")):
            for level, count in subset[variable].fillna("Missing").value_counts().sort_index().items():
                rows.append({"Characteristic": name, "Level": str(level), "Column": label,
                             "Value": f"{count:,} ({count / n * 100:.1f}%)"})
    wide = pd.DataFrame(rows).pivot(index=["Characteristic", "Level"], columns="Column", values="Value").reset_index()
    return wide[["Characteristic", "Level", "Overall", "Development", "Test"]]


def main() -> None:
    setup_style()
    for directory in (TABLE_DIR, FIGURE_DIR, FIGURE_SOURCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig", low_memory=False)
    data = pd.read_csv(ML_PATH, encoding="utf-8-sig", low_memory=False)
    assert len(data) == 85680 and set(data.event_type.unique()) == {0, 1, 2, 3}
    assert data.time_months.between(1, 239).all()

    flow = make_flow(master, data)
    flow.to_csv(TABLE_DIR / "Figure_1_flow_source_data.csv", index=False, encoding="utf-8-sig")
    event_counts = pd.DataFrame([
        {"event": "No first event observed", "n": int(data.event_type.eq(0).sum())},
        {"event": EVENTS[1][0], "n": int(data.event_type.eq(1).sum())},
        {"event": EVENTS[2][0], "n": int(data.event_type.eq(2).sum())},
        {"event": EVENTS[3][0], "n": int(data.event_type.eq(3).sum())},
    ])
    event_counts.to_csv(TABLE_DIR / "Figure_1_outcome_source_data.csv", index=False, encoding="utf-8-sig")
    draw_flow(flow, event_counts)

    overall_long, overall_summary = cif_source(data, "Overall")
    subtype_long, subtype_summary = [], []
    for subtype in ("PV", "ET", "PMF"):
        long, summary = cif_source(data.loc[data.mpn_subtype.eq(subtype)], subtype)
        subtype_long.append(long.loc[long.event_type.eq(1)])
        subtype_summary.append(summary.loc[summary.event_type.eq(1)])
    cif_long = pd.concat([overall_long, *subtype_long], ignore_index=True)
    cif_summary = pd.concat([overall_summary, *subtype_summary], ignore_index=True)
    cif_long.to_csv(FIGURE_SOURCE_DIR / "Figure_2_and_S1_CIF_source_data.csv", index=False, encoding="utf-8-sig")
    cif_summary.to_csv(TABLE_DIR / "Table_2_CIF_at_1_3_5_years.csv", index=False, encoding="utf-8-sig")

    pairs = [("PV", "ET"), ("PV", "PMF"), ("ET", "PMF")]
    tests = []
    overall_test = gray_test(data.time_months.to_numpy(), data.event_type.to_numpy(), data.mpn_subtype.to_numpy(), cause=1)
    tests.append({"comparison": "PV vs ET vs PMF", "p_raw": float(overall_test.pvalue), "p_holm": np.nan,
                  "test": "Overall Gray's test, rho=0"})
    pvalues = []
    pair_results = []
    for left, right in pairs:
        subset = data.loc[data.mpn_subtype.isin([left, right])]
        result = gray_test(subset.time_months.to_numpy(), subset.event_type.to_numpy(), subset.mpn_subtype.to_numpy(), cause=1)
        pair_results.append((left, right, result))
        pvalues.append(float(result.pvalue))
    for (left, right, result), adjusted in zip(pair_results, holm(pvalues)):
        tests.append({"comparison": f"{left} vs {right}", "p_raw": float(result.pvalue), "p_holm": adjusted,
                      "test": "Pairwise Gray's test, Holm adjusted"})
    test_table = pd.DataFrame(tests)
    test_table.to_csv(TABLE_DIR / "Table_S1_subtype_Gray_tests.csv", index=False, encoding="utf-8-sig")
    draw_competing_cif(data, overall_long)
    draw_subtype_cif(data, test_table)

    baseline_table(data).to_csv(TABLE_DIR / "Table_1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    outcome_by_split = (data.assign(event_label=data.event_type.map({0: "No first event observed", **{k: v[0] for k, v in EVENTS.items()}}))
                       .groupby(["data_split", "event_type", "event_label"], as_index=False).size()
                       .rename(columns={"size": "n"}))
    outcome_by_split.to_csv(TABLE_DIR / "Table_S2_outcomes_by_split.csv", index=False, encoding="utf-8-sig")
    print(f"PASS: descriptive delivery written to {DELIVERY_DIR}")


if __name__ == "__main__":
    main()
