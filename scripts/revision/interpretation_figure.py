"""Create editable manuscript-ready figures for the revision analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from runtime_paths import work_root
ROOT = work_root()
OUT = ROOT / "revision"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

SCENARIO_LABELS = {
    "main_tumour_priority": "Primary analysis\n(tumour priority)",
    "death_priority": "Death priority",
    "exclude_same_month_tumour_death": "Exclude same-month\ntumour–death pairs",
    "exclude_month_1_2_events": "Exclude events\nin months 1–2",
}
COLORS = ["#2B6F8A", "#D97941", "#6B8E5A", "#7B6BA8"]


def panel_label(axis, label: str) -> None:
    axis.text(-0.13, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def save_figure(fig, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)




def shap_distribution_figure() -> None:
    folder = OUT / "02_model_explainability"
    frame = pd.read_csv(folder / "SHAP_patient_level_sample.csv")
    labels = {
        "SHAP_age": "Age",
        "SHAP_sex": "Sex",
        "SHAP_diagnosis_year": "Diagnosis year",
        "SHAP_race": "Race",
        "SHAP_median_household_income": "Household income",
        "SHAP_mpn_subtype": "MPN subtype",
        "SHAP_rural_urban": "Rural–urban category",
        "SHAP_chemotherapy": "Chemotherapy",
    }
    ordered = sorted(labels, key=lambda c: frame[c].abs().mean(), reverse=True)
    rng = np.random.default_rng(20260917)
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    source_rows = []
    for y, column in enumerate(ordered):
        values = frame[column].to_numpy(float)
        jitter = rng.normal(0, 0.08, len(values))
        ax.scatter(values, y + jitter, s=5, alpha=0.17, color="#2B6F8A", rasterized=True)
        mean_abs = np.mean(np.abs(values))
        ax.plot([-mean_abs, mean_abs], [y, y], color="#D97941", linewidth=3.0, solid_capstyle="round")
        source_rows.append({"variable": labels[column], "mean_absolute_SHAP": mean_abs})
    ax.axvline(0, color="#5F6368", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered)), [labels[x] for x in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Grouped TreeSHAP contribution to XGBoost log-risk")
    ax.set_title("Distribution of patient-level SHAP contributions\n(orange segment: ± mean absolute SHAP)")
    pd.DataFrame(source_rows).to_csv(folder / "Figure_SHAP_summary_source_data.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, folder, "Figure_SHAP_summary")




