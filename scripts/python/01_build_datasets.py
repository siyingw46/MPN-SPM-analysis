"""Build patient-level master and ML tables from the raw SEER MPN export.

The script is intentionally deterministic. It preserves the raw export, emits
patient-level outputs, and writes a stepwise cohort flow and patient-level
exclusion/quality logs for every run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


from runtime_paths import work_root
ROOT = work_root()
RAW = ROOT / "data/raw/SEER_MPN_first_primary_person_selection_2000_2023.txt"
ANALYSIS_VERSION = os.environ.get("MPN_ANALYSIS_VERSION", "v0.4")
V04 = ANALYSIS_VERSION == "v0.4"
CONFIG = ROOT / ("config/analysis_config_v0.4.yml" if V04 else "config/analysis_config.yml")
SEED = 20260723 if V04 else 20260718
LANDMARK_MONTHS = 1
ML_YEAR_MIN = 2004
ML_YEAR_MAX = 2023
RUN_ID = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_{ANALYSIS_VERSION}_patient_level_cleaning")
RUN_DIR = ROOT / "logs/runs" / RUN_ID
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed" / ANALYSIS_VERSION if V04 else ROOT / "data/processed"
METADATA = ROOT / "data/metadata" / ANALYSIS_VERSION if V04 else ROOT / "data/metadata"

MPN_CODES = {"9950", "9961", "9962"}
MPN_LABELS = {"9950": "PV", "9961": "PMF", "9962": "ET"}

# Working AML morphology list for the SEER ICD-O-3 export. The list is kept
# explicit and versioned so it can be reviewed before the analysis is frozen.
AML_CODES = {
    "9840", "9861", "9865", "9866", "9867", "9869", "9870", "9871",
    "9872", "9873", "9874", "9891", "9895", "9896", "9897", "9898",
    "9879", "9910", "9911", "9912", "9920", "9931",
}
HEMATOLOGIC_CODE_MIN = 9590
HEMATOLOGIC_CODE_MAX = 9999

RAW_FIELDS = [
    "Patient ID", "Record number recode", "Sequence number",
    "Total number of in situ/malignant tumors for patient", "Year of diagnosis",
    "Age recode with single ages and 85+", "Sex", "Race recode (W, B, AI, API)",
    "COD to site recode", "ICD-O-3 Hist/behav", "Type of Reporting Source",
    "Diagnostic Confirmation", "SEER cause-specific death classification",
    "SEER other cause of death classification", "Chemotherapy recode (yes, no/unk) (2004+)",
    "Vital status recode (study cutoff used)", "Survival months", "Survival months flag",
    "Site recode ICD-O-3/WHO 2008", "Histologic Type ICD-O-3", "Behavior code ICD-O-3",
    "Rural-Urban Continuum Code", "Median household income inflation adj to 2024",
    "Primary Site - labeled", "Total number of benign/borderline tumors for patient",
    "COD to site recode ICD-O-3 2023 Revision Expanded (1999+)",
    "Site recode ICD-O-3 2023 Revision Expanded", "Year of follow-up recode",
    "Site recode ICD-O-3/WHO 2008 (for SIRs)",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def morphology_code(row: dict[str, str]) -> str:
    return row["Histologic Type ICD-O-3"].strip() or row["ICD-O-3 Hist/behav"].split("/", 1)[0]


def behavior(row: dict[str, str]) -> str:
    return row["Behavior code ICD-O-3"].strip().lower()


def sequence_rank(value: str) -> int | None:
    match = re.match(r"(\d+)(?:st|nd|rd|th)", value.strip(), flags=re.I)
    if match:
        return int(match.group(1))
    if value.strip() in {"One primary only", "Only one state registry-defined neoplasm"}:
        return 1
    return None


def parse_age(value: str) -> tuple[float | None, int | None, str]:
    text = value.strip()
    if text == "85+ years":
        return 85.0, 1, "topcoded_85_plus"
    match = re.search(r"(\d+)", text)
    if match:
        age = float(match.group(1))
        return age, int(age >= 85), "parsed"
    return None, None, "unparseable"


def numeric_months(value: str) -> int | None:
    text = value.strip()
    return int(text) if text.isdigit() else None


def is_malignant_nonaml(row: dict[str, str]) -> bool:
    if V04:
        code = morphology_code(row)
        return behavior(row) == "malignant" and not (code.isdigit() and HEMATOLOGIC_CODE_MIN <= int(code) <= HEMATOLOGIC_CODE_MAX)
    return behavior(row) == "malignant" and morphology_code(row) not in MPN_CODES and morphology_code(row) not in AML_CODES


def is_aml(row: dict[str, str]) -> bool:
    if V04:
        code = morphology_code(row)
        return behavior(row) == "malignant" and code.isdigit() and HEMATOLOGIC_CODE_MIN <= int(code) <= HEMATOLOGIC_CODE_MAX
    return behavior(row) == "malignant" and morphology_code(row) in AML_CODES


def is_other_tumor(row: dict[str, str]) -> bool:
    return behavior(row) in {"malignant", "in situ", "benign", "borderline malignancy"}


def clean_label(value: str) -> str:
    return value.strip() if value is not None else ""


def event_record(row: dict[str, str], delta: int, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "time_months_raw": delta,
        "histology_code": morphology_code(row),
        "histology_label": row["ICD-O-3 Hist/behav"],
        "primary_site": row["Primary Site - labeled"],
        "diagnosis_year": int(row["Year of diagnosis"]),
        "sequence_number": row["Sequence number"],
        "sequence_rank": sequence_rank(row["Sequence number"]) or 99,
        "behavior": behavior(row),
        "survival_months": row["Survival months"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)

    with RAW.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_patient[record["Patient ID"]].append(record)

    flow: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []

    def add_flow(step: str, before: int, excluded: int, reason: str = "") -> None:
        flow.append({"step": step, "n_before": before, "n_excluded": excluded, "n_after": before - excluded, "reason": reason})

    n_patients = len(by_patient)
    add_flow("all_unique_patients", n_patients, 0)
    selected: list[tuple[str, list[dict[str, str]], dict[str, str]]] = []
    for patient_id, patient_records in by_patient.items():
        mpn = [r for r in patient_records if morphology_code(r) in MPN_CODES]
        if not mpn:
            exclusions.append({"patient_id": patient_id, "stage": "index_definition", "reason": "no_index_mpn_record"})
            continue
        mpn_sorted = sorted(mpn, key=lambda r: (int(r["Year of diagnosis"]), sequence_rank(r["Sequence number"]) or 99, r["Record number recode"]))
        index = mpn_sorted[0]
        rank = sequence_rank(index["Sequence number"])
        if rank != 1:
            exclusions.append({"patient_id": patient_id, "stage": "index_definition", "reason": "mpn_not_first_primary", "sequence_number": index["Sequence number"]})
            continue
        selected.append((patient_id, patient_records, index))
    add_flow("mpn_first_primary", n_patients, n_patients - len(selected), "MPN absent or not first primary")

    age_eligible: list[tuple[str, list[dict[str, str]], dict[str, str]]] = []
    for patient_id, patient_records, index in selected:
        age_numeric, age_85plus, age_status = parse_age(index["Age recode with single ages and 85+"])
        if age_numeric is None or age_numeric < 18:
            exclusions.append({"patient_id": patient_id, "stage": "adult_definition", "reason": "age_missing_or_under_18", "age_raw": index["Age recode with single ages and 85+"]})
            continue
        age_eligible.append((patient_id, patient_records, index))
    add_flow("adult_age_ge_18", len(selected), len(selected) - len(age_eligible), "Age missing or <18")

    for patient_id, patient_records, index in age_eligible:
        index_surv = numeric_months(index["Survival months"])
        age_numeric, age_85plus, age_status = parse_age(index["Age recode with single ages and 85+"])
        if index_surv is None:
            quality.append({"patient_id": patient_id, "flag": "index_survival_months_unknown", "value": index["Survival months"]})
        index_year = int(index["Year of diagnosis"])
        index_rank = sequence_rank(index["Sequence number"]) or 1
        candidates: list[dict[str, Any]] = []
        for record in patient_records:
            if record is index:
                continue
            rank = sequence_rank(record["Sequence number"])
            if rank is None or rank <= index_rank:
                continue
            year = int(record["Year of diagnosis"])
            if year < index_year:
                quality.append({"patient_id": patient_id, "flag": "post_index_record_precedes_index_year", "value": record["Year of diagnosis"]})
                continue
            survival = numeric_months(record["Survival months"])
            delta = index_surv - survival if index_surv is not None and survival is not None else None
            if delta is None:
                quality.append({"patient_id": patient_id, "flag": "post_index_survival_months_unknown", "value": record["Survival months"]})
                continue
            if delta < 0:
                quality.append({"patient_id": patient_id, "flag": "negative_time_difference", "value": delta})
                continue
            kind = "aml" if is_aml(record) else "nonaml_spm" if is_malignant_nonaml(record) else "other_tumor"
            candidates.append(event_record(record, delta, kind))

        def earliest(kind: str, minimum: int | None = None) -> dict[str, Any] | None:
            pool = [e for e in candidates if e["kind"] == kind and (minimum is None or e["time_months_raw"] >= minimum)]
            return min(pool, key=lambda e: (e["time_months_raw"], e["diagnosis_year"], e["sequence_number"])) if pool else None

        zero_nonaml = earliest("nonaml_spm", 0)
        zero_aml = earliest("aml", 0)
        first_nonaml = earliest("nonaml_spm", LANDMARK_MONTHS)
        first_aml = earliest("aml", LANDMARK_MONTHS)
        if zero_nonaml and zero_nonaml["time_months_raw"] == 0:
            quality.append({"patient_id": patient_id, "flag": "synchronous_nonaml_malignancy_at_zero", "value": zero_nonaml["histology_label"]})
        if zero_aml and zero_aml["time_months_raw"] == 0:
            quality.append({"patient_id": patient_id, "flag": "synchronous_hematologic_malignancy_at_zero" if V04 else "synchronous_aml_at_zero", "value": zero_aml["histology_label"]})

        death_time = index_surv if clean_label(index["Vital status recode (study cutoff used)"]) == "Dead" else None
        baseline_malignancy = bool((zero_nonaml and zero_nonaml["time_months_raw"] == 0) or (zero_aml and zero_aml["time_months_raw"] == 0))
        if baseline_malignancy:
            outcome_type, outcome_time, outcome_kind = -1, 0, "baseline_malignancy"
        else:
            possible = []
            if first_nonaml:
                possible.append((first_nonaml["time_months_raw"], first_nonaml["sequence_rank"], 1, first_nonaml))
            if first_aml:
                possible.append((first_aml["time_months_raw"], first_aml["sequence_rank"], 3, first_aml))
            if death_time is not None:
                # Registered cancer takes priority over death in the same month.
                possible.append((death_time, 999, 2, None))
            if possible:
                outcome_time, _, outcome_type, outcome = min(possible, key=lambda x: (x[0], x[1], x[2]))
                if outcome_type == 1:
                    outcome_kind = outcome["histology_label"]
                elif outcome_type == 3:
                    outcome_kind = outcome["histology_label"]
                else:
                    outcome_kind = ""
            else:
                outcome_time, outcome_type, outcome_kind = index_surv, 0, ""
        # The confirmed primary rule is a minimum observed time of 1 month,
        # not a shifted time origin. Retain the original SEER-derived months.
        outcome_time_for_ml = outcome_time

        row: dict[str, Any] = {f"index_{key}": index[key] for key in RAW_FIELDS if key in index}
        row.update({
            "patient_id": patient_id,
            "index_mpn_subtype": MPN_LABELS[morphology_code(index)],
            "index_mpn_histology_code": morphology_code(index),
            "index_year": index_year,
            "age_numeric": age_numeric,
            "age_85plus": age_85plus,
            "age_parse_status": age_status,
            "index_survival_months_numeric": index_surv,
            "death_before_cancer_event": int(outcome_type == 2),
            "aml_before_nonaml_spm": int(outcome_type == 3) if not V04 else None,
            "hematologic_before_nonhematologic_spm": int(outcome_type == 3) if V04 else None,
            "baseline_malignancy_at_zero": int(baseline_malignancy),
            "event_type": outcome_type,
            "event_spm": int(outcome_type == 1),
            "event_aml": int(outcome_type == 3) if not V04 else None,
            "event_hematologic": int(outcome_type == 3) if V04 else None,
            "event_death": int(outcome_type == 2),
            "time_months_raw": outcome_time,
            "time_months": outcome_time_for_ml,
            "time_years": outcome_time_for_ml / 12 if outcome_time_for_ml is not None else None,
            "first_nonaml_spm_time_raw": first_nonaml["time_months_raw"] if first_nonaml else None,
            "first_aml_time_raw": first_aml["time_months_raw"] if first_aml and not V04 else None,
            "first_hematologic_time_raw": first_aml["time_months_raw"] if first_aml and V04 else None,
            "first_nonaml_spm_histology": first_nonaml["histology_label"] if first_nonaml else "",
            "first_nonaml_spm_site": first_nonaml["primary_site"] if first_nonaml else "",
            "first_aml_histology": first_aml["histology_label"] if first_aml and not V04 else "",
            "first_hematologic_histology": first_aml["histology_label"] if first_aml and V04 else "",
            "later_tumor_record_count": len(candidates),
            "later_malignant_record_count": sum(e["kind"] in {"nonaml_spm", "aml"} for e in candidates),
            "later_nonaml_spm_record_count": sum(e["kind"] == "nonaml_spm" for e in candidates),
            "later_aml_record_count": sum(e["kind"] == "aml" for e in candidates) if not V04 else None,
            "later_hematologic_record_count": sum(e["kind"] == "aml" for e in candidates) if V04 else None,
            "ml_year_eligible": int(ML_YEAR_MIN <= index_year <= ML_YEAR_MAX),
            "ml_landmark_eligible": int(not baseline_malignancy and outcome_time_for_ml is not None and outcome_time_for_ml >= LANDMARK_MONTHS),
            "ml_time_valid": int(outcome_time_for_ml is not None and outcome_time_for_ml >= LANDMARK_MONTHS),
        })
        master_rows.append(row)

    master_columns = list(master_rows[0].keys()) if master_rows else []
    master_path = PROCESSED / ("patient_master_v0.4_2000_2023.csv" if V04 else "patient_master_2000_2023.csv")
    write_csv(master_path, master_rows, master_columns)

    # Machine-learning table: remove synchronous baseline malignancy and rows
    # with unavailable index/event time, then restrict to 2004–2023.
    ml_rows: list[dict[str, Any]] = []
    ml_exclusions: list[dict[str, Any]] = []
    predictor_columns = [
        "patient_id", "age_numeric", "age_85plus", "index_sex", "index_race recode (W, B, AI, API)",
        "index_mpn_subtype", "index_year", "index_rural-urban continuum code",
        "index_median household income inflation adj to 2024", "index_chemotherapy recode (yes, no/unk) (2004+)",
        "time_months", "time_years", "event_type", "event_spm", "event_aml", "event_hematologic", "event_death",
    ]
    # Stable analysis names avoid spaces and make downstream code portable.
    rename = {
        "index_Sex": "sex",
        "index_Race recode (W, B, AI, API)": "race",
        "index_Rural-Urban Continuum Code": "rural_urban",
        "index_Median household income inflation adj to 2024": "median_household_income",
        "index_Chemotherapy recode (yes, no/unk) (2004+)": "chemotherapy_raw",
    }
    for row in master_rows:
        pid = row["patient_id"]
        if not row["ml_year_eligible"]:
            ml_exclusions.append({"patient_id": pid, "stage": "ml_cohort", "reason": "diagnosis_year_outside_2004_2023"})
            continue
        if row["baseline_malignancy_at_zero"]:
            ml_exclusions.append({"patient_id": pid, "stage": "ml_cohort", "reason": "malignancy_at_time_zero"})
            continue
        if row["time_months"] is None:
            ml_exclusions.append({"patient_id": pid, "stage": "ml_cohort", "reason": "time_unknown_or_invalid"})
            continue
        if row["time_months"] < LANDMARK_MONTHS:
            ml_exclusions.append({"patient_id": pid, "stage": "ml_cohort", "reason": "event_or_followup_time_lt_1_month"})
            continue
        ml = {
            "patient_id": pid,
            "age_numeric": row["age_numeric"],
            "age_85plus": row["age_85plus"],
            "sex": row.get("index_Sex", ""),
            "race": row.get("index_Race recode (W, B, AI, API)", ""),
            "mpn_subtype": row["index_mpn_subtype"],
            "diagnosis_year": row["index_year"],
            "rural_urban": row.get("index_Rural-Urban Continuum Code", ""),
            "median_household_income": row.get("index_Median household income inflation adj to 2024", ""),
            "chemotherapy_raw": row.get("index_Chemotherapy recode (yes, no/unk) (2004+)", ""),
            "chemotherapy": int(row.get("index_Chemotherapy recode (yes, no/unk) (2004+)", "") == "Yes"),
            "time_months": row["time_months"],
            "time_years": row["time_years"],
            "event_type": row["event_type"],
            "event_spm": row["event_spm"],
            "event_aml": row["event_aml"],
            "event_hematologic": row["event_hematologic"],
            "event_death": row["event_death"],
        }
        ml_rows.append(ml)

    # Deterministic patient-level 70/30 split, jointly stratified by event,
    # MPN subtype and broad diagnosis era. The split label is metadata, never
    # a predictor.
    rng = np.random.default_rng(SEED)
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(ml_rows):
        year = int(row["diagnosis_year"])
        era = "2004_2008" if year <= 2008 else "2009_2013" if year <= 2013 else "2014_2018" if year <= 2018 else "2019_2023"
        row["diagnosis_era"] = era
        strata[(row["event_type"], row["mpn_subtype"], era)].append(index)
    for indices in strata.values():
        shuffled = np.array(indices, dtype=int)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_development = int(round(0.70 * n))
        if n >= 2:
            n_development = min(max(n_development, 1), n - 1)
        for position, row_index in enumerate(shuffled):
            ml_rows[int(row_index)]["data_split"] = "development" if position < n_development else "test"
            ml_rows[int(row_index)]["split_seed"] = SEED
    ml_path = PROCESSED / ("ml_competing_risk_v0.4_2004_2023.csv" if V04 else "ml_competing_risk_2004_2023.csv")
    ml_columns = list(ml_rows[0].keys()) if ml_rows else []
    write_csv(ml_path, ml_rows, ml_columns)

    all_exclusions = exclusions + ml_exclusions
    write_csv(RUN_DIR / "cohort_flow.csv", flow)
    write_csv(RUN_DIR / "exclusion_log.csv", all_exclusions)
    write_csv(RUN_DIR / "data_quality_flags.csv", quality, ["patient_id", "flag", "value"])
    write_csv(RUN_DIR / "event_code_counts.csv", [{"event_type": k, "n": v} for k, v in Counter(r["event_type"] for r in ml_rows).items()])
    split_balance = []
    for key, count in Counter((r["data_split"], r["event_type"], r["mpn_subtype"], r["diagnosis_era"]) for r in ml_rows).items():
        split_balance.append({"data_split": key[0], "event_type": key[1], "mpn_subtype": key[2], "diagnosis_era": key[3], "n": count})
    write_csv(RUN_DIR / "split_balance.csv", split_balance)

    variable_dictionary = []
    for column in ml_columns:
        variable_dictionary.append({
            "analysis_variable": column,
            "role": "outcome" if column in {"time_months", "time_years", "event_type", "event_spm", "event_aml", "event_hematologic", "event_death"} else "predictor_or_linkage",
            "source_or_derivation": "derived from index record" if column not in {"time_months", "time_years", "event_type", "event_spm", "event_aml", "event_hematologic", "event_death"} else "derived event/time rule",
            "notes": "Patient ID is linkage-only; chemotherapy Yes=1 and No/Unknown/Blank=0 in the 2004+ ML cohort." if column == "patient_id" or column == "chemotherapy" else "",
        })
    write_csv(METADATA / "variable_dictionary_ml_v0.1.csv", variable_dictionary)

    manifest = {
        "run_id": RUN_ID,
        "seed": SEED,
        "landmark_months": LANDMARK_MONTHS,
        "raw_path": str(RAW),
        "raw_sha256": sha256(RAW),
        "raw_records": len(records),
        "raw_unique_patients": n_patients,
        "patient_master_path": str(master_path),
        "patient_master_rows": len(master_rows),
        "ml_path": str(ml_path),
        "ml_rows": len(ml_rows),
        "ml_event_counts": dict(Counter(r["event_type"] for r in ml_rows)),
        "quality_flag_rows": len(quality),
        "exclusion_rows": len(all_exclusions),
        "aml_codes": sorted(AML_CODES),
        "hematologic_definition": "malignant ICD-O-3 morphology 9590-9999" if V04 else None,
        "status": "complete",
    }
    (RUN_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(CONFIG, RUN_DIR / "config_snapshot.yml")
    (RUN_DIR / "software_environment.txt").write_text("Python/pandas deterministic cleaning script; see script source and manifest.\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
