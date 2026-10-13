"""Audit v0.3 nested-CV fold checkpoints before ensemble construction.

The audit is deliberately independent of downstream model comparison.  It
checks checkpoint completeness, patient coverage within each repeat, array
shape/range/monotonicity, matching JSON metadata, and candidate-fit failures.
"""

from __future__ import annotations

import argparse
import os
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


from runtime_paths import work_root
ROOT = work_root()
MODEL_ROOT = Path(os.environ.get("MPN_MODEL_ROOT", ROOT / "output/models/model_development_v0.4"))
DATA_PATH = Path(os.environ.get("MPN_DATA_PATH", ROOT / "data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv"))
OOF_DIR = MODEL_ROOT / "nested_oof_folds"
AUDIT_DIR = MODEL_ROOT / "audit"
SIGNATURE = os.environ.get("MPN_RUN_SIGNATURE", "v0.3_nested5x3_inner3_20260718")
EXPECTED_REPEATS = 3
EXPECTED_FOLDS = 5
EXPECTED_TIMES = np.arange(12.0, 61.0, 1.0)
ALL_FAMILIES = [
    "cs_cox",
    "fine_gray",
    "elastic_net_cox",
    "linear_survival_svm",
    "kernel_survival_svm",
    "rsf",
    "extra_survival_trees",
    "xgboost_cox",
    "deephit_cr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        default=",".join(ALL_FAMILIES),
        help="Comma-separated model family ids",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with an error if any requested family lacks all 15 folds",
    )
    return parser.parse_args()


def audit_family(family: str, development_ids: np.ndarray) -> tuple[dict, list[dict]]:
    family_dir = OOF_DIR / family
    problems: list[str] = []
    warnings: list[str] = []
    fold_rows: list[dict] = []
    coverage: dict[int, list[np.ndarray]] = {repeat: [] for repeat in range(1, 4)}
    parameter_counter: Counter[str] = Counter()
    seen_pairs: set[tuple[int, int]] = set()

    for npz_path in sorted(family_dir.glob("repeat_*_fold_*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            problems.append(f"missing JSON metadata: {json_path.name}")
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        repeat = int(metadata["repeat"])
        fold = int(metadata["fold"])
        pair = (repeat, fold)
        if pair in seen_pairs:
            problems.append(f"duplicate repeat/fold: {pair}")
        seen_pairs.add(pair)

        with np.load(npz_path, allow_pickle=False) as saved:
            signature = str(saved["signature"].item())
            saved_family = str(saved["family"].item())
            validation_index = saved["validation_index"].astype(int)
            patient_id = saved["patient_id"]
            eval_times = saved["eval_times"].astype(float)
            prediction = saved["target_cif"].astype(float)

        if signature != SIGNATURE or metadata.get("signature") != SIGNATURE:
            problems.append(f"signature mismatch: {npz_path.name}")
        if saved_family != family or metadata.get("family") != family:
            problems.append(f"family mismatch: {npz_path.name}")
        if not np.array_equal(eval_times, EXPECTED_TIMES):
            problems.append(f"evaluation-time mismatch: {npz_path.name}")
        expected_shape = (len(validation_index), len(EXPECTED_TIMES))
        if prediction.shape != expected_shape:
            problems.append(
                f"prediction shape {prediction.shape} != {expected_shape}: {npz_path.name}"
            )
        if len(np.unique(validation_index)) != len(validation_index):
            problems.append(f"duplicate validation indices: {npz_path.name}")
        if np.any(validation_index < 0) or np.any(validation_index >= len(development_ids)):
            problems.append(f"out-of-range validation index: {npz_path.name}")
        elif not np.array_equal(patient_id.astype(str), development_ids[validation_index].astype(str)):
            problems.append(f"patient/index mismatch: {npz_path.name}")
        if not np.isfinite(prediction).all():
            problems.append(f"non-finite prediction: {npz_path.name}")
        if np.nanmin(prediction) < -1e-6 or np.nanmax(prediction) > 1.0 + 1e-6:
            problems.append(f"prediction outside [0,1]: {npz_path.name}")
        if np.nanmin(np.diff(prediction, axis=1)) < -1e-6:
            problems.append(f"non-monotone CIF: {npz_path.name}")
        if metadata.get("test_partition_used") is not False:
            problems.append(f"test-use flag is not false: {npz_path.name}")
        if metadata.get("cif_validation", {}).get("status") != "PASS":
            problems.append(f"CIF validation failed: {npz_path.name}")

        failures = [
            row.get("failure")
            for row in metadata.get("candidate_results", [])
            if row.get("failure")
        ]
        if failures:
            # A rejected hyperparameter candidate is not a family-level model
            # failure when at least one valid candidate remains and the selected
            # outer model passes all CIF checks.  Preserve it as an audit warning.
            warnings.append(f"candidate rejection(s) in {npz_path.name}: {failures}")
        parameters = json.dumps(
            metadata.get("selected_parameters", {}), sort_keys=True, ensure_ascii=False
        )
        parameter_counter[parameters] += 1
        coverage.setdefault(repeat, []).append(validation_index)
        fold_rows.append(
            {
                "family": family,
                "repeat": repeat,
                "fold": fold,
                "outer_ibs": float(metadata["outer_ibs"]),
                "elapsed_seconds": float(metadata["elapsed_seconds"]),
                "selected_parameters": parameters,
                "validation_rows": len(validation_index),
                "candidate_rejection_count": len(failures),
            }
        )

    expected_pairs = {
        (repeat, fold)
        for repeat in range(1, EXPECTED_REPEATS + 1)
        for fold in range(1, EXPECTED_FOLDS + 1)
    }
    missing_pairs = sorted(expected_pairs - seen_pairs)
    extra_pairs = sorted(seen_pairs - expected_pairs)
    if extra_pairs:
        problems.append(f"unexpected repeat/fold pairs: {extra_pairs}")

    coverage_ok = True
    if not missing_pairs:
        expected_index = np.arange(len(development_ids))
        for repeat in range(1, EXPECTED_REPEATS + 1):
            combined = np.concatenate(coverage[repeat]) if coverage[repeat] else np.array([])
            if not np.array_equal(np.sort(combined), expected_index):
                problems.append(f"development-index coverage failed in repeat {repeat}")
                coverage_ok = False
    else:
        coverage_ok = False

    outer_ibs = np.asarray([row["outer_ibs"] for row in fold_rows], dtype=float)
    elapsed = np.asarray([row["elapsed_seconds"] for row in fold_rows], dtype=float)
    complete = not missing_pairs and len(fold_rows) == EXPECTED_REPEATS * EXPECTED_FOLDS
    summary = {
        "family": family,
        "status": "PASS" if complete and coverage_ok and not problems else "INCOMPLETE" if missing_pairs and not problems else "FAIL",
        "completed_folds": len(fold_rows),
        "expected_folds": EXPECTED_REPEATS * EXPECTED_FOLDS,
        "missing_pairs": missing_pairs,
        "coverage_ok": coverage_ok,
        "problem_count": len(problems),
        "problems": problems,
        "warning_count": len(warnings),
        "warnings": warnings,
        "mean_outer_ibs": float(np.mean(outer_ibs)) if len(outer_ibs) else None,
        "sd_outer_ibs": float(np.std(outer_ibs, ddof=1)) if len(outer_ibs) > 1 else None,
        "mean_elapsed_seconds": float(np.mean(elapsed)) if len(elapsed) else None,
        "parameter_selection_counts": dict(parameter_counter),
    }
    return summary, fold_rows


def main() -> None:
    args = parse_args()
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = sorted(set(families) - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")

    frame = pd.read_csv(DATA_PATH, usecols=["patient_id", "data_split"], low_memory=False)
    development_ids = (
        frame.loc[frame["data_split"].eq("development"), "patient_id"]
        .reset_index(drop=True)
        .to_numpy()
    )
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    fold_rows: list[dict] = []
    for family in families:
        summary, rows = audit_family(family, development_ids)
        summaries.append(summary)
        fold_rows.extend(rows)
        print(
            f"{family}: {summary['status']} | folds={summary['completed_folds']}/15 | "
            f"mean IBS={summary['mean_outer_ibs']} | problems={summary['problem_count']}",
            flush=True,
        )

    (AUDIT_DIR / "oof_audit_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(
        AUDIT_DIR / "oof_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary_table = pd.DataFrame(
        [
            {
                key: row[key]
                for key in [
                    "family",
                    "status",
                    "completed_folds",
                    "expected_folds",
                    "coverage_ok",
                    "problem_count",
                    "warning_count",
                    "mean_outer_ibs",
                    "sd_outer_ibs",
                    "mean_elapsed_seconds",
                ]
            }
            for row in summaries
        ]
    )
    summary_table.to_csv(
        AUDIT_DIR / "oof_audit_summary.csv", index=False, encoding="utf-8-sig"
    )
    if args.require_complete and any(row["status"] != "PASS" for row in summaries):
        raise SystemExit("OOF audit did not pass for every requested family")


if __name__ == "__main__":
    main()
