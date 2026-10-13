"""Build the locked test-prediction bundle used by the 5,000-resample analysis."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


from runtime_paths import work_root
ROOT = work_root()
SOURCE = ROOT / "output/models/model_development_v0.4/final_test_evaluation"
REVISION = ROOT / "revision"
DEEPHIT = REVISION / "05_deephit_retraining"
TARGET = REVISION / "04_bootstrap5000/model_root/final_test_evaluation"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE / "test_all_models_cif.npz", allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    with np.load(DEEPHIT / "deephit_retrained_predictions.npz", allow_pickle=False) as retrained:
        if not np.array_equal(arrays["patient_id"], retrained["test_patient_id"]):
            raise RuntimeError("Retuned DeepHit patient IDs do not match the locked test set")
        if not np.array_equal(arrays["eval_times"], retrained["eval_times"]):
            raise RuntimeError("Retuned DeepHit evaluation grid does not match the locked bundle")
        arrays["deephit_cr"] = retrained["test_cif"].astype(np.float32)
    np.savez_compressed(TARGET / "test_all_models_cif.npz", **arrays)

    table = pd.read_csv(SOURCE / "table_test_main_comparison.csv")
    summary = json.loads((DEEPHIT / "deephit_retraining_summary.json").read_text(encoding="utf-8"))
    metrics = summary["locked_test_metrics"]
    row = table.index[table.model.eq("deephit_cr")]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one DeepHit row in the locked model table")
    row = row[0]
    for column, value in metrics.items():
        if column in table.columns:
            table.loc[row, column] = value
    table.loc[row, "development_role_locked_before_test"] = "comparator_retrained_without_test_tuning"
    table.to_csv(TARGET / "table_test_main_comparison.csv", index=False, encoding="utf-8-sig")

    shutil.copy2(DEEPHIT / "deephit_retraining_summary.json", TARGET / "deephit_retraining_summary.json")
    metadata = {
        "status": "PASS",
        "prediction_bundle": "All original locked predictions, with deephit_cr replaced by the development-selected retrained DeepHit prediction.",
        "test_set_used_for_tuning": False,
        "bootstrap_replicates_planned": 5000,
    }
    (TARGET / "bootstrap5000_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(TARGET)


if __name__ == "__main__":
    main()
