import json
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 4: PROCESSED OUTPUT VALIDATION")
    print("=" * 70)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    errors = []

    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")

    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "errors": errors,
    }

    report_path = artifacts_dir / "preprocessing_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for err in errors:
            print(f"  - {err}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
