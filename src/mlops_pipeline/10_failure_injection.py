import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


def run_script(script_path):
    script = PROJECT_ROOT / script_path
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    return result.returncode


def restore_file(backup_path, original_path):
    if os.path.exists(original_path):
        os.remove(original_path)
    os.replace(backup_path, original_path)


def test_missing_data(config):
    print("\n[LAB 9] Scenario 1: Missing data")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")

    os.replace(data_path, backup_path)
    try:
        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "missing_data",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_invalid_datatype(config):
    print("\n[LAB 9] Scenario 2: Invalid datatype")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")

    shutil.copy2(data_path, backup_path)
    try:
        df = pd.read_csv(data_path)
        df["Global_Sales"] = df["Global_Sales"].astype(object)
        df.loc[0, "Global_Sales"] = "INVALID_NUMBER"
        df.to_csv(data_path, index=False)

        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "invalid_datatype",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_schema_mismatch(config):
    print("\n[LAB 9] Scenario 3: Schema mismatch")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")

    shutil.copy2(data_path, backup_path)
    try:
        df = pd.read_csv(data_path)
        df = df.drop(columns=["Genre"])
        df.to_csv(data_path, index=False)

        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "schema_mismatch",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_preprocessing_inconsistency(config):
    print("\n[LAB 9] Scenario 4: Preprocessing inconsistency")
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    path = processed_dir / "X_train_final.npy"
    backup_path = Path(str(path) + ".lab9_backup")

    if not path.exists():
        print("[ERROR] Processed training output is missing. Run the normal pipeline first.")
        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "NOT_TESTED",
            "passed": False,
        }

    shutil.copy2(path, backup_path)
    try:
        X_train = np.load(path)
        X_train[0, 0] = np.nan
        np.save(path, X_train)

        exit_code = run_script("src/mlops_pipeline/04_validate_outputs.py")
        passed = exit_code != 0
        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, path)


def test_missing_model_with_recovery(config):
    print("\n[LAB 9] Scenario 5: Missing model artifact + recovery")
    model_path = PROJECT_ROOT / config["paths"]["model"]
    model_name = config["registry"]["model_name"]

    if not model_path.exists():
        print("[ERROR] Local model is missing before the test. Run the normal pipeline first.")
        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "NOT_TESTED",
            "passed": False,
        }

    backup_path = Path(str(model_path) + ".lab9_backup")
    shutil.copy2(model_path, backup_path)
    os.remove(model_path)

    try:
        failure_exit_code = run_script("src/mlops_pipeline/06_model_evaluation.py")
        failure_detected = failure_exit_code != 0

        configure_mlflow(config)
        production_uri = f"models:/{model_name}/Production"
        print(f"[INFO] Recovering model from {production_uri}...")

        production_model = mlflow.sklearn.load_model(production_uri)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(production_model, model_path)

        recovery_exit_code = run_script("src/mlops_pipeline/06_model_evaluation.py")
        recovered = recovery_exit_code == 0

        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "RECOVERED" if (failure_detected and recovered) else "RECOVERY_FAILED",
            "failure_detected": failure_detected,
            "recovery_succeeded": recovered,
            "passed": failure_detected and recovered,
        }
    finally:
        restore_file(backup_path, model_path)


def run_failure_tests():
    print("[INFO] ==================================================")
    print("[INFO] Failure Tests: Pipeline Failure Handling and Recovery")
    print("[INFO] ==================================================")

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repetitions = int(config.get("lab9", {}).get("repetitions", 3))
    all_runs = []

    tests = [
        test_missing_data,
        test_invalid_datatype,
        test_schema_mismatch,
        test_preprocessing_inconsistency,
        test_missing_model_with_recovery,
    ]

    for repetition in range(1, repetitions + 1):
        print(f"\n{'=' * 70}\n[INFO] Lab 9 Repetition {repetition}/{repetitions}\n{'=' * 70}")
        repetition_results = []
        for test in tests:
            repetition_results.append(test(config))

        all_runs.append(
            {
                "repetition": repetition,
                "results": repetition_results,
                "passed": all(item["passed"] for item in repetition_results),
            }
        )

    report = {
        "stage": "Failure Handling and Recovery",
        "repetitions": repetitions,
        "runs": all_runs,
        "overall_status": "PASSED" if all(run["passed"] for run in all_runs) else "FAILED",
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / config["paths"]["failure_report"]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if report["overall_status"] == "PASSED":
        print("\n[SUCCESS] All Lab 9 failure and recovery tests PASSED.")
        print(f"[INFO] Report saved to {report_path}")
        return True

    print("\n[ERROR] One or more Lab 9 scenarios FAILED.")
    print(f"[INFO] Report saved to {report_path}")
    return False


if __name__ == "__main__":
    passed = run_failure_tests()
    sys.exit(0 if passed else 1)
