from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_SCRIPT = PROJECT_ROOT / "src/mlops_pipeline/12_drift_analysis.py"
DRIFT_REPORT = PROJECT_ROOT / "outputs/drift/drift_analysis_report.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data/raw/new_vgsales.csv"
PIPELINE_SCRIPT = PROJECT_ROOT / "src/mlops_pipeline/vgsales_pipeline.py"
TRIGGER_REPORT = PROJECT_ROOT / "outputs/monitoring/monitor_trigger_report.json"
DATA_LOAD_REPORT = PROJECT_ROOT / "artifacts/data_load_report.json"


def run_drift_analysis():
    print("[INFO] Running drift analysis...")
    result = subprocess.run([sys.executable, str(DRIFT_SCRIPT)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def detect_drift():
    if not DRIFT_REPORT.exists():
        return False
    df = pd.read_csv(DRIFT_REPORT)
    for col in df.columns:
        vals = df[col].astype(str).str.strip().str.upper()
        if vals.eq("DRIFT").any():
            return True
    return False


def main():
    print("=" * 70)
    print("LAB 13 MONITORING + RETRAINING TRIGGER")
    print("=" * 70)

    drift_ok = run_drift_analysis()
    if not drift_ok:
        print("[ERROR] Drift analysis failed.")
        return False

    drift_detected = detect_drift()
    new_data_present = NEW_DATA_PATH.exists()

    print(f"[INFO] Drift detected     : {drift_detected}")
    print(f"[INFO] New dataset present: {new_data_present}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": drift_detected,
        "new_data_present": new_data_present,
        "pipeline_triggered": False,
        "status": "idle",
    }

    if drift_detected and new_data_present:
        print("[INFO] Drift detected and new dataset is present. Triggering MLOps pipeline...")
        pipeline_res = subprocess.run([sys.executable, str(PIPELINE_SCRIPT)], cwd=PROJECT_ROOT)
        report["pipeline_triggered"] = True
        report["pipeline_exit_code"] = pipeline_res.returncode
        report["status"] = "retraining_triggered" if pipeline_res.returncode == 0 else "pipeline_failed"
    elif drift_detected and not new_data_present:
        print("[INFO] Drift detected, but awaiting new data ingestion batch (new_vgsales.csv).")
        report["status"] = "drift_waiting_for_data"
    else:
        print("[INFO] System healthy. No retraining trigger required.")
        report["status"] = "healthy_no_drift"

    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRIGGER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Trigger report saved to: {TRIGGER_REPORT}")
    return True


if __name__ == "__main__":
    main()
