import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DRIFT_PATH = PROJECT_ROOT / "outputs/drift/data_drift_report.csv"
PREDICTION_DRIFT_PATH = PROJECT_ROOT / "outputs/drift/prediction_drift_report.csv"
CONCEPT_DRIFT_PATH = PROJECT_ROOT / "outputs/drift/concept_drift_report.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs/drift/drift_analysis_report.csv"


def ensure_reports():
    if not (PROJECT_ROOT / "data/monitoring/drift/evolved_data.csv").exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_create_evolved_dataset.py")], cwd=PROJECT_ROOT)
    if not (PROJECT_ROOT / "data/monitoring/drift/evolved_labeled_data.csv").exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_create_concept_drift_dataset.py")], cwd=PROJECT_ROOT)

    if not DATA_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_data_drift.py")], cwd=PROJECT_ROOT)
    if not PREDICTION_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_prediction_drift.py")], cwd=PROJECT_ROOT)
    if not CONCEPT_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_concept_drift.py")], cwd=PROJECT_ROOT)


def main():
    print("=" * 70)
    print("COMPREHENSIVE DRIFT ANALYSIS REPORT")
    print("=" * 70)

    ensure_reports()

    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH)
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)

    drifted_features = data_drift[data_drift["status"] == "DRIFT"]
    pred_row = prediction_drift.iloc[0]

    degraded_concept = concept_drift[(concept_drift["metric"] == "r2") & (concept_drift["change"] < 0)]

    summary = [
        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": len(drifted_features),
            "status": "DRIFT" if len(drifted_features) > 0 else "OK",
            "details": ", ".join(drifted_features["feature"].tolist()) if len(drifted_features) > 0 else "None",
        },
        {
            "category": "Prediction Drift",
            "metric": "Prediction distribution",
            "value": pred_row["prediction_distribution_ks"],
            "status": pred_row["prediction_distribution_status"],
            "details": f"KS statistic = {pred_row['prediction_distribution_ks']}",
        },
        {
            "category": "Concept Drift",
            "metric": "R2 Score degradation",
            "value": float(concept_drift[concept_drift["metric"] == "r2"]["change"].iloc[0]),
            "status": "DRIFT" if len(degraded_concept) > 0 else "OK",
            "details": "R2 change on evolved data",
        },
    ]

    summary_df = pd.DataFrame(summary)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nComprehensive report saved to: {OUTPUT_PATH}")
    return True


if __name__ == "__main__":
    main()
