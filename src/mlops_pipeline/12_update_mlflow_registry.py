import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V2_MODEL_PATH = PROJECT_ROOT / "models/retrained_model_v2.pkl"
V2_PREPROCESSOR_PATH = PROJECT_ROOT / "models/retrained_preprocessor_v2.pkl"
RETRAINING_REPORT_PATH = PROJECT_ROOT / "outputs/retraining/retraining_report.json"
COMPARISON_REPORT_PATH = PROJECT_ROOT / "outputs/retraining/model_comparison_report.json"
REGISTRY_REPORT_PATH = PROJECT_ROOT / "outputs/retraining/mlflow_registry_update_report.json"


def configure_mlflow(config):
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


def main():
    print("=" * 70)
    print("STAGE 12: REGISTER V2 CANDIDATE IN MLFLOW")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    with open(RETRAINING_REPORT_PATH, "r", encoding="utf-8") as f:
        retraining_report = json.load(f)

    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        comparison_report = json.load(f)

    run_id = retraining_report["run_id"]
    model_v2 = joblib.load(V2_MODEL_PATH)

    print(f"[INFO] Registering V2 model from run {run_id} into '{model_name}'...")
    with mlflow.start_run(run_id=run_id):
        reg = mlflow.sklearn.log_model(
            sk_model=model_v2,
            name="gradient_boosting_model_v2",
            registered_model_name=model_name,
        )

    all_versions = client.search_model_versions(f"name='{model_name}'")
    latest_v2 = max(all_versions, key=lambda v: int(v.version))
    print(f"[SUCCESS] Created Model Version {latest_v2.version}")

    should_promote = comparison_report.get("recommendation") == "PROMOTE_V2"

    if should_promote:
        print(f"[INFO] Promoting Version {latest_v2.version} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v2.version,
            stage="Production",
            archive_existing_versions=True,
        )
        stage_assigned = "Production"
    else:
        print(f"[INFO] Transitioning Version {latest_v2.version} to Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v2.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        stage_assigned = "Staging"

    report = {
        "status": "success",
        "model_name": model_name,
        "v2_version": str(latest_v2.version),
        "stage": stage_assigned,
        "run_id": run_id,
        "recommendation": comparison_report.get("recommendation"),
    }

    REGISTRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] MLflow registry update report saved to {REGISTRY_REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
