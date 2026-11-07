import json
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_report():
    print("[INFO] Generating Model Registry Report...")

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}"))

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        print("[WARNING] No Production model found to generate report for.")
        if versions:
            prod_versions = [versions[0]]
        else:
            return False

    prod_model = prod_versions[0]
    run = client.get_run(prod_model.run_id)

    report = {
        "model_name": model_name,
        "version": prod_model.version,
        "stage": prod_model.current_stage,
        "run_id": prod_model.run_id,
        "source": prod_model.source,
        "parameters": run.data.params,
        "metrics": run.data.metrics,
        "creation_timestamp": prod_model.creation_timestamp,
    }

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "production_model_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Report successfully generated for Version {prod_model.version}")
    print(f"[INFO] Saved to: {report_path}")
    return True


if __name__ == "__main__":
    generate_report()
