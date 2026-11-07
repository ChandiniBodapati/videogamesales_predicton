import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    model_name = config["registry"]["model_name"]
    client = MlflowClient()

    print(f"Model: {model_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print()
    print("VERSION | STAGE | RUN_ID")

    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    for version in versions:
        print(f"{version.version} | {version.current_stage} | {version.run_id}")


if __name__ == "__main__":
    main()
