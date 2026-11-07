import os
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

client = MlflowClient()
model_name = config["registry"]["model_name"]

print("MODEL:", model_name)
print("VERSIONS:")
versions = client.search_model_versions(f"name='{model_name}'")
for version in versions:
    print(f"version={version.version} stage={version.current_stage} run_id={version.run_id}")
