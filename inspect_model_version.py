from pathlib import Path
import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent
db_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

client = MlflowClient()
model_name = "Video_Game_Sales_Production_Model"

versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version: {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id} | Source: {v.source}")
