import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
db_path = PROJECT_ROOT / "mlflow.db"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}")
mlflow.set_tracking_uri(tracking_uri)

MODEL_NAME = "Video_Game_Sales_Model"


def register_model():
    client = MlflowClient()

    experiment = client.get_experiment_by_name("Video_Game_Sales_Prediction")

    if experiment is None:
        print("Experiment 'Video_Game_Sales_Prediction' not found. Creating it or searching runs...")
        # Check all experiments
        exps = client.search_experiments()
        if not exps:
            print("No experiments found in MLflow.")
            return None
        experiment = exps[0]

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.R2 DESC"]
    )

    if not runs:
        print("No MLflow runs found.")
        return None

    best_run = runs[0]
    run_id = best_run.info.run_id
    r2 = best_run.data.metrics.get("R2")
    model_uri = f"runs:/{run_id}/model"

    print("Best Run ID:", run_id)
    print("Best R2:", r2)

    try:
        client.create_registered_model(MODEL_NAME)
        print(f"Registered model '{MODEL_NAME}' created.")
    except Exception:
        print(f"Registered model '{MODEL_NAME}' already exists.")

    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    print("\nModel registered successfully!")
    print("Model Name:", MODEL_NAME)
    print("Version:", model_version.version)
    print("Run ID:", run_id)
    return model_version


if __name__ == "__main__":
    register_model()
