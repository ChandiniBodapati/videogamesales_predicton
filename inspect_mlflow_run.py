from pathlib import Path
import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent
db_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

client = MlflowClient()
exp = client.get_experiment_by_name("Video_Game_Sales_Prediction")

if exp:
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    if runs:
        run = runs[0]
        run_id = run.info.run_id
        print("EXPERIMENT ID:", exp.experiment_id)
        print("RUN ID       :", run_id)
        print("STATUS       :", run.info.status)
        print("METRICS      :", run.data.metrics)
        print("PARAMS       :", run.data.params)
        print("\nARTIFACTS:")
        for art in client.list_artifacts(run_id):
            print(f" - {art.path} (is_dir={art.is_dir})")
    else:
        print("No runs found in experiment.")
else:
    print("Experiment not found.")
