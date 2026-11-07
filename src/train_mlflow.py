import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_and_track(run_name="GradientBoosting_Lab4", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}"))
    mlflow.set_experiment("Video_Game_Sales_Prediction")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "GradientBoostingRegressor")

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))

        metrics = {"R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}
        mlflow.log_metrics(metrics)

        print(f"[SUCCESS] Metrics logged: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

        # Diagnostic plot: Actual vs Predicted
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, alpha=0.3, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax.set_xlabel("Actual Global Sales (Millions)")
        ax.set_ylabel("Predicted Global Sales (Millions)")
        ax.set_title(f"Actual vs Predicted - {run_name}")
        plot_path = artifacts_dir / "actual_vs_predicted.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # Save model
        mlflow.sklearn.log_model(sk_model=model, name="model")

        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "final_model.pkl")

        print(f"Run '{run_name}' successfully tracked! Run ID: {run.info.run_id}")
        return True


if __name__ == "__main__":
    train_and_track()
