import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 5: TRAINING + MLFLOW + MODEL REGISTRY")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}"))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model_cfg = config["model"]
    params = {
        "n_estimators": model_cfg["n_estimators"],
        "learning_rate": model_cfg["learning_rate"],
        "max_depth": model_cfg["max_depth"],
        "random_state": model_cfg["random_state"],
    }

    with mlflow.start_run(run_name="GradientBoosting_Stage5") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "GradientBoostingRegressor")
        mlflow.log_param("lab", "ML Pipeline")

        print("[INFO] Training GradientBoosting model...")
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating model on test set...")
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))

        metrics = {"R2": r2, "r2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}
        mlflow.log_metrics(metrics)

        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, PROJECT_ROOT / config["paths"]["model"])

        preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
        if preprocessor_path.exists():
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="gradient_boosting_model",
            registered_model_name=config["registry"]["model_name"],
        )

        print(
            f"[SUCCESS] R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}"
        )
        print(f"[SUCCESS] Model saved to {config['paths']['model']}")
        print(f"[SUCCESS] Model registered as '{config['registry']['model_name']}'")
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")

    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
