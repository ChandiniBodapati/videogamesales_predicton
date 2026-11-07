import json
import os
from pathlib import Path
import subprocess
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPDATED_DATASET_PATH = PROJECT_ROOT / "data/processed/updated_training_data.csv"
TEST_FEATURES_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
TEST_LABELS_PATH = PROJECT_ROOT / "data/processed/y_test.npy"

MODEL_V2_PATH = PROJECT_ROOT / "models/retrained_model_v2.pkl"
PREPROCESSOR_V2_PATH = PROJECT_ROOT / "models/retrained_preprocessor_v2.pkl"
REPORT_PATH = PROJECT_ROOT / "outputs/retraining/retraining_report.json"


def configure_mlflow(config):
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


def retrain():
    print("=" * 70)
    print("STAGE 12: MODEL RETRAINING (V2)")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if not UPDATED_DATASET_PATH.exists():
        print("[INFO] Updated dataset not found. Generating post-drift data...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/mlops_pipeline/12_create_updated_dataset.py")], cwd=PROJECT_ROOT)

    df_train = pd.read_csv(UPDATED_DATASET_PATH)
    X_test_raw = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    features = config["data"]["features"]
    target = config["data"]["target_column"]

    X_train = df_train[features].copy()
    y_train = df_train[target].to_numpy()

    # Preprocessing pipeline
    categorical_features = ["Platform", "Genre", "Publisher"]
    numerical_features = ["Year"]

    # Impute missing values
    X_train["Year"] = pd.to_numeric(X_train["Year"], errors="coerce").fillna(2006.0)
    X_train["Publisher"] = X_train["Publisher"].fillna("Unknown")
    X_train["Genre"] = X_train["Genre"].fillna("Other")

    X_test_clean = X_test_raw.copy()
    X_test_clean["Year"] = pd.to_numeric(X_test_clean["Year"], errors="coerce").fillna(2006.0)
    X_test_clean["Publisher"] = X_test_clean["Publisher"].fillna("Unknown")
    X_test_clean["Genre"] = X_test_clean["Genre"].fillna("Other")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test_clean)

    with mlflow.start_run(run_name="GradientBoosting_V2_Retrained") as run:
        params = {
            "n_estimators": 120,
            "learning_rate": 0.08,
            "max_depth": 4,
            "random_state": 42,
        }
        mlflow.log_params(params)
        mlflow.log_param("version", "v2")

        model_v2 = GradientBoostingRegressor(**params)
        model_v2.fit(X_train_trans, y_train)

        preds = model_v2.predict(X_test_trans)
        r2 = float(r2_score(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mse))

        metrics = {"r2": r2, "R2": r2, "mse": mse, "mae": mae, "rmse": rmse}
        mlflow.log_metrics(metrics)

        MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_v2, MODEL_V2_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)

        mlflow.log_artifact(str(PREPROCESSOR_V2_PATH), artifact_path="preprocessing_pipeline_v2")

        report = {
            "model_version": "v2",
            "run_id": run.info.run_id,
            "training_samples": len(X_train),
            "test_samples": len(X_test_clean),
            "metrics": metrics,
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Retrained Model V2 R2: {r2:.4f} | RMSE: {rmse:.4f}")
        print(f"[SUCCESS] Saved model to: {MODEL_V2_PATH}")
        print(f"[SUCCESS] Saved preprocessor to: {PREPROCESSOR_V2_PATH}")
        print(f"[SUCCESS] Report saved to: {REPORT_PATH}")
    return True


if __name__ == "__main__":
    retrain()
