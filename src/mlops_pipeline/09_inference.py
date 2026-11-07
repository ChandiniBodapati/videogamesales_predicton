import os
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe(df, config):
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    # Fill missing values consistently
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        median_year = df["Year"].median()
        df["Year"] = df["Year"].fillna(median_year if pd.notnull(median_year) else 2006.0)

    if "Publisher" in df.columns:
        df["Publisher"] = df["Publisher"].fillna("Unknown")

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("Other")

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    features = config["data"]["features"]
    available_features = [col for col in features if col in df.columns]
    return df[available_features]


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    try:
        print(f"[INFO] Loading Production model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"[ERROR] Could not load Production model from MLflow: {exc}")
        print("[ERROR] Ensure a model is promoted to Production first.")
        return False

    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
    raw_data_path = PROJECT_ROOT / config["data"]["raw_path"]

    if not preprocessor_path.exists():
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}")
        return False

    if not raw_data_path.exists():
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    sample = df.sample(n=5, random_state=42)
    X_sample = clean_dataframe(sample, config)
    X_sample_trans = preprocessor.transform(X_sample)

    predictions = model.predict(X_sample_trans)

    result = sample.copy()
    result["Predicted_Global_Sales"] = np.round(predictions, 4)

    display_cols = ["Rank", "Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales", "Predicted_Global_Sales"]
    present_cols = [c for c in display_cols if c in result.columns]

    print("\n--- ML Pipeline Production Inference Results ---")
    print(result[present_cols].to_string(index=False))
    print("------------------------------------------------\n")

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
