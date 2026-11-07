import os
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# PROJECT PATH
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

# CHECK MODEL
print("Model path:", MODEL_PATH)
if not MODEL_PATH.exists():
    print("ERROR: Model file not found!")
    print("Expected:", MODEL_PATH)
    exit(1)

print("Model found successfully!")

# LOAD MODEL
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# MLFLOW TRACKING SERVER
db_path = BASE_DIR / "mlflow.db"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Video_Game_Sales_Prediction")

# START RUN
with mlflow.start_run(run_name="GradientBoosting_Tracking"):
    # Parameters
    mlflow.log_param("model", "Gradient Boosting Regressor")
    mlflow.log_param("dataset", "Video Game Sales")
    mlflow.log_param("features", "Platform, Year, Genre, Publisher")
    mlflow.log_param("target", "Global_Sales")
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)

    # Hyperparameters
    if hasattr(model, "named_steps"):
        model_step = model.named_steps.get("model")
        if model_step is not None:
            if hasattr(model_step, "n_estimators"):
                mlflow.log_param("n_estimators", model_step.n_estimators)
            if hasattr(model_step, "learning_rate"):
                mlflow.log_param("learning_rate", model_step.learning_rate)
            if hasattr(model_step, "max_depth"):
                mlflow.log_param("max_depth", model_step.max_depth)

    # Evaluation on test set
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np

    raw_path = BASE_DIR / "data" / "raw" / "vgsales.csv"
    df = pd.read_csv(raw_path).dropna(subset=["Platform", "Year", "Genre", "Publisher", "Global_Sales"])
    X = df[["Platform", "Year", "Genre", "Publisher"]]
    y = df["Global_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    mlflow.log_metrics({"R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse})

    # Log model
    mlflow.sklearn.log_model(sk_model=model, name="model")

    # Run information
    run_id = mlflow.active_run().info.run_id

    print("MLflow Tracking Successful!")
    print("Experiment :", "Video_Game_Sales_Prediction")
    print("Run ID     :", run_id)
    print("R2 Score   :", r2)
    print("MSE        :", mse)
    print("RMSE       :", rmse)
    print("Model      :", "Gradient Boosting Regressor")
