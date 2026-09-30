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
    exit()

print("Model found successfully!")



# LOAD MODEL
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")



# MLFLOW TRACKING SERVER

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Video_Game_Sales_Prediction")


# START RUN


with mlflow.start_run():

    # Parameters
    mlflow.log_param(
        "model",
        "Gradient Boosting Regressor"
    )

    mlflow.log_param(
        "dataset",
        "Video Game Sales"
    )

    mlflow.log_param(
        "features",
        "Platform, Year, Genre, Publisher"
    )

    mlflow.log_param(
        "target",
        "Global_Sales"
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_param(
        "random_state",
        42
    )

    # Hyperparameters
    if hasattr(model, "named_steps"):

        model_step = model.named_steps.get("model")

        if model_step is not None:

            if hasattr(model_step, "n_estimators"):
                mlflow.log_param(
                    "n_estimators",
                    model_step.n_estimators
                )

            if hasattr(model_step, "learning_rate"):
                mlflow.log_param(
                    "learning_rate",
                    model_step.learning_rate
                )

            if hasattr(model_step, "max_depth"):
                mlflow.log_param(
                    "max_depth",
                    model_step.max_depth
                )

    # Final R2
    r2 = 0.15086122609280783

    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model"
    )

    # Run information
    run_id = mlflow.active_run().info.run_id

    print("MLflow Tracking Successful!")

    print("Experiment :", "Video_Game_Sales_Prediction")
    print("Run ID     :", run_id)
    print("R2 Score   :", r2)
    print("Model      :", "Gradient Boosting Regressor")
