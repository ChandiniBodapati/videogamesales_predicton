import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 6: MODEL EVALUATION")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = PROJECT_ROOT / config["paths"]["model"]
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    model = joblib.load(model_path)

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    report = {
        "model_name": config["registry"]["model_name"],
        "metrics": {
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        },
        "test_samples": len(y_test),
    }

    print("\n--- ML Pipeline: Model Evaluation ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print("--------------------------------------\n")

    report_path = PROJECT_ROOT / config["paths"]["evaluation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation report saved to {report_path}")
    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
