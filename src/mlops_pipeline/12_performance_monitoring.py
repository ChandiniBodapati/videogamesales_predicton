import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

X_TEST_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
Y_TEST_PATH = PROJECT_ROOT / "data/processed/y_test.npy"


def evaluate_model(model_loader, X_test, y_test):
    X_clean = clean_prediction_data(X_test)
    X_trans = model_loader.preprocessor.transform(X_clean)
    preds = model_loader.model.predict(X_trans)

    mse = float(mean_squared_error(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    rmse = float(np.sqrt(mse))

    return {"r2": r2, "mse": mse, "mae": mae, "rmse": rmse}


def main():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    model_loader = ModelLoader()
    model_loader.load()

    metrics = evaluate_model(model_loader, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE MONITORING")
    print("=" * 60)
    print(f"Evaluation dataset : {X_TEST_PATH.name}")
    print(f"Test samples       : {len(X_test)}")
    print(f"Model name         : {model_loader.model_name}")
    print(f"Model version      : {model_loader.model_version}")
    print(f"MLflow Run ID      : {model_loader.run_id}")
    print("\nPerformance metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper():<8}: {v:.4f}")

    gate_passed = metrics["r2"] >= 0.08
    print(f"\nQuality Gate (R2 >= 0.08): {'PASSED' if gate_passed else 'FAILED'}")
    return True


if __name__ == "__main__":
    main()
