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


def evaluate_window(model_loader, X, y):
    X_clean = clean_prediction_data(X)
    X_trans = model_loader.preprocessor.transform(X_clean)
    preds = model_loader.model.predict(X_trans)

    mse = float(mean_squared_error(y, preds))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    rmse = float(np.sqrt(mse))

    return {"r2": r2, "rmse": rmse, "mae": mae, "mse": mse}


def main():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    model_loader = ModelLoader()
    model_loader.load()

    num_windows = 3
    X_splits = np.array_split(X_test, num_windows)
    y_splits = np.array_split(y_test, num_windows)

    results = []
    for i, (xw, yw) in enumerate(zip(X_splits, y_splits), start=1):
        m = evaluate_window(model_loader, xw, yw)
        results.append({"window": f"Window {i}", "samples": len(xw), **m})

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE TREND ANALYSIS (TEMPORAL WINDOWS)")
    print("=" * 80)
    print(df_results.to_string(index=False, formatters={"r2": "{:.4f}".format, "rmse": "{:.4f}".format, "mae": "{:.4f}".format, "mse": "{:.4f}".format}))

    r2_change = df_results["r2"].iloc[-1] - df_results["r2"].iloc[0]
    print(f"\nR2 change across windows: {r2_change:+.4f}")
    return True


if __name__ == "__main__":
    main()
