import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_X_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
BASELINE_Y_PATH = PROJECT_ROOT / "data/processed/y_test.npy"
EVOLVED_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_labeled_data.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/drift/concept_drift_report.csv"


def evaluate(model_loader, X, y):
    X_clean = clean_prediction_data(X)
    X_trans = model_loader.preprocessor.transform(X_clean)
    preds = model_loader.model.predict(X_trans)

    mse = float(mean_squared_error(y, preds))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    rmse = float(np.sqrt(mse))

    return {"r2": r2, "mse": mse, "mae": mae, "rmse": rmse}


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH)

    evolved = pd.read_csv(EVOLVED_PATH)
    evolved_X = evolved.drop(columns=["Global_Sales"])
    evolved_y = evolved["Global_Sales"].to_numpy()

    model_loader = ModelLoader()
    model_loader.load()

    baseline_metrics = evaluate(model_loader, baseline_X, baseline_y)
    evolved_metrics = evaluate(model_loader, evolved_X, evolved_y)

    results = []
    for m in baseline_metrics:
        b_val = baseline_metrics[m]
        e_val = evolved_metrics[m]
        diff = e_val - b_val
        results.append({
            "metric": m,
            "baseline": round(b_val, 4),
            "evolved": round(e_val, 4),
            "change": round(diff, 4),
        })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
