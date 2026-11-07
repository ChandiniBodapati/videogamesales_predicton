import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

TEST_FEATURES_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
TEST_LABELS_PATH = PROJECT_ROOT / "data/processed/y_test.npy"

V2_MODEL_PATH = PROJECT_ROOT / "models/retrained_model_v2.pkl"
V2_PREPROCESSOR_PATH = PROJECT_ROOT / "models/retrained_preprocessor_v2.pkl"

REPORT_PATH = PROJECT_ROOT / "outputs/retraining/model_comparison_report.csv"
JSON_REPORT_PATH = PROJECT_ROOT / "outputs/retraining/model_comparison_report.json"


def evaluate(model, preprocessor, X_test, y_test):
    X_clean = clean_prediction_data(X_test)
    X_trans = preprocessor.transform(X_clean)
    preds = model.predict(X_trans)

    mse = float(mean_squared_error(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    rmse = float(np.sqrt(mse))

    return {"r2": r2, "mse": mse, "mae": mae, "rmse": rmse}


def main():
    print("=" * 70)
    print("MODEL COMPARISON: V1 (CHAMPION) VS V2 (CHALLENGER)")
    print("=" * 70)

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    # V1 evaluation via ModelLoader
    v1_loader = ModelLoader()
    v1_loader.load()
    v1_metrics = evaluate(v1_loader.model, v1_loader.preprocessor, X_test, y_test)

    # V2 evaluation
    if not V2_MODEL_PATH.exists() or not V2_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Retrained V2 model or preprocessor not found. Run 12_retrain_model.py first.")

    v2_model = joblib.load(V2_MODEL_PATH)
    v2_preprocessor = joblib.load(V2_PREPROCESSOR_PATH)
    v2_metrics = evaluate(v2_model, v2_preprocessor, X_test, y_test)

    rows = []
    for metric in ["r2", "rmse", "mae", "mse"]:
        v1_val = v1_metrics[metric]
        v2_val = v2_metrics[metric]
        diff = v2_val - v1_val
        improved = (diff > 0) if metric == "r2" else (diff < 0)
        rows.append({
            "metric": metric,
            "v1_champion": round(v1_val, 4),
            "v2_challenger": round(v2_val, 4),
            "delta": round(diff, 4),
            "improved": improved,
        })

    report_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    candidate_promoted = bool(v2_metrics["r2"] >= v1_metrics["r2"])

    summary_json = {
        "status": "success",
        "champion_r2": v1_metrics["r2"],
        "challenger_r2": v2_metrics["r2"],
        "recommendation": "PROMOTE_V2" if candidate_promoted else "KEEP_V1",
        "comparison": rows,
    }
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    print(report_df.to_string(index=False))
    print(f"\nRecommendation : {summary_json['recommendation']}")
    print(f"Reports saved to: {REPORT_PATH} and {JSON_REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
