import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
EVOLVED_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_data.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/drift/prediction_drift_report.csv"

KS_THRESHOLD = 0.10


def load_data():
    if not BASELINE_PATH.exists() or not EVOLVED_PATH.exists():
        raise FileNotFoundError("Baseline or evolved test data missing.")
    return pd.read_csv(BASELINE_PATH), pd.read_csv(EVOLVED_PATH)


def predict(model_loader, data):
    clean_data = clean_prediction_data(data)
    transformed_data = model_loader.preprocessor.transform(clean_data)
    predictions = model_loader.model.predict(transformed_data)
    return predictions


def main():
    baseline, evolved = load_data()

    model_loader = ModelLoader()
    model_loader.load()

    baseline_preds = predict(model_loader, baseline)
    evolved_preds = predict(model_loader, evolved)

    stat, p_val = ks_2samp(baseline_preds, evolved_preds)
    has_drift = stat > KS_THRESHOLD

    report = pd.DataFrame([{
        "prediction_distribution_ks": round(float(stat), 4),
        "prediction_distribution_p_value": round(float(p_val), 4),
        "prediction_distribution_status": "DRIFT" if has_drift else "OK",
        "baseline_mean_prediction": round(float(np.mean(baseline_preds)), 4),
        "evolved_mean_prediction": round(float(np.mean(evolved_preds)), 4),
        "baseline_median_prediction": round(float(np.median(baseline_preds)), 4),
        "evolved_median_prediction": round(float(np.median(evolved_preds)), 4),
        "threshold": KS_THRESHOLD,
    }])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 70)
    print("PREDICTION DRIFT REPORT")
    print("=" * 70)
    print(report.to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
