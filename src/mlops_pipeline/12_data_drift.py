import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
EVOLVED_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_data.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/drift/data_drift_report.csv"

NUMERICAL_COLUMNS = ["Year"]
CATEGORICAL_COLUMNS = ["Platform", "Genre", "Publisher"]

KS_THRESHOLD = 0.10
JS_THRESHOLD = 0.10


def load_data():
    if not BASELINE_PATH.exists() or not EVOLVED_PATH.exists():
        raise FileNotFoundError("Baseline or Evolved dataset not found. Ensure stages run first.")
    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)
    return baseline, evolved


def calculate_js_divergence(baseline, evolved):
    categories = sorted(set(baseline.dropna().unique()) | set(evolved.dropna().unique()))

    baseline_counts = baseline.value_counts(normalize=True).reindex(categories, fill_value=0).values
    evolved_counts = evolved.value_counts(normalize=True).reindex(categories, fill_value=0).values

    baseline_counts = np.asarray(baseline_counts, dtype=float)
    evolved_counts = np.asarray(evolved_counts, dtype=float)

    b_sum = baseline_counts.sum()
    e_sum = evolved_counts.sum()
    if b_sum > 0:
        baseline_counts /= b_sum
    if e_sum > 0:
        evolved_counts /= e_sum

    midpoint = (baseline_counts + evolved_counts) / 2

    baseline_mask = baseline_counts > 0
    evolved_mask = evolved_counts > 0

    kl_baseline = np.sum(baseline_counts[baseline_mask] * np.log2(baseline_counts[baseline_mask] / midpoint[baseline_mask]))
    kl_evolved = np.sum(evolved_counts[evolved_mask] * np.log2(evolved_counts[evolved_mask] / midpoint[evolved_mask]))

    return float((kl_baseline + kl_evolved) / 2)


def analyze_drift():
    baseline, evolved = load_data()
    results = []

    # Numerical drift: KS test
    for col in NUMERICAL_COLUMNS:
        if col in baseline.columns and col in evolved.columns:
            b_vals = baseline[col].dropna()
            e_vals = evolved[col].dropna()
            stat, p_val = ks_2samp(b_vals, e_vals)
            has_drift = stat > KS_THRESHOLD
            results.append({
                "feature": col,
                "type": "numerical",
                "test": "Kolmogorov-Smirnov",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "threshold": KS_THRESHOLD,
                "status": "DRIFT" if has_drift else "OK",
            })

    # Categorical drift: JS divergence
    for col in CATEGORICAL_COLUMNS:
        if col in baseline.columns and col in evolved.columns:
            js = calculate_js_divergence(baseline[col], evolved[col])
            has_drift = js > JS_THRESHOLD
            results.append({
                "feature": col,
                "type": "categorical",
                "test": "Jensen-Shannon",
                "statistic": round(float(js), 4),
                "p_value": np.nan,
                "threshold": JS_THRESHOLD,
                "status": "DRIFT" if has_drift else "OK",
            })

    report_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    print("=" * 70)
    print("DATA DRIFT ANALYSIS")
    print("=" * 70)
    print(report_df.to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")
    return True


if __name__ == "__main__":
    analyze_drift()
