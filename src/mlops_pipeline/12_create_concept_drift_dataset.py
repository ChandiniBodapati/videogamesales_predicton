import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_data.csv"
LABELS_PATH = PROJECT_ROOT / "data/processed/y_test.npy"
OUTPUT_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_labeled_data.csv"

RANDOM_SEED = 42


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    if not FEATURES_PATH.exists():
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("ev_mod", str(PROJECT_ROOT / "src/mlops_pipeline/12_create_evolved_dataset.py"))
        ev_mod = module_from_spec(spec)
        spec.loader.exec_module(ev_mod)
        ev_mod.create_evolved_dataset()

    X = pd.read_csv(FEATURES_PATH)
    if LABELS_PATH.exists():
        y = np.load(LABELS_PATH).astype(float)
        if len(y) != len(X):
            y = rng.choice(y, size=len(X), replace=True)
    else:
        y = rng.exponential(scale=0.5, size=len(X))

    y_evolved = y.copy()

    # Concept shift: Modern PS4 / Shooter games have boosted digital/global sales
    if "Genre" in X.columns:
        shooter_mask = X["Genre"].eq("Shooter")
        y_evolved[shooter_mask] = y_evolved[shooter_mask] * rng.uniform(1.3, 1.8, size=shooter_mask.sum())

    if "Platform" in X.columns:
        ps4_mask = X["Platform"].eq("PS4")
        y_evolved[ps4_mask] = y_evolved[ps4_mask] * rng.uniform(1.2, 1.6, size=ps4_mask.sum())

    # Add random noise and round to 2 decimal places
    y_evolved = np.clip(np.round(y_evolved + rng.normal(0.05, 0.1, size=len(y_evolved)), 2), 0.01, None)

    result = X.copy()
    result["Global_Sales"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)
    print(f"Feature source : {FEATURES_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Baseline mean  : {np.mean(y):.4f}")
    print(f"Evolved mean   : {np.mean(y_evolved):.4f}")
    return True


if __name__ == "__main__":
    create_concept_drift_dataset()
