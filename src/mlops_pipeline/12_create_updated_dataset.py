import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FEATURES_PATH = PROJECT_ROOT / "data/processed/X_train_raw.csv"
TRAIN_LABELS_PATH = PROJECT_ROOT / "data/processed/y_train.npy"
TEST_FEATURES_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"

NEW_BATCH_PATH = PROJECT_ROOT / "data/monitoring/drift/post_drift_training_data.csv"
UPDATED_DATASET_PATH = PROJECT_ROOT / "data/processed/updated_training_data.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs/retraining/updated_dataset_summary.csv"

RANDOM_SEED = 42
NEW_BATCH_SIZE = 2000


def load_original_training_data():
    if not TRAIN_FEATURES_PATH.exists() or not TRAIN_LABELS_PATH.exists():
        raw_df = pd.read_csv(PROJECT_ROOT / "data/raw/vgsales.csv")
        X = raw_df[["Platform", "Year", "Genre", "Publisher"]].copy()
        y = raw_df["Global_Sales"].to_numpy()
        return X, y, X.sample(n=1000, random_state=RANDOM_SEED)

    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH).astype(float)
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    return X_train, y_train, X_test


def create_post_drift_batch(X_train, y_train):
    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(len(X_train), size=NEW_BATCH_SIZE, replace=True)
    X_new = X_train.iloc[sample_indices].copy().reset_index(drop=True)
    y_new = y_train[sample_indices].copy()

    # Shift Year
    if "Year" in X_new.columns:
        X_new["Year"] = (
            pd.to_numeric(X_new["Year"], errors="coerce").fillna(2006) + rng.normal(4, 2, len(X_new))
        ).clip(1980, 2024).round()

    # Shift Platform
    if "Platform" in X_new.columns:
        modern_platforms = ["PS4", "XOne", "PC", "PS3", "X360"]
        X_new["Platform"] = rng.choice(modern_platforms, size=len(X_new), p=[0.40, 0.30, 0.15, 0.10, 0.05])

    # Shift Genre
    if "Genre" in X_new.columns:
        genres = ["Action", "Shooter", "Role-Playing", "Sports", "Racing"]
        X_new["Genre"] = rng.choice(genres, size=len(X_new), p=[0.40, 0.30, 0.15, 0.10, 0.05])

    # Sales adjustment
    y_new = np.clip(np.round(y_new * rng.uniform(1.1, 1.5, size=len(y_new)), 2), 0.01, None)

    batch_df = X_new.copy()
    batch_df["Global_Sales"] = y_new
    return batch_df


def create_updated_dataset():
    X_train, y_train, _ = load_original_training_data()
    original_df = X_train.copy()
    original_df["Global_Sales"] = y_train

    post_drift_batch = create_post_drift_batch(X_train, y_train)

    NEW_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    post_drift_batch.to_csv(NEW_BATCH_PATH, index=False)

    updated_df = pd.concat([original_df, post_drift_batch], ignore_index=True)
    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(UPDATED_DATASET_PATH, index=False)

    summary_df = pd.DataFrame([
        {"dataset": "Original Training", "rows": len(original_df), "mean_sales": float(original_df["Global_Sales"].mean())},
        {"dataset": "Post-Drift Batch", "rows": len(post_drift_batch), "mean_sales": float(post_drift_batch["Global_Sales"].mean())},
        {"dataset": "Updated Combined", "rows": len(updated_df), "mean_sales": float(updated_df["Global_Sales"].mean())},
    ])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("=" * 70)
    print("UPDATED DATASET CREATION COMPLETED")
    print("=" * 70)
    print(f"Original rows   : {len(original_df)}")
    print(f"Post-drift rows : {len(post_drift_batch)}")
    print(f"Combined rows   : {len(updated_df)}")
    print(f"Saved to        : {UPDATED_DATASET_PATH}")
    return True


if __name__ == "__main__":
    create_updated_dataset()
