import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_preprocessing_pipeline():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    raw_path = PROJECT_ROOT / "data" / "raw" / "vgsales.csv"
    df = pd.read_csv(raw_path)

    # Clean missing values
    df["Year"] = df["Year"].fillna(df["Year"].median())
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df["Genre"] = df["Genre"].fillna("Other")

    features = ["Platform", "Year", "Genre", "Publisher"]
    target = "Global_Sales"

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("[INFO] Fitting and transforming training data...")
    categorical_features = ["Platform", "Genre", "Publisher"]
    numerical_features = ["Year"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )

    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_test_final.npy", X_test_final)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(processed_dir / "X_test_raw.csv", index=False)

    # Save test identifiers
    df.loc[X_test.index, ["Rank", "Name"]].to_csv(processed_dir / "test_ranks.csv", index=False)

    preprocessor_path = models_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "raw_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_in": features,
        "features_out_count": int(X_train_final.shape[1]),
        "target": target,
    }
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing completed! Output shape: {X_train_final.shape}")
    print(f"[SUCCESS] Preprocessor saved as {preprocessor_path}")
    return True


if __name__ == "__main__":
    run_preprocessing_pipeline()
