import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 3: PREPROCESSING & FEATURE PIPELINE")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    # Clean missing values
    df["Year"] = df["Year"].fillna(df["Year"].median())
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df["Genre"] = df["Genre"].fillna("Other")

    features = config["data"]["features"]
    target = config["data"]["target_column"]

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    models_dir = PROJECT_ROOT / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    test_ranks_file = processed_dir / "test_ranks.csv"

    # Use fixed test set if exists, otherwise generate new split
    if test_ranks_file.exists():
        print(f"[INFO] Using fixed test set from {test_ranks_file}")
        test_ranks_df = pd.read_csv(test_ranks_file)
        test_mask = df["Rank"].isin(test_ranks_df["Rank"])
        train_df = df[~test_mask]
        test_df = df[test_mask]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
    else:
        print("[INFO] Creating fresh reproducible train/test split...")
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["split"]["test_size"], random_state=config["split"]["random_state"]
        )
        df.loc[X_test.index, ["Rank", "Name"]].to_csv(test_ranks_file, index=False)

    print(f"[INFO] Training records: {len(X_train)}")
    print(f"[INFO] Test records    : {len(X_test)}")

    categorical_features = ["Platform", "Genre", "Publisher"]
    numerical_features = ["Year"]

    print("[INFO] Fitting preprocessing pipeline on training data...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )

    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_test_final.npy", X_test_final)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(processed_dir / "X_test_raw.csv", index=False)

    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
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

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {preprocessor_path}")
    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
