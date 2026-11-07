import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    data_path = PROJECT_ROOT / "data" / "processed" / "vgsales_cleaned.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "raw" / "vgsales.csv"

    df = pd.read_csv(data_path).dropna(subset=["Platform", "Year", "Genre", "Publisher", "Global_Sales"])
    X = df[["Platform", "Year", "Genre", "Publisher"]]
    y = df["Global_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Platform", "Genre", "Publisher"]),
            ("num", StandardScaler(), ["Year"]),
        ]
    )

    params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}

    # Execution 1
    pipe_1 = Pipeline([("prep", preprocessor), ("model", GradientBoostingRegressor(**params))])
    pipe_1.fit(X_train, y_train)
    score_1 = float(r2_score(y_test, pipe_1.predict(X_test)))

    # Execution 2
    pipe_2 = Pipeline([("prep", preprocessor), ("model", GradientBoostingRegressor(**params))])
    pipe_2.fit(X_train, y_train)
    score_2 = float(r2_score(y_test, pipe_2.predict(X_test)))

    is_reproducible = bool(abs(score_1 - score_2) < 1e-9)

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_r2": score_1,
        "execution_2_r2": score_2,
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED",
    }

    report_dir = PROJECT_ROOT / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "reproducibility_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 R2: {score_1:.6f}")
    print(f"Execution 2 R2: {score_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
        return True
    else:
        print("FAILED: Pipeline is non-deterministic.")
        return False


if __name__ == "__main__":
    passed = run_deterministic_test()
    exit(0 if passed else 1)
