import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs/inference/inference_log.jsonl"


def main():
    if not LOG_PATH.exists():
        print(f"[INFO] Inference log not found at {LOG_PATH}.")
        return True

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("[INFO] Inference log is empty.")
        return True

    df = pd.DataFrame(records)
    preds = df["prediction"].dropna()

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION MONITORING")
    print("=" * 60)
    print(f"Total logged inferences: {len(df)}")
    if not preds.empty:
        print("\nPrediction summary statistics (Global Sales in M units):")
        print(preds.describe().to_string())
    return True


if __name__ == "__main__":
    main()
