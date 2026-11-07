import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs/inference/inference_log.jsonl"


def analyze_performance(df):
    total = len(df)
    success = (df["status"] == "success").sum() if "status" in df.columns else 0
    failure = (df["status"] == "failure").sum() if "status" in df.columns else 0

    print("\n" + "=" * 60)
    print("API PERFORMANCE MONITORING")
    print("=" * 60)
    print(f"Total requests : {total}")
    print(f"Successful     : {success} ({success / total * 100:.1f}%)")
    print(f"Failed         : {failure} ({failure / total * 100:.1f}%)")

    if "latency_ms" in df.columns:
        lat = df["latency_ms"].dropna()
        if not lat.empty:
            print("\nLatency (ms):")
            print(f"  Average : {lat.mean():.2f} ms")
            print(f"  Median  : {lat.median():.2f} ms")
            print(f"  P95     : {lat.quantile(0.95):.2f} ms")
            print(f"  Min     : {lat.min():.2f} ms")
            print(f"  Max     : {lat.max():.2f} ms")


def main():
    if not LOG_PATH.exists():
        print(f"[INFO] No inference log found at {LOG_PATH}.")
        return True

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("[INFO] Inference log is empty.")
        return True

    df = pd.DataFrame(records)
    analyze_performance(df)
    return True


if __name__ == "__main__":
    main()
