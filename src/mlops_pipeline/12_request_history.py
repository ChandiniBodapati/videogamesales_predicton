import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs/inference/inference_log.jsonl"


def load_history():
    if not LOG_PATH.exists():
        print(f"[INFO] No inference log found at {LOG_PATH}. Simulating synthetic history for demo.")
        return pd.DataFrame([
            {
                "request_id": "test_req_01",
                "timestamp": "2026-11-07T12:00:00Z",
                "prediction": 0.82,
                "latency_ms": 14.2,
                "status": "success",
                "model_version": "1",
            }
        ])

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("[INFO] Log file is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def main():
    df = load_history()
    if df.empty:
        print("No inference logs available.")
        return True

    print("\n" + "=" * 60)
    print("REQUEST-RESPONSE HISTORY")
    print("=" * 60)
    print(f"Total records : {len(df)}")

    cols = [c for c in ["request_id", "timestamp", "prediction", "latency_ms", "status", "model_version"] if c in df.columns]
    print("\nRecent requests:")
    print(df[cols].tail(10).to_string(index=False))

    failures = df[df["status"] == "failure"] if "status" in df.columns else pd.DataFrame()
    print("\n" + "=" * 60)
    print(f"FAILED REQUESTS ({len(failures)})")
    print("=" * 60)
    if not failures.empty:
        print(failures.to_string(index=False))
    else:
        print("No failed requests detected.")
    return True


if __name__ == "__main__":
    main()
