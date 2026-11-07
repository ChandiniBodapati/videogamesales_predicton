from pathlib import Path
import pandas as pd
import yaml
from validation import validate_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 2: RAW DATA VALIDATION")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not raw_path.exists():
        print(f"[ERROR] Target file not found at: {raw_path}")
        return False

    df = pd.read_csv(raw_path)
    return validate_schema(df, output_report_name="baseline_validation.csv")


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
