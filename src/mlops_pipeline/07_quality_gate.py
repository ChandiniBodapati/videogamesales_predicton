import json
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 7: MODEL QUALITY GATE")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_report_path = PROJECT_ROOT / config["paths"]["evaluation_report"]
    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report not found: {eval_report_path}")
        return False

    with open(eval_report_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    gate_metric = config["quality_gate"]["metric"].lower()
    min_value = float(config["quality_gate"]["minimum_value"])

    observed_value = float(eval_data["metrics"].get(gate_metric, -999.0))
    passed = observed_value >= min_value

    print(f"[INFO] Quality Gate: {gate_metric}={observed_value:.4f} | required>={min_value:.4f}")

    report = {
        "status": "PASSED" if passed else "FAILED",
        "metric": gate_metric,
        "observed_value": observed_value,
        "threshold": min_value,
        "passed": passed,
    }

    report_path = PROJECT_ROOT / config["paths"]["quality_gate_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
        return True
    else:
        print("[HALT] Model Quality Gate FAILED. Candidate rejected.")
        return False


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
