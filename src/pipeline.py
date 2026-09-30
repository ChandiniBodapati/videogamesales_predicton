import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


def run_step(script):
    print("\n" + "=" * 50)
    print(f"Running: {script}")
    print("=" * 50)

    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\n Pipeline stopped at {script}")
        sys.exit(1)

    print(f" {script} completed successfully")


def main():

    print("\n VIDEO GAME SALES ML PIPELINE")
    print("=" * 50)

    # 1. Data validation
    run_step("validate_data.py")

    # 2. Data preprocessing
    run_step("preprocess.py")

    # 3. Model training
    run_step("train.py")

    # 4. Model evaluation
    run_step("evaluate.py")

    # 5. MLflow experiment tracking
    run_step("mlflow_tracking.py")

    # 6. Model registration
    run_step("model_registry.py")

    # 7. Prediction
    run_step("predict.py")

    print("\n" + "=" * 50)
    print("COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()