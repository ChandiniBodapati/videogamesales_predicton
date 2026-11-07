from vgsales_pipeline import run_pipeline, STAGES
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Video Game Sales Prediction MLOps pipeline.")
    parser.add_argument("--force", action="store_true", help="Run all ML stages even when no new dataset was ingested.")
    args = parser.parse_args()

    passed = run_pipeline(force=args.force)
    sys.exit(0 if passed else 1)
