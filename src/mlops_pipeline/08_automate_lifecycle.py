import json
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("STAGE 8: AUTOMATED MODEL LIFECYCLE DECISION")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}"))

    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"].get("decision_metric", "r2").lower()

    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        print("[ERROR] No model versions found in registry.")
        return False

    # Move any 'None' stage models to 'Staging'
    for mv in all_versions:
        if mv.current_stage == "None":
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
            )
            print(f"[INFO] Version {mv.version} moved to Staging.")

    all_versions = client.search_model_versions(f"name='{model_name}'")
    staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
    production_models = [mv for mv in all_versions if mv.current_stage == "Production"]

    if not staging_models:
        print("[INFO] No staging models to evaluate.")
        return True

    # Find best candidate among staging models
    best_candidate = None
    best_candidate_score = -999.0

    for mv in staging_models:
        run = client.get_run(mv.run_id)
        score = run.data.metrics.get(metric_to_optimize)
        if score is None:
            score = run.data.metrics.get(metric_to_optimize.upper(), -999.0)
        score = float(score)
        print(f"[INFO] Staging Version {mv.version} | {metric_to_optimize}={score:.4f}")

        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate = mv

    current_champion = production_models[0] if production_models else None
    promoted = False
    prod_version_before = int(current_champion.version) if current_champion else None
    prod_score_before = None

    if current_champion is None:
        print("[INFO] No model currently in Production. Candidate promoted by default.")
        client.transition_model_version_stage(
            name=model_name, version=best_candidate.version, stage="Production", archive_existing_versions=True
        )
        promoted = True
        prod_version_after = int(best_candidate.version)
    else:
        champ_run = client.get_run(current_champion.run_id)
        champ_score = champ_run.data.metrics.get(metric_to_optimize)
        if champ_score is None:
            champ_score = champ_run.data.metrics.get(metric_to_optimize.upper(), -999.0)
        champ_score = float(champ_score)
        prod_score_before = champ_score

        print(f"[INFO] Production Version {current_champion.version} | {metric_to_optimize}={champ_score:.4f}")

        if best_candidate_score > champ_score:
            print("[SUCCESS] Staging candidate is better than current Production model.")
            print(f"[SUCCESS] Version {best_candidate.version} is now Production.")
            client.transition_model_version_stage(
                name=model_name, version=best_candidate.version, stage="Production", archive_existing_versions=True
            )
            promoted = True
            prod_version_after = int(best_candidate.version)
        else:
            print("[INFO] Production model remains unchanged.")
            print(f"[INFO] Version {best_candidate.version} archived because it did not improve.")
            client.transition_model_version_stage(
                name=model_name, version=best_candidate.version, stage="Archived", archive_existing_versions=False
            )
            prod_version_after = int(current_champion.version)

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric_to_optimize,
        "candidate_version": int(best_candidate.version),
        "candidate_run_id": best_candidate.run_id,
        "candidate_metric": best_candidate_score,
        "production_version_before": prod_version_before,
        "production_metric_before": prod_score_before,
        "promoted": promoted,
        "decision": "PROMOTED" if promoted else "REJECTED",
        "production_version_after": prod_version_after,
    }

    report_path = PROJECT_ROOT / config["paths"]["lifecycle_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("[SUCCESS] Automated Model Lifecycle execution complete!")
    print(f"[INFO] Lifecycle report saved to {report_path}")
    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
