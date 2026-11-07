import json
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path.as_posix()}"))

    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"].get("decision_metric", "r2")

    all_versions = client.search_model_versions(f"name='{model_name}'")

    if not all_versions:
        print("[INFO] No model versions found in registry.")
        return False

    # 1. Discover untracked models (Stage: None) and push to Staging
    new_models = [mv for mv in all_versions if mv.current_stage == "None"]
    if new_models:
        print(f"[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")
        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
            )
            print(f"  -> Version {mv.version} is now in Staging.")

    all_versions = client.search_model_versions(f"name='{model_name}'")
    staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]

    if not staging_models:
        print("[INFO] No models in Staging to evaluate. Exiting.")
        return True

    # 2. Evaluate Staging Models
    print("\n[INFO] Evaluating Staging candidates...")
    best_challenger = None
    best_challenger_score = -999.0

    for mv in staging_models:
        run = client.get_run(mv.run_id)
        score = run.data.metrics.get(metric_to_optimize)
        if score is None:
            score = run.data.metrics.get(metric_to_optimize.upper(), -999.0)
        score = float(score)
        print(f"  -> Staging Candidate: Version {mv.version} | {metric_to_optimize}: {score:.4f}")

        if score > best_challenger_score:
            best_challenger_score = score
            best_challenger = mv

    print(f"\n[INFO] Best Challenger is Version {best_challenger.version} ({metric_to_optimize}: {best_challenger_score:.4f})")

    # 3. Find current Champion in Production
    production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
    current_champion = production_models[0] if production_models else None

    promoted = False
    prod_version_before = int(current_champion.version) if current_champion else None
    prod_score_before = None

    if current_champion is None:
        print("[INFO] No model currently in Production. Challenger wins by default!")
        print(f"\n[INFO] Promoting Version {best_challenger.version} to Production...")
        client.transition_model_version_stage(
            name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=True
        )
        promoted = True
        prod_version_after = int(best_challenger.version)
    else:
        champ_run = client.get_run(current_champion.run_id)
        current_champ_score = champ_run.data.metrics.get(metric_to_optimize)
        if current_champ_score is None:
            current_champ_score = champ_run.data.metrics.get(metric_to_optimize.upper(), -999.0)
        current_champ_score = float(current_champ_score)
        prod_score_before = current_champ_score

        print(f"[INFO] Current Champion: Version {current_champion.version} ({metric_to_optimize}: {current_champ_score:.4f})")

        if best_challenger_score > current_champ_score:
            print(f"\n[SUCCESS] Challenger (V{best_challenger.version}) outperforms Champion (V{current_champion.version})!")
            print(f"[INFO] Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=True
            )
            promoted = True
            prod_version_after = int(best_challenger.version)
        else:
            print(f"\n[INFO] Challenger (V{best_challenger.version}) did NOT outperform Champion. Champion remains active.")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Archived", archive_existing_versions=False
            )
            prod_version_after = int(current_champion.version)

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric_to_optimize,
        "candidate_version": int(best_challenger.version),
        "candidate_run_id": best_challenger.run_id,
        "candidate_metric": best_challenger_score,
        "production_version_before": prod_version_before,
        "production_metric_before": prod_score_before,
        "promoted": promoted,
        "decision": "PROMOTED" if promoted else "REJECTED",
        "production_version_after": prod_version_after,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "lifecycle_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[INFO] Lifecycle report saved to {reports_dir / 'lifecycle_report.json'}")
    return True


if __name__ == "__main__":
    automate_champion_challenger()
