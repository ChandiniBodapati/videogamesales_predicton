import json
import os
from pathlib import Path
import shutil
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as file:
        return PROJECT_ROOT, yaml.safe_load(file)


def configure_mlflow(config):
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get(
        "tracking_uri", f"sqlite:///{db_path.as_posix()}"
    )
    if tracking_uri.startswith("sqlite:///") and not os.path.isabs(tracking_uri.replace("sqlite:///", "")):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    production_versions = [v for v in versions if v.current_stage == "Production"]

    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    if len(production_versions) > 1:
        # Sort descending by version number and take the newest
        production_versions = sorted(production_versions, key=lambda v: int(v.version), reverse=True)

    return production_versions[0]


def prepare_deployment():
    project_root, config = load_config()
    configure_mlflow(config)

    deployment_dir = project_root / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    model_version = get_production_model(client, model_name)

    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    print("[INFO] Downloading Production model...")
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model saved to {model_path}")

    print("[INFO] Downloading matching preprocessor...")
    final_preprocessor_path = deployment_dir / "preprocessor.pkl"

    preprocessor_artifact = None
    candidate_paths = [
        "preprocessing_pipeline/preprocessor.pkl",
        "preprocessing_pipeline_v2/retrained_preprocessor_v2.pkl",
        "preprocessing_pipeline_v2/preprocessor.pkl",
    ]
    for cp in candidate_paths:
        try:
            parent = cp.split("/")[0]
            for a in client.list_artifacts(run_id, parent):
                if a.path == cp:
                    preprocessor_artifact = cp
                    break
            if preprocessor_artifact:
                break
        except Exception:
            pass

    if preprocessor_artifact:
        try:
            downloaded = client.download_artifacts(run_id, preprocessor_artifact, dst_path=str(deployment_dir))
            shutil.copy2(downloaded, final_preprocessor_path)
        except Exception as e:
            print(f"[WARNING] Artifact download failed ({e}), falling back to local file.")
            shutil.copy2(project_root / config["paths"]["preprocessor"], final_preprocessor_path)
    else:
        local_v2 = project_root / "models" / "retrained_preprocessor_v2.pkl"
        local_v1 = project_root / config["paths"]["preprocessor"]
        src_path = local_v2 if (local_v2.exists() and "v2" in str(model_name).lower()) else local_v1
        shutil.copy2(src_path, final_preprocessor_path)

    print(f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}")

    print("[INFO] Verifying deployment artifacts...")
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(final_preprocessor_path)

    if loaded_model is None or loaded_preprocessor is None:
        raise RuntimeError("Verification of loaded deployment bundle failed.")

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "model_id": model_source.split("/")[-1] if "/" in model_source else model_source,
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "source": "MLflow",
    }

    metadata_path = deployment_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")
    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("|-- model.pkl")
    print("|-- preprocessor.pkl")
    print(r"\-- model_metadata.json")
    print("\n[SUCCESS] Deployment preparation completed.")
    return True


if __name__ == "__main__":
    prepare_deployment()
