from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import pandas as pd
import yaml
from validation import validate_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)


def get_next_archive_version(archive_dir):
    versions = []
    for file_path in archive_dir.glob("vgsales_v*.csv"):
        try:
            version = int(file_path.stem.replace("vgsales_v", ""))
            versions.append(version)
        except ValueError:
            continue
    return max(versions, default=0) + 1


def save_data_load_report(report):
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "data_load_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Data load report saved: {report_path}")


def main():
    print("=" * 70)
    print("DATA LOADING & LINEAGE CHECK")
    print("=" * 70)

    config = load_config()
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    new_data_path = PROJECT_ROOT / config["data"].get("new_data_path", "data/raw/new_vgsales.csv")
    archive_dir = PROJECT_ROOT / "data" / "raw" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = archive_dir / "dataset_lineage.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Primary dataset not found: {raw_path}")

    current_df = pd.read_csv(raw_path)
    current_rows = len(current_df)
    current_hash = calculate_file_hash(raw_path)

    lineage = load_lineage(lineage_path)

    # Initialize lineage if empty
    if not lineage:
        v1_archive = archive_dir / "vgsales_v01.csv"
        if not v1_archive.exists():
            shutil.copy2(raw_path, v1_archive)
        lineage.append({
            "version": "v01",
            "archive_file": str(v1_archive.relative_to(PROJECT_ROOT)),
            "rows": current_rows,
            "sha256": current_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_type": "baseline",
        })
        save_lineage(lineage_path, lineage)

    if not new_data_path.exists():
        print(f"[INFO] No new batch detected at {new_data_path}.")
        report = {
            "status": "no_new_data",
            "current_rows": current_rows,
            "raw_dataset": str(raw_path.relative_to(PROJECT_ROOT)),
            "sha256": current_hash,
        }
        save_data_load_report(report)
        print(f"[SUCCESS] Dataset loaded: {current_rows} rows.")
        return True

    print(f"[INFO] Incoming batch detected at: {new_data_path}")
    batch_df = pd.read_csv(new_data_path)
    batch_rows = len(batch_df)
    batch_hash = calculate_file_hash(new_data_path)

    print(f"[INFO] Validating batch schema ({batch_rows} records)...")
    if not validate_schema(batch_df, "new_batch_validation.csv"):
        print("[ERROR] New batch failed schema validation!")
        return False

    already_ingested = any(entry.get("sha256") == batch_hash for entry in lineage)

    if already_ingested:
        print("[INFO] This batch hash was already recorded as ingested in lineage.")
        print("[INFO] No append or retraining action will be performed.")
        report = {
            "status": "already_ingested",
            "current_rows": current_rows,
            "batch_rows": batch_rows,
            "batch_sha256": batch_hash,
        }
        save_data_load_report(report)
        return True

    # New batch ingestion: archive previous dataset snapshot, then append
    next_ver_num = get_next_archive_version(archive_dir)
    ver_str = f"v{next_ver_num:02d}"
    archive_snapshot = archive_dir / f"vgsales_{ver_str}.csv"
    shutil.copy2(raw_path, archive_snapshot)

    # Append new batch rows
    combined_df = pd.concat([current_df, batch_df], ignore_index=True)
    combined_df.to_csv(raw_path, index=False)
    new_total_rows = len(combined_df)
    new_total_hash = calculate_file_hash(raw_path)

    lineage.append({
        "version": ver_str,
        "archive_file": str(archive_snapshot.relative_to(PROJECT_ROOT)),
        "batch_rows": batch_rows,
        "new_total_rows": new_total_rows,
        "sha256": batch_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_type": "incremental",
    })
    save_lineage(lineage_path, lineage)

    report = {
        "status": "ingested",
        "previous_rows": current_rows,
        "new_rows_added": batch_rows,
        "total_rows": new_total_rows,
        "batch_sha256": batch_hash,
        "archive_version": ver_str,
    }
    save_data_load_report(report)
    print(f"[SUCCESS] Ingested {batch_rows} new records. New total: {new_total_rows} rows.")
    return True


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
