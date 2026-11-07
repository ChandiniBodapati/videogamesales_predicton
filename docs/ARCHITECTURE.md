# Video Game Sales Prediction — MLOps Architecture

## Purpose

This repository implements a production-grade Video Game Global Sales prediction system featuring automated data validation, feature engineering, MLflow experiment tracking and Model Registry, R2 quality gating, automated lifecycle promotion (Champion vs. Challenger), serving APIs, failure recovery, drift monitoring suite, and containerized deployment.

## End-to-End Pipeline Workflow

```text
GitHub Actions / Pipeline Runner
              |
              v
vgsales_pipeline.py (--force or event-driven)
              |
              v
[01_load_data.py]
  - SHA-256 batch hash detection
  - Lineage tracking (dataset_lineage.json)
  - Historical snapshot archiving (data/raw/archive/vgsales_v*.csv)
              |
              v
[02_validate_data.py]
  - Pandera schema validation (Rank, Platform, Year, Genre, Publisher, Global_Sales)
              |
              v
[03_preprocessing.py]
  - Categorical encoding (OneHotEncoder for Platform, Genre, Publisher)
  - Numerical scaling (StandardScaler for Year)
  - Fixed test set split retention (test_ranks.csv)
  - Exports models/preprocessor.pkl and dataset_metadata.json
              |
              v
[04_validate_outputs.py]
  - Shape, null checks, feature dimensions, test alignment
              |
              v
[05_train_registry.py]
  - Trains GradientBoostingRegressor
  - Tracks metrics (R2, MSE, MAE, RMSE) and artifacts in MLflow
  - Registers candidate in MLflow Model Registry
              |
              v
[06_model_evaluation.py]
  - Calculates held-out performance -> reports/evaluation_report.json
              |
              v
[07_quality_gate.py]
  - Quality Gate rule: R2 >= 0.08
  - Pass/Fail assertion -> reports/quality_gate_report.json
              |
              v
[08_automate_lifecycle.py]
  - Champion vs Challenger promotion logic
  - Transitions staging candidate to Production if R2 improves
  - Generates reports/lifecycle_report.json
              |
        +-----+----------------+
        |                      |
   Not Promoted             Promoted
        |                      |
        v                      v
 Skip Deployment       [deploy_if_promoted.py]
                               |
                               v
                       [11_prepare_deployment.py]
                         - Extracts model.pkl, preprocessor.pkl, model_metadata.json
                               |
                               v
                       [deployment/deploy_production.py]
                         - Builds Docker image (videogamesales-api:model-vN)
                         - Updates Kubernetes deployment rollout
              |
              v
[09_inference.py]
  - Loads active Production model from MLflow registry
  - Runs batch inferences -> outputs/inference_predictions.csv
```

## Core Modules and Capabilities

| Component | Responsibility |
|---|---|
| `src/mlops_pipeline/01_load_data.py` | Lineage tracking, SHA-256 fingerprinting, versioned snapshots |
| `src/mlops_pipeline/02_validate_data.py` | Schema assertion, range checks, datatype validation with Pandera |
| `src/mlops_pipeline/03_preprocessing.py` | Scalers, one-hot encoders, reproducible test sets |
| `src/mlops_pipeline/04_validate_outputs.py` | Feature vector dimension assertions, NaN prevention |
| `src/mlops_pipeline/05_train_registry.py` | Model training, MLflow tracking, artifact registration |
| `src/mlops_pipeline/06_model_evaluation.py` | Held-out validation report generation |
| `src/mlops_pipeline/07_quality_gate.py` | Metric gate validation (R2 >= 0.08) |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Champion vs Challenger automated stage transitions |
| `src/mlops_pipeline/deploy_if_promoted.py` | Conditional deployment hook based on lifecycle report |
| `src/mlops_pipeline/09_inference.py` | MLflow Production model loader and inference engine |
| `src/mlops_pipeline/10_failure_injection.py` | 5 failure scenarios with automated recovery validation |
| `src/mlops_pipeline/11_prepare_deployment.py` | Deployment bundle extractor |
| `deployment/deploy_production.py` | Docker container build & Kubernetes deployment updater |
| `api/main.py` | FastAPI serving endpoints (`/health`, `/metadata`, `/predict`, `/predict/batch`) |
| `src/mlops_pipeline/12_monitor_trigger.py` | Drift detection and automated retraining trigger suite |
