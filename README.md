# Video Game Sales Prediction — End-to-End MLOps Platform

A complete, production-grade MLOps system for predicting Video Game Global Sales (regression in millions of units), built across Labs 1 through 13.

## Project Architecture

```text
├── .github/workflows/          # Automated GitHub Actions CI/CD workflows
│   ├── lab8_ci.yml
│   ├── lab9_failure_tests.yml
│   └── lab13_monitoring.yml
├── api/                        # FastAPI Serving Layer (Lab 10)
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── monitoring_logger.py
│   └── preprocessing.py
├── artifacts/                  # Schema validation, lineage & pipeline reports
├── config.yaml                 # Centralized project configuration
├── data/                       # Raw data, archives, and monitoring datasets
│   ├── raw/
│   │   ├── vgsales.csv
│   │   └── archive/
│   ├── processed/
│   └── monitoring/drift/
├── deployment/                 # Deployment artifacts & automated updater (Lab 11)
│   ├── deploy_production.py
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── docs/                       # Architecture, Runbook, and Lab reports
├── k8s/                        # Kubernetes deployment & service definitions
├── logs/                       # Pipeline run logs & inference telemetry
├── models/                     # Trained models and preprocessing transformers
├── outputs/                    # Retraining, drift, and inference outputs
├── pipelines/                  # Lab runners (Labs 3 to 6)
├── reports/                    # Evaluation, quality gate, and failure reports
├── src/                        # Core ML modules and MLOps stages
│   └── mlops_pipeline/         # 10-stage automated pipeline & drift suite
└── tests/                      # Pytest suite for API and performance testing
```

## Quick Start & Verification

### 1. Run Baseline Coursework Labs (Labs 3-6)
```powershell
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python pipelines/run_lab6_registry.py
```

### 2. Run the Full 10-Stage MLOps Pipeline (Labs 7 & 8)
```powershell
python src/mlops_pipeline/vgsales_pipeline.py --force
```

### 3. Run Pipeline Failure Injection & Recovery (Lab 9)
```powershell
python src/mlops_pipeline/10_failure_injection.py
```

### 4. Run Pytest Test Suite (Lab 10)
```powershell
pytest tests/ -v
```

### 5. Run Continuous Drift Monitoring & Retraining (Labs 12 & 13)
```powershell
python src/mlops_pipeline/12_monitor_trigger.py
```
