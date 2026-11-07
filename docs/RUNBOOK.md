# Video Game Sales Prediction — Operations Runbook

## 1. Environment Setup

Activate the project Python virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Verify dependencies:
```powershell
python -c "import pandas, sklearn, mlflow, pandera, fastapi; print('Environment healthy')"
```

## 2. Running Individual Coursework Labs

### Lab 3: Baseline Regression Model
```powershell
python pipelines/run_lab3_baseline.py
```

### Lab 4: MLflow Tracking & Reproducibility
```powershell
python pipelines/run_lab4_tracking.py
```

### Lab 5: Pipeline & Output Validation
```powershell
python pipelines/run_lab5_pipeline.py
```

### Lab 6: Model Registry & Lifecycle Promotion
```powershell
python pipelines/run_lab6_registry.py
```

## 3. Running the Master MLOps Pipeline (Labs 7 & 8)

Run the 10-stage automated pipeline with force flag:
```powershell
python src/mlops_pipeline/vgsales_pipeline.py --force
```

Verify MLflow Model Registry:
```powershell
python src/mlops_pipeline/check_mlflow_registry.py
```

## 4. Running Failure Handling & Recovery (Lab 9)

Run the 5 failure injection scenarios across 3 repetitions:
```powershell
python src/mlops_pipeline/10_failure_injection.py
```
Output report: `reports/failure_report.json`

## 5. Serving API & Integration Testing (Lab 10)

Start the FastAPI local server:
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Run test suite:
```powershell
pytest tests/test_api.py tests/test_performance.py -v
```

Inspect inference history and performance:
```powershell
python src/mlops_pipeline/12_request_history.py
python src/mlops_pipeline/12_api_performance_monitoring.py
```

## 6. Containerization & Deployment (Lab 11)

Prepare deployment bundle:
```powershell
python src/mlops_pipeline/11_prepare_deployment.py
```

Deploy container:
```powershell
python deployment/deploy_production.py
```

## 7. Monitoring & Automated Retraining (Labs 12 & 13)

Run drift detection suite:
```powershell
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_create_concept_drift_dataset.py
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_drift_analysis.py
```

Run model comparison and retraining:
```powershell
python src/mlops_pipeline/12_retrain_model.py
python src/mlops_pipeline/12_model_comparison.py
python src/mlops_pipeline/12_update_mlflow_registry.py
```

Trigger automated monitoring orchestrator:
```powershell
python src/mlops_pipeline/12_monitor_trigger.py
```
