# Lab 13: Continuous Monitoring & Automated Retraining Loop

## Monitoring System Overview
1. **Data Drift**: Calculates KS test for numeric feature `Year` and JS divergence for categorical features (`Platform`, `Genre`, `Publisher`).
2. **Prediction Drift**: Computes KS statistic between baseline model predictions and evolved incoming requests.
3. **Concept Drift**: Measures performance degradation (delta R2, MSE, MAE) when true labels shift.
4. **Comprehensive Analysis**: Synthesizes metrics into `outputs/drift/drift_analysis_report.csv`.
5. **Retraining & Promotion**: Fits retrained V2 model on post-drift data, benchmarks Champion vs Challenger, updates MLflow Model Registry, and promotes candidate if superior.
6. **Trigger Orchestrator**: `12_monitor_trigger.py` closes the loop by polling drift status and triggering retraining upon detection.
