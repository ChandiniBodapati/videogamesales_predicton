# Lab 9: Pipeline Failure Handling and Recovery

## Failure Scenarios Tested

1. **Missing Data**: Target dataset file removed. Pipeline halts gracefully at raw validation stage.
2. **Invalid Datatype**: Non-numeric string injected into `Global_Sales`. Pandera schema check catches corruption and rejects batch.
3. **Schema Mismatch**: Feature column `Genre` dropped. Schema check detects missing column and halts downstream execution.
4. **Preprocessing Inconsistency**: NaN value injected into transformed array `X_train_final.npy`. Stage 4 output validation catches corruption.
5. **Missing Model Artifact + MLflow Recovery**: Local model deleted. Stage 6 fails, triggers recovery by downloading active Production model from MLflow registry, restoring service.

Execution is verified across 3 repetitions and recorded in `reports/failure_report.json`.
