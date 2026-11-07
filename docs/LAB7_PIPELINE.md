# Lab 7: Automated MLOps Pipeline Architecture

## Pipeline Stages

The automated pipeline sequentially executes:
1. **Data Loading (`01_load_data.py`)**: Checks dataset SHA-256 hash. If new batch arrives, archives to `data/raw/archive/vgsales_v*.csv` and appends to `dataset_lineage.json`.
2. **Raw Data Validation (`02_validate_data.py`)**: Enforces Pandera schema on inputs (Rank, Platform, Year, Genre, Publisher, Global_Sales).
3. **Feature Preprocessing (`03_preprocessing.py`)**: Uses ColumnTransformer with OneHotEncoder for categoricals and StandardScaler for Year. Saves fixed test ranks (`test_ranks.csv`) to prevent leakage.
4. **Output Validation (`04_validate_outputs.py`)**: Asserts dimension consistency, non-null values, and expected column shapes.
5. **Training + MLflow Registry (`05_train_registry.py`)**: Fits GradientBoostingRegressor, logs parameters, metrics (R2, MSE, MAE, RMSE) and artifacts to MLflow, and registers candidate in the model registry.
6. **Model Evaluation (`06_model_evaluation.py`)**: Evaluates candidate on test set, writing `reports/evaluation_report.json`.
7. **Model Quality Gate (`07_quality_gate.py`)**: Verifies candidate satisfies R2 >= 0.08.
8. **Automated Lifecycle (`08_automate_lifecycle.py`)**: Compares candidate R2 against active Production model. Promotes to Production if superior.
9. **Deployment Update (`deploy_if_promoted.py`)**: Prepares production artifacts and triggers deployment if promoted.
10. **Production Inference (`09_inference.py`)**: Loads Production model from MLflow registry and performs inference.
