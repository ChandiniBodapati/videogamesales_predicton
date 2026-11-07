# Lab 8: Continuous Integration (CI) Workflow

## GitHub Actions CI Pipeline

The CI workflow `.github/workflows/lab8_ci.yml` triggers on push or pull request to `main` when configuration, pipeline scripts, or data change.

### Steps:
1. Environment provision (Python 3.11 on Ubuntu)
2. Dependency installation (`pip install -r requirements.txt`)
3. Pipeline execution (`python src/mlops_pipeline/vgsales_pipeline.py --force`)
4. Artifact collection (`artifacts/**`, `reports/**`, `logs/**`, `outputs/**`, `models/**`)
