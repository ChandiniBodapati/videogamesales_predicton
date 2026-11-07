# Lab 11: Containerization & Kubernetes Deployment

## Architecture
- **Dockerfile**: Minimal python:3.11-slim container serving FastAPI via Uvicorn on port 8000.
- **Artifact Bundle**: `deployment/model.pkl`, `deployment/preprocessor.pkl`, `deployment/model_metadata.json`.
- **Kubernetes Deployment**: `k8s/deployment.yaml` with replica specification, readiness probes, and label selectors.
- **Kubernetes Service**: `k8s/service.yaml` exposing NodePort service on port 8000.
- **Automated Rollout**: `deployment/deploy_production.py` safely inspects Docker daemon and Kubernetes connectivity before initiating container builds and rolling updates.
