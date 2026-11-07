from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    GameInput,
    PredictionResponse,
)

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        model_loader.load()
    except Exception as e:
        print(f"[WARNING] ModelLoader deferred startup loading: {e}")
    yield
    # Shutdown
    print("[INFO] Video Game Sales Prediction API shutting down...")


app = FastAPI(
    title="Video Game Sales Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)
        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service error",
            "details": [{"field": "prediction", "message": str(exc.detail)}],
        },
    )


@app.get("/")
def root():
    return {
        "message": "Video Game Sales Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "service": "Video Game Sales Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: GameInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump()

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame([data_dict])
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)
        pred = float(model_loader.model.predict(X)[0])
        pred_rounded = round(float(max(0.0, pred)), 4)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            prediction=pred_rounded,
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        return {
            "predicted_global_sales": pred_rounded,
            "unit": "millions of units",
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    batch_request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dicts = [game.model_dump() for game in data.games]

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error="Model or preprocessor is not available",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=batch_request_id,
            )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame(data_dicts)
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)
        preds = model_loader.model.predict(X)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        results = []

        for data_dict, pred in zip(data_dicts, preds):
            val = round(float(max(0.0, float(pred))), 4)
            log_inference(
                input_data=data_dict,
                prediction=val,
                latency_ms=total_latency_ms,
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )
            results.append({
                "predicted_global_sales": val,
                "unit": "millions of units",
            })

        return {
            "count": len(results),
            "predictions": results,
        }
    except Exception as e:
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        for data_dict in data_dicts:
            log_inference(
                input_data=data_dict,
                latency_ms=total_latency_ms,
                status="failure",
                error=str(e),
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
            )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction",
        )
