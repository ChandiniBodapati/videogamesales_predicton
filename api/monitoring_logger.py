from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"
_LOCK = threading.Lock()


def log_inference(
    input_data: dict[str, Any],
    prediction: float | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error: str | None = None,
    model_name: str | None = None,
    model_version: int | str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
) -> str:
    request_id = request_id or uuid.uuid4().hex[:12]

    safe_input = dict(input_data)
    safe_input.pop("Rank", None)

    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": safe_input,
        "prediction": prediction,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return request_id
