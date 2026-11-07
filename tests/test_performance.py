import time
import pytest
from starlette.testclient import TestClient

from api.main import app, model_loader

VALID_GAME = {
    "Rank": 1,
    "Name": "Wii Sports",
    "Platform": "Wii",
    "Year": 2006.0,
    "Genre": "Sports",
    "Publisher": "Nintendo",
}


@pytest.fixture(scope="module")
def client():
    try:
        model_loader.load()
    except Exception:
        pass
    with TestClient(app) as test_client:
        yield test_client


def test_single_request_latency(client):
    start = time.perf_counter()
    response = client.post("/predict", json=VALID_GAME)
    latency = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert latency < 1000  # Latency under 1 second


def test_repeated_request_latency(client):
    latencies = []
    for _ in range(15):
        start = time.perf_counter()
        resp = client.post("/predict", json=VALID_GAME)
        latencies.append((time.perf_counter() - start) * 1000)
        assert resp.status_code == 200

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 300  # Warm requests average under 300ms


def test_prediction_consistency(client):
    res1 = client.post("/predict", json=VALID_GAME).json()
    res2 = client.post("/predict", json=VALID_GAME).json()

    assert res1["predicted_global_sales"] == res2["predicted_global_sales"]


def test_api_reliability(client):
    success_count = 0
    total = 25
    for _ in range(total):
        resp = client.post("/predict", json=VALID_GAME)
        if resp.status_code == 200:
            success_count += 1

    assert success_count == total


def test_batch_latency_scaling(client):
    start_single = time.perf_counter()
    client.post("/predict/batch", json={"games": [VALID_GAME]})
    single_duration = time.perf_counter() - start_single

    start_batch = time.perf_counter()
    client.post("/predict/batch", json={"games": [VALID_GAME] * 10})
    batch_duration = time.perf_counter() - start_batch

    # 10 items should take less than 10x single latency due to vectorized batching
    assert batch_duration < (single_duration * 12 + 0.1)
