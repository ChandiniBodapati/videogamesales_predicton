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

SECOND_GAME = {
    "Rank": 2,
    "Name": "Super Mario Bros.",
    "Platform": "NES",
    "Year": 1985.0,
    "Genre": "Platform",
    "Publisher": "Nintendo",
}


@pytest.fixture(scope="module")
def client():
    # Ensure model loader is initialized
    try:
        model_loader.load()
    except Exception:
        pass
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data


def test_valid_prediction(client):
    response = client.post("/predict", json=VALID_GAME)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_global_sales" in data
    assert isinstance(data["predicted_global_sales"], (int, float))
    assert data["predicted_global_sales"] >= 0.0


def test_invalid_prediction_year(client):
    payload = dict(VALID_GAME)
    payload["Year"] = 1950.0  # Before valid range 1980
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_invalid_prediction_future_year(client):
    payload = dict(VALID_GAME)
    payload["Year"] = 2100.0  # Above valid range 2050
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_prediction_field(client):
    payload = dict(VALID_GAME)
    del payload["Platform"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_valid_batch_prediction(client):
    batch = {"games": [VALID_GAME, SECOND_GAME]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_batch_prediction(client):
    invalid = dict(VALID_GAME)
    invalid["Year"] = "not_a_year"
    batch = {"games": [VALID_GAME, invalid]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 422


def test_empty_batch(client):
    batch = {"games": []}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 422


def test_extreme_numeric_values(client):
    payload = dict(VALID_GAME)
    payload["Year"] = -50
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_null_value(client):
    payload = dict(VALID_GAME)
    payload["Platform"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_malformed_json(client):
    response = client.post(
        "/predict",
        content="{'malformed': True}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_large_batch(client):
    batch = {"games": [VALID_GAME] * 25}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    assert response.json()["count"] == 25
