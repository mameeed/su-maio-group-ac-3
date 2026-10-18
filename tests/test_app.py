from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PATIENT = {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
    "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_version" in response.json()


def test_predict_returns_number():
    response = client.post("/predict", json=PATIENT)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_missing_feature_returns_422():
    patient = dict(PATIENT)
    del patient["s6"]
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    assert "s6" in response.text


def test_wrong_type_returns_422():
    patient = dict(PATIENT)
    patient["age"] = "abc"
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
