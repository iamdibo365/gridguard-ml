"""
Integration tests for the FastAPI inference service.
Run against a live container during CI/CD Job 3.
"""
import httpx

BASE_URL = "http://localhost:8000"

NORMAL_PAYLOAD = {
    "station_id":               "STATION_001",
    "energy_kwh":               450.0,
    "voltage_v":                241.5,
    "current_a":                10.4,
    "power_factor":             0.92,
    "temperature_c":            22.1,
    "hour":                     14,
    "dow":                      2,
    "month":                    6,
    "is_weekend":               0,
    "is_peak":                  0,
    "energy_rolling_mean_24h":  450.0,
    "energy_rolling_std_24h":   45.0,
    "energy_rolling_mean_7d":   430.0,
    "energy_rolling_max_24h":   520.0,
    "voltage_rolling_mean_24h": 240.0,
    "energy_lag_1h":            448.0,
    "energy_lag_24h":           442.0,
    "energy_delta_1h":          2.0,
    "energy_vs_24h_mean":       1.0,
    "z_score_24h":              0.04
}

ANOMALY_PAYLOAD = {
    **NORMAL_PAYLOAD,
    "energy_kwh":       9000.0,
    "voltage_v":        310.0,
    "energy_delta_1h":  8552.0,
    "energy_vs_24h_mean": 20.0,
    "z_score_24h":      190.0
}


def test_health_check():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_normal():
    r = httpx.post(f"{BASE_URL}/predict", json=NORMAL_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert "prediction_id" in data
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert data["is_anomaly"] is False


def test_predict_anomaly():
    r = httpx.post(f"{BASE_URL}/predict", json=ANOMALY_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["anomaly_score"] > 0.5
    assert data["is_anomaly"] is True


def test_predict_missing_field():
    bad = {k: v for k, v in NORMAL_PAYLOAD.items() if k != "energy_kwh"}
    r   = httpx.post(f"{BASE_URL}/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict():
    r = httpx.post(f"{BASE_URL}/predict/batch", json=[NORMAL_PAYLOAD, ANOMALY_PAYLOAD])
    assert r.status_code == 200
    assert len(r.json()) == 2
