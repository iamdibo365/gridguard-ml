import httpx
import random
import json

BASE_URL = "http://localhost:8000"

def random_reading(anomaly=False):
    base = {
        "station_id":               f"STATION_{random.randint(1,50):03d}",
        "energy_kwh":               random.uniform(200, 800),
        "voltage_v":                random.uniform(235, 245),
        "current_a":                random.uniform(5, 15),
        "power_factor":             random.uniform(0.85, 0.99),
        "temperature_c":            random.uniform(18, 28),
        "hour":                     random.randint(0, 23),
        "dow":                      random.randint(1, 7),
        "month":                    random.randint(1, 12),
        "is_weekend":               random.randint(0, 1),
        "is_peak":                  random.randint(0, 1),
        "energy_rolling_mean_24h":  random.uniform(300, 600),
        "energy_rolling_std_24h":   random.uniform(20, 80),
        "energy_rolling_mean_7d":   random.uniform(300, 600),
        "energy_rolling_max_24h":   random.uniform(400, 700),
        "voltage_rolling_mean_24h": random.uniform(238, 242),
        "energy_lag_1h":            random.uniform(200, 800),
        "energy_lag_24h":           random.uniform(200, 800),
        "energy_delta_1h":          random.uniform(-50, 50),
        "energy_vs_24h_mean":       random.uniform(0.8, 1.2),
        "z_score_24h":              random.uniform(-2, 2),
    }
    if anomaly:
        base["energy_kwh"]        = random.uniform(3000, 9000)
        base["z_score_24h"]       = random.uniform(10, 20)
        base["energy_vs_24h_mean"]= random.uniform(8, 20)
        base["energy_delta_1h"]   = random.uniform(2000, 8000)
    return base

print("Generating 100 predictions...")
for i in range(100):
    is_anomaly = random.random() < 0.05   # 5% anomaly rate
    payload    = random_reading(anomaly=is_anomaly)
    r          = httpx.post(f"{BASE_URL}/predict", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"  {i+1}/100 station={payload['station_id']} score={data['anomaly_score']} anomaly={data['is_anomaly']}")
    else:
        print(f"  {i+1}/100 ERROR: {r.status_code}")

print("\nDone. Check inference/logs/predictions.jsonl")