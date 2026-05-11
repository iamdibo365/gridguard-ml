"""
Data quality tests — run by GitHub Actions as Job 1.
Uses Databricks REST API instead of databricks-connect.
"""
import pytest
import requests
import os

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
HEADERS          = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


def run_query(sql: str) -> list:
    """Run SQL via Databricks SQL Statement API."""
    url  = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    body = {
        "statement":  sql,
        "warehouse_id": os.getenv("DATABRICKS_WAREHOUSE_ID"),
        "wait_timeout": "30s",
    }
    r = requests.post(url, headers=HEADERS, json=body)
    r.raise_for_status()
    result = r.json()
    return result.get("result", {}).get("data_array", [])


def test_silver_table_exists():
    rows = run_query("SHOW TABLES IN gridguard.energy LIKE 'silver_readings'")
    assert len(rows) > 0, "Silver table does not exist"


def test_silver_row_count():
    rows  = run_query("SELECT COUNT(*) FROM gridguard.energy.silver_readings")
    count = int(rows[0][0])
    assert count > 1000, f"Silver table has only {count} rows"


def test_no_null_station_ids():
    rows  = run_query(
        "SELECT COUNT(*) FROM gridguard.energy.silver_readings WHERE station_id IS NULL"
    )
    count = int(rows[0][0])
    assert count == 0, f"Found {count} null station_ids"


def test_energy_values_non_negative():
    rows  = run_query(
        "SELECT COUNT(*) FROM gridguard.energy.silver_readings WHERE energy_kwh < 0"
    )
    count = int(rows[0][0])
    assert count == 0, f"Found {count} negative energy_kwh values"


def test_gold_feature_columns_present():
    rows     = run_query("DESCRIBE gridguard.energy.gold_features")
    col_names = [r[0] for r in rows]
    required  = [
        "energy_rolling_mean_24h", "energy_rolling_std_24h",
        "z_score_24h", "energy_lag_24h"
    ]
    missing = [c for c in required if c not in col_names]
    assert not missing, f"Missing Gold columns: {missing}"

    # Write version for downstream jobs
    with open(".model_version", "w") as f:
        f.write(str(mv.version))
    
    # Verify it was written
    with open(".model_version") as f:
        written = f.read().strip()
    print(f"✓ Model version written: '{written}'")