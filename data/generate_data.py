"""
GridGuard ML — Synthetic Energy Data Generator
Generates 90 days of hourly energy readings for 50 grid substations,
with injected anomalies, and uploads to AWS S3.
"""
import pandas as pd
import numpy as np
import boto3
import yaml
import io
from datetime import datetime, timedelta

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET       = cfg["aws"]["s3_bucket"]
REGION       = cfg["aws"]["region"]
NUM_STATIONS = 50
DAYS         = 90
SEED         = 42
np.random.seed(SEED)


def generate_station_data(station_id: int, start_date: datetime) -> pd.DataFrame:
    hours = DAYS * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    df = pd.DataFrame({"timestamp": timestamps})
    df["station_id"] = f"STATION_{station_id:03d}"

    base_load   = np.random.uniform(100, 800)
    hour_of_day = df["timestamp"].dt.hour
    daily_cycle = (
        15 * np.sin(2 * np.pi * (hour_of_day - 8) / 24) +
        10 * np.sin(2 * np.pi * (hour_of_day - 18) / 24)
    )
    dow            = df["timestamp"].dt.dayofweek
    weekly_factor  = np.where(dow >= 5, 0.75, 1.0)
    noise          = np.random.normal(0, base_load * 0.03, hours)

    df["energy_kwh"]    = (base_load + daily_cycle + noise) * weekly_factor
    df["voltage_v"]     = np.random.normal(240, 2, hours)
    df["current_a"]     = df["energy_kwh"] / df["voltage_v"]
    df["power_factor"]  = np.random.uniform(0.85, 0.99, hours)
    df["temperature_c"] = (
        20 + 10 * np.sin(2 * np.pi * (hour_of_day - 14) / 24) +
        np.random.normal(0, 1, hours)
    )
    return df


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anomaly_label"] = 0
    df["anomaly_type"]  = "normal"
    n = len(df)

    # Voltage spike
    spike_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    df.loc[spike_idx, "energy_kwh"] *= np.random.uniform(3, 6, len(spike_idx))
    df.loc[spike_idx, "voltage_v"]  *= np.random.uniform(1.4, 1.8, len(spike_idx))
    df.loc[spike_idx, "anomaly_label"] = 1
    df.loc[spike_idx, "anomaly_type"]  = "voltage_spike"

    # Outage
    outage_starts = np.random.choice(n - 6, size=int(n * 0.005), replace=False)
    for start in outage_starts:
        end = min(start + np.random.randint(2, 6), n)
        df.loc[start:end, "energy_kwh"]    = np.random.uniform(0, 2, end - start + 1)
        df.loc[start:end, "anomaly_label"] = 1
        df.loc[start:end, "anomaly_type"]  = "outage"

    # Sensor fault
    flat_starts = np.random.choice(n - 12, size=3, replace=False)
    for start in flat_starts:
        flat_val = df.loc[start, "energy_kwh"]
        end = min(start + np.random.randint(4, 12), n)
        df.loc[start:end, "energy_kwh"]    = flat_val
        df.loc[start:end, "anomaly_label"] = 1
        df.loc[start:end, "anomaly_type"]  = "sensor_fault"

    return df


def upload_to_s3(df: pd.DataFrame, date_str: str):
    s3  = boto3.client("s3", region_name=REGION)
    key = f"raw/energy_readings/energy_readings_{date_str}.csv"

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
        Metadata={
            "generator_version": "1.0.0",
            "num_stations":      str(NUM_STATIONS),
            "anomaly_rate":      "0.03"
        }
    )
    print(f"  Uploaded s3://{BUCKET}/{key}  ({len(df):,} rows)")


if __name__ == "__main__":
    print(f"Generating data for {NUM_STATIONS} stations x {DAYS} days...")
    start_date = datetime(2024, 1, 1)
    all_dfs    = []

    for station_id in range(1, NUM_STATIONS + 1):
        df = generate_station_data(station_id, start_date)
        df = inject_anomalies(df)
        all_dfs.append(df)
        if station_id % 10 == 0:
            print(f"  Generated {station_id}/{NUM_STATIONS} stations...")

    combined             = pd.concat(all_dfs, ignore_index=True)
    combined["ingested_at"] = datetime.utcnow().isoformat()

    print("\nUploading to S3 by month...")
    for month in sorted(combined["timestamp"].dt.month.unique()):
        month_df = combined[combined["timestamp"].dt.month == month]
        upload_to_s3(month_df, f"2024_{month:02d}")

    total     = len(combined)
    anomalies = combined["anomaly_label"].sum()
    print(f"\n✓ Total records:   {total:,}")
    print(f"✓ Total anomalies: {anomalies:,} ({100*anomalies/total:.1f}%)")
    print("✓ Data generation complete.")
