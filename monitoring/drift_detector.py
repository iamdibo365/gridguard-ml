"""
GridGuard ML — Drift Monitor
Run weekly to detect feature drift and SLO breaches.
"""
import pandas as pd
import boto3
import json
import yaml
import os
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

BUCKET    = cfg["aws"]["s3_bucket"]
REGION    = cfg["aws"]["region"]
PSI_ALERT = cfg["monitoring"]["psi_alert_threshold"]
SLO_MIN   = cfg["monitoring"]["slo_accuracy_min"]

FEATURE_COLS = [
    "energy_kwh", "voltage_v", "current_a", "power_factor",
    "energy_rolling_mean_24h", "energy_rolling_std_24h",
    "z_score_24h", "energy_delta_1h"
]


def load_baseline() -> pd.DataFrame:
    s3  = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=BUCKET, Key="monitoring/baselines/feature_baseline.csv")
    return pd.read_csv(obj["Body"])


def load_recent_predictions(days: int = 7) -> pd.DataFrame:
    log_path = os.getenv("PREDICTION_LOG_PATH", "inference/logs/predictions.jsonl")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    cutoff  = datetime.utcnow() - timedelta(days=days)
    records = []
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("timestamp", "") > cutoff.isoformat():
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records) if records else pd.DataFrame()


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_clean = reference[FEATURE_COLS].dropna()
    cur_clean = current[[c for c in FEATURE_COLS if c in current.columns]].dropna()

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        ColumnDriftMetric(column_name="energy_kwh"),
        ColumnDriftMetric(column_name="z_score_24h"),
    ])
    report.run(reference_data=ref_clean, current_data=cur_clean)

    result        = report.as_dict()
    drift_summary = {}
    for metric in result.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            for col, stats in metric["result"]["drift_by_columns"].items():
                drift_summary[col] = {
                    "drift_detected": stats["drift_detected"],
                    "drift_score":    round(stats["drift_score"], 4),
                    "test_name":      stats["stattest_name"],
                }
    return drift_summary


def check_slo(predictions: pd.DataFrame) -> dict:
    if "anomaly_label" not in predictions.columns:
        return {"slo_check": "skipped_no_labels"}
    labeled = predictions.dropna(subset=["anomaly_label"])
    if len(labeled) < 100:
        return {"slo_check": "skipped_insufficient_data"}

    from sklearn.metrics import recall_score, precision_score
    y_true    = labeled["anomaly_label"].astype(int)
    y_pred    = (labeled["anomaly_score"] >= 0.5).astype(int)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    return {
        "recall":    round(recall, 4),
        "precision": round(precision, 4),
        "slo_met":   recall >= SLO_MIN,
    }


def generate_alerts(drift: dict, slo: dict) -> list:
    alerts = []
    for col, stats in drift.items():
        if stats.get("drift_score", 0) >= PSI_ALERT:
            alerts.append(f"DRIFT: {col} score={stats['drift_score']:.4f}")
    if slo.get("slo_met") is False:
        alerts.append(f"SLO BREACH: recall={slo['recall']:.4f} < min={SLO_MIN}")
    return alerts


if __name__ == "__main__":
    print("Loading baseline...")
    baseline = load_baseline()

    print("Loading recent predictions...")
    recent = load_recent_predictions(days=7)

    if len(recent) < 50:
        print(f"Only {len(recent)} predictions — skipping drift analysis.")
        exit(0)

    print("Running drift analysis...")
    drift  = run_drift_report(baseline, recent)
    slo    = check_slo(recent)
    alerts = generate_alerts(drift, slo)

    if alerts:
        print("\n=== ALERTS ===")
        for a in alerts:
            print(f"  {a}")
    else:
        print("✓ No drift or SLO breaches detected.")

    report_payload = {
        "run_timestamp":   datetime.utcnow().isoformat(),
        "drift_summary":   drift,
        "slo_status":      slo,
        "alerts":          alerts,
        "samples_checked": len(recent),
    }
    s3  = boto3.client("s3", region_name=REGION)
    key = f"monitoring/reports/drift_{datetime.utcnow().strftime('%Y%m%d')}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(report_payload, indent=2))
    print(f"\n✓ Report saved: s3://{BUCKET}/{key}")
