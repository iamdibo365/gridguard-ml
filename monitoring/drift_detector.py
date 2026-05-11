"""
GridGuard ML — Drift Monitor
Uses scipy stats directly — no Evidently version dependency.
Run weekly to detect feature drift and SLO breaches.

Usage:
    PREDICTION_LOG_PATH=inference/logs/predictions.jsonl \
    python monitoring/drift_detector.py
"""
import pandas as pd
import boto3
import json
import yaml
import os
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_baseline() -> pd.DataFrame:
    """Load training feature distribution baseline from S3."""
    s3  = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(
        Bucket=BUCKET,
        Key="monitoring/baselines/feature_baseline.csv"
    )
    df = pd.read_csv(obj["Body"])
    print(f"Baseline loaded: {len(df):,} rows")
    return df


def load_recent_predictions(days: int = 7) -> pd.DataFrame:
    """Load recent prediction logs from local JSONL file."""
    log_path = os.getenv(
        "PREDICTION_LOG_PATH",
        "inference/logs/predictions.jsonl"
    )
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return pd.DataFrame()

    cutoff  = datetime.utcnow() - timedelta(days=days)
    records = []
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("timestamp", "") > cutoff.isoformat():
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"Recent predictions loaded: {len(df):,} rows (last {days} days)")
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Compute drift per feature using PSI and KS test.
    PSI > 0.2 or KS p-value < 0.05 = drift detected.
    """
    drift_summary = {}

    for col in FEATURE_COLS:
        if col not in reference.columns or col not in current.columns:
            print(f"  Skipping {col} — not in both datasets")
            continue

        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            print(f"  Skipping {col} — insufficient data")
            continue

        # KS Test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)

        # PSI Calculation
        n_bins         = 10
        ref_hist, bins = np.histogram(ref_vals, bins=n_bins)
        cur_hist, _    = np.histogram(cur_vals, bins=bins)
        ref_pct        = (ref_hist + 0.0001) / len(ref_vals)
        cur_pct        = (cur_hist + 0.0001) / len(cur_vals)
        psi            = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        drift_summary[col] = {
            "psi":            round(psi, 4),
            "ks_statistic":   round(float(ks_stat), 4),
            "ks_pvalue":      round(float(ks_pvalue), 4),
            "drift_detected": bool(psi >= PSI_ALERT or ks_pvalue < 0.05),
        }

    return drift_summary


def check_slo(predictions: pd.DataFrame) -> dict:
    """Check recall SLO if ground truth labels are available."""
    if "anomaly_label" not in predictions.columns:
        return {"slo_check": "skipped_no_labels"}

    labeled = predictions.dropna(subset=["anomaly_label"])
    if len(labeled) < 100:
        return {
            "slo_check": "skipped_insufficient_data",
            "n":         int(len(labeled))
        }

    from sklearn.metrics import recall_score, precision_score
    y_true    = labeled["anomaly_label"].astype(int)
    y_pred    = (labeled["anomaly_score"] >= 0.5).astype(int)
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    return {
        "recall":          round(recall, 4),
        "precision":       round(precision, 4),
        "slo_met":         bool(recall >= SLO_MIN),
        "slo_threshold":   SLO_MIN,
        "labeled_samples": int(len(labeled)),
    }


def generate_alerts(drift: dict, slo: dict) -> list:
    """Collect threshold breaches as alert strings."""
    alerts = []

    for col, s in drift.items():
        if s.get("psi", 0) >= PSI_ALERT:
            alerts.append(
                f"DRIFT ALERT: {col} PSI={s['psi']:.4f} >= threshold {PSI_ALERT}"
            )
        if s.get("ks_pvalue", 1) < 0.05:
            alerts.append(
                f"DRIFT ALERT: {col} KS p-value={s['ks_pvalue']:.4f} (distribution shift)"
            )

    if slo.get("slo_met") is False:
        alerts.append(
            f"SLO BREACH: recall={slo['recall']:.4f} < minimum {SLO_MIN}"
        )

    return alerts


def save_report(drift: dict, slo: dict, alerts: list, n_samples: int):
    """Save report locally and upload to S3."""
    report = {
        "run_timestamp":   datetime.utcnow().isoformat(),
        "drift_summary":   drift,
        "slo_status":      slo,
        "alerts":          alerts,
        "samples_checked": int(n_samples),
    }

    # Save locally
    os.makedirs("monitoring/reports", exist_ok=True)
    date_str   = datetime.utcnow().strftime("%Y%m%d")
    local_path = f"monitoring/reports/drift_{date_str}.json"

    with open(local_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nReport saved locally: {local_path}")

    # Upload to S3
    try:
        s3  = boto3.client("s3", region_name=REGION)
        key = f"monitoring/reports/drift_{date_str}.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(report, indent=2, cls=NumpyEncoder)
        )
        print(f"Report uploaded to S3: s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"S3 upload skipped (non-fatal): {e}")

    return report


if __name__ == "__main__":
    print("=" * 50)
    print("GridGuard ML — Drift Monitor")
    print("=" * 50)

    # Step 1: Load baseline
    print("\nStep 1: Loading baseline from S3...")
    try:
        baseline = load_baseline()
    except Exception as e:
        print(f"ERROR loading baseline: {e}")
        print("\nFix: Run this in a Databricks notebook:")
        print("  import boto3, io")
        print("  baseline = spark.table('gridguard.energy.feature_baseline').toPandas()")
        print("  s3 = boto3.client('s3', aws_access_key_id='KEY', aws_secret_access_key='SECRET')")
        print("  buf = io.StringIO()")
        print("  baseline.to_csv(buf, index=False)")
        print("  s3.put_object(Bucket='gridguard-ml-data', Key='monitoring/baselines/feature_baseline.csv', Body=buf.getvalue().encode())")
        exit(1)

    # Step 2: Load predictions
    print("\nStep 2: Loading recent predictions...")
    recent = load_recent_predictions(days=7)

    if len(recent) < 50:
        print(f"Only {len(recent)} predictions found — need at least 50.")
        print("Run: python inference/generate_test_predictions.py")
        exit(0)

    # Step 3: Drift analysis
    print("\nStep 3: Running drift analysis...")
    drift  = run_drift_report(baseline, recent)
    slo    = check_slo(recent)
    alerts = generate_alerts(drift, slo)

    # Print results
    print("\n" + "=" * 50)
    print("DRIFT RESULTS")
    print("=" * 50)
    for col, s in drift.items():
        status = "DRIFT" if s["drift_detected"] else "OK   "
        print(f"  [{status}] {col}")
        print(f"           PSI={s['psi']:.4f}  KS_stat={s['ks_statistic']:.4f}  KS_p={s['ks_pvalue']:.4f}")

    print("\n" + "=" * 50)
    print("SLO STATUS")
    print("=" * 50)
    for k, v in slo.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    if alerts:
        print(f"ALERTS ({len(alerts)})")
        print("=" * 50)
        for a in alerts:
            print(f"  ⚠  {a}")
    else:
        print("ALL CLEAR — No drift or SLO breaches detected.")
        print("=" * 50)

    # Save report
    save_report(drift, slo, alerts, len(recent))
    print("\nDrift monitoring complete.")