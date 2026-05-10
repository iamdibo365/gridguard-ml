# Databricks notebook source
# GridGuard ML — Phase 4: Model Training with MLflow
# FIXES APPLIED:
#   - mlflow.set_registry_uri("databricks-uc") for Unity Catalog
#   - Explicit mlflow.register_model() instead of registered_model_name param
#   - Uses search_model_versions() instead of deprecated get_latest_versions()
#   - Champion alias instead of Staging/Production stages
#   - Inline config — no file upload needed
#   - pip install cell at top for serverless clusters

# ── INSTALL PACKAGES (required for serverless clusters) ────────────────────
# Run this cell first, then Run All from the top
# %pip install mlflow==2.9.2 xgboost==2.0.3 scikit-learn==1.3.2 pyyaml==6.0.1
# dbutils.library.restartPython()

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
import json, yaml

# ── DATABRICKS CONNECTION ──────────────────────────────────────────────────
# Replace with your actual values
os.environ["DATABRICKS_HOST"]  = "https://<YOUR_WORKSPACE>.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "<YOUR_DATABRICKS_TOKEN>"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# ── INLINE CONFIG ──────────────────────────────────────────────────────────
config_yaml = """
aws:
  region: us-east-1
  s3_bucket: gridguard-ml-data
  ecr_repo: gridguard-inference
  ecr_registry: <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
databricks:
  workspace_url: https://<YOUR_WORKSPACE>.cloud.databricks.com
  cluster_id: <YOUR_CLUSTER_ID>
  catalog: gridguard
  schema: energy
  bronze_table: gridguard.energy.bronze_readings
  silver_table: gridguard.energy.silver_readings
  gold_table: gridguard.energy.gold_features
mlflow:
  tracking_uri: databricks
  experiment_name: /gridguard/anomaly-detection
  model_name: gridguard.energy.anomaly_detector
  auc_threshold: 0.85
  psi_alert_threshold: 0.2
monitoring:
  slo_accuracy_min: 0.90
  slo_window_days: 30
  prediction_log_table: gridguard.energy.prediction_log
"""
cfg = yaml.safe_load(config_yaml)
print("✓ Config loaded")

GOLD_TABLE = cfg["databricks"]["gold_table"]
MODEL_NAME = cfg["mlflow"]["model_name"]
EXPERIMENT = cfg["mlflow"]["experiment_name"]
AUC_GATE   = cfg["mlflow"]["auc_threshold"]

FEATURE_COLS = [
    "energy_kwh", "voltage_v", "current_a", "power_factor",
    "temperature_c", "hour", "dow", "month", "is_weekend", "is_peak",
    "energy_rolling_mean_24h", "energy_rolling_std_24h",
    "energy_rolling_mean_7d",  "energy_rolling_max_24h",
    "voltage_rolling_mean_24h","energy_lag_1h", "energy_lag_24h",
    "energy_delta_1h", "energy_vs_24h_mean", "z_score_24h"
]
TARGET_COL = "anomaly_label"

# ── LOAD GOLD DATA ─────────────────────────────────────────────────────────
df = spark.table(GOLD_TABLE).toPandas()
print(f"Total rows: {len(df):,} | Anomaly rate: {df[TARGET_COL].mean():.3f}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── THREE-WAY SPLIT (70 / 15 / 15) ────────────────────────────────────────
# Step 1: Lock away test set first — never touched during training
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Step 2: Split remaining into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nData split (stratified):")
print(f"  Train:      {len(X_train):,} rows ({100*len(X_train)/len(X):.0f}%) anomaly rate: {y_train.mean():.4f}")
print(f"  Validation: {len(X_val):,}  rows ({100*len(X_val)/len(X):.0f}%) anomaly rate: {y_val.mean():.4f}")
print(f"  Test:       {len(X_test):,} rows ({100*len(X_test)/len(X):.0f}%) anomaly rate: {y_test.mean():.4f}")

# ── MLFLOW TRAINING RUN ────────────────────────────────────────────────────
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="xgboost_three_way_split") as run:

    params = {
        "n_estimators":          500,
        "max_depth":             6,
        "learning_rate":         0.05,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "scale_pos_weight":      (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state":          42,
        "eval_metric":           "auc",
        "early_stopping_rounds": 20,
        "use_label_encoder":     False
    }
    mlflow.log_params(params)
    mlflow.log_params({
        "train_size":         len(X_train),
        "val_size":           len(X_val),
        "test_size":          len(X_test),
        "split_strategy":     "stratified_three_way_70_15_15",
        "anomaly_rate_train": float(y_train.mean()),
        "anomaly_rate_val":   float(y_val.mean()),
        "anomaly_rate_test":  float(y_test.mean()),
    })

    # TRAIN — eval_set uses VALIDATION only. Test set never passed to fit().
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    best_iter = model.best_iteration + 1
    mlflow.log_param("best_iteration", best_iter)
    print(f"\nEarly stopping fired at iteration {best_iter} (out of {params['n_estimators']})")

    # VALIDATION METRICS
    val_proba   = model.predict_proba(X_val)[:, 1]
    val_pred    = model.predict(X_val)
    val_metrics = {
        "val_auc":       round(roc_auc_score(y_val, val_proba), 4),
        "val_f1":        round(f1_score(y_val, val_pred), 4),
        "val_precision": round(precision_score(y_val, val_pred), 4),
        "val_recall":    round(recall_score(y_val, val_pred), 4),
    }
    mlflow.log_metrics(val_metrics)
    print(f"\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # TEST METRICS — evaluated ONCE. These go in the model card.
    test_proba   = model.predict_proba(X_test)[:, 1]
    test_pred    = model.predict(X_test)
    test_metrics = {
        "auc":       round(roc_auc_score(y_test, test_proba), 4),
        "f1":        round(f1_score(y_test, test_pred), 4),
        "precision": round(precision_score(y_test, test_pred), 4),
        "recall":    round(recall_score(y_test, test_pred), 4),
    }
    mlflow.log_metrics(test_metrics)
    print(f"\nTest Metrics (FINAL — one-time evaluation):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Val/Test AUC gap
    auc_gap = abs(val_metrics["val_auc"] - test_metrics["auc"])
    mlflow.log_metric("val_test_auc_gap", round(auc_gap, 4))
    if auc_gap > 0.05:
        print(f"\nWARNING: Val/Test AUC gap = {auc_gap:.4f}. Overfitting signal.")
    else:
        print(f"\nVal/Test AUC gap = {auc_gap:.4f}. Healthy generalization.")

    # Confusion matrix
    cm        = confusion_matrix(y_test, test_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    mlflow.log_dict({"confusion_matrix": cm, "labels": ["normal", "anomaly"]},
                    "confusion_matrix.json")
    print(f"\nConfusion Matrix (Test Set):")
    print(f"  True Negatives  (correct normal):  {tn}")
    print(f"  False Positives (false alarms):    {fp}")
    print(f"  False Negatives (missed anomalies):{fn}")
    print(f"  True Positives  (caught anomalies):{tp}")

    # Bias evaluation by anomaly type
    test_df = X_test.copy()
    test_df["y_true"]       = y_test.values
    test_df["y_pred"]       = test_pred
    test_df["y_proba"]      = test_proba
    test_df["anomaly_type"] = df.loc[X_test.index, "anomaly_type"].values

    bias_report = {}
    for atype in test_df["anomaly_type"].unique():
        subset = test_df[test_df["anomaly_type"] == atype]
        if len(subset) > 10:
            bias_report[atype] = {
                "n":         len(subset),
                "recall":    round(recall_score(subset["y_true"], subset["y_pred"], zero_division=0), 4),
                "precision": round(precision_score(subset["y_true"], subset["y_pred"], zero_division=0), 4),
            }
    mlflow.log_dict(bias_report, "bias_evaluation.json")
    print(f"\nBias Evaluation by Anomaly Type:")
    for k, v in bias_report.items():
        print(f"  {k}: recall={v['recall']}  precision={v['precision']}  n={v['n']}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    mlflow.log_dict(importance, "feature_importance.json")

    # AUC GATE
    if test_metrics["auc"] < AUC_GATE:
        raise ValueError(
            f"QUALITY GATE FAILED: Test AUC {test_metrics['auc']:.4f} < {AUC_GATE}. "
            f"Model not registered."
        )
    mlflow.log_param("quality_gate_passed", True)
    print(f"\n✓ Quality gate passed: Test AUC {test_metrics['auc']:.4f} >= {AUC_GATE}")

    # LOG MODEL ARTIFACT
    signature = infer_signature(X_train, test_proba[:len(X_train)])
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
    )
    print(f"✓ Model artifact logged to run: {run.info.run_id}")

    mlflow.set_tags({
        "project":        "gridguard",
        "model_type":     "anomaly_detector",
        "framework":      "xgboost",
        "governed":       "true",
        "auc_gate":       str(AUC_GATE),
        "split_strategy": "three_way_70_15_15",
        "run_id":         run.info.run_id,
    })
    print(f"\nRun ID: {run.info.run_id}")

# ── REGISTER INTO UNITY CATALOG MODEL REGISTRY ─────────────────────────────
print("\nRegistering model in Unity Catalog...")
registered = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/model",
    name=MODEL_NAME
)
print(f"✓ Model registered: {MODEL_NAME} v{registered.version}")

# ── SET CHAMPION ALIAS ─────────────────────────────────────────────────────
client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered.version
)
print(f"✓ Champion alias set on v{registered.version}")

# ── VERIFY ─────────────────────────────────────────────────────────────────
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"\n✓ Verified champion model:")
print(f"  Name:    {mv.name}")
print(f"  Version: {mv.version}")
print(f"  Run ID:  {mv.run_id}")
print(f"  URI:     models:/{MODEL_NAME}@champion")
print("\nNext: run download_model.py then build Docker image")
