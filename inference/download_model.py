"""
GridGuard ML — Model Downloader
Downloads the champion model from Databricks Unity Catalog to local disk.
Run this BEFORE building the Docker image.
Reads credentials from inference/.env — no hardcoded secrets.

Usage:
    cd gridguard-ml/inference
    pip install python-dotenv 'mlflow-skinny[databricks]'
    python download_model.py
"""
import mlflow
import os
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv(".env")

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
MODEL_URI        = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/gridguard.energy.anomaly_detector@champion"
)

if not DATABRICKS_HOST:
    raise ValueError("DATABRICKS_HOST not set in .env")
if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN not set in .env")

os.environ["DATABRICKS_HOST"]  = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

print(f"Connecting to:  {DATABRICKS_HOST}")
print(f"Downloading:    {MODEL_URI}")
print(f"Destination:    ./model/")

path = mlflow.artifacts.download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path="./model"
)

print(f"\n✓ Model downloaded to: {path}")
print("✓ Ready to build Docker image")
print("\nNext steps:")
print("  DOCKER_BUILDKIT=1 docker build -t gridguard-inference:latest .")
