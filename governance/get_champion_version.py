# governance/get_champion_version.py
import mlflow
import os

os.environ["DATABRICKS_HOST"]  = os.getenv("DATABRICKS_HOST", "")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "")

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()
mv     = client.get_model_version_by_alias(
    "gridguard.energy.anomaly_detector", "champion"
)
print(mv.version)