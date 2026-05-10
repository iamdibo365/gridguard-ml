"""
CI/CD Quality Gate — blocks deployment if model does not meet standards.
Gate 1: Test AUC >= threshold
Gate 2: Val/Test AUC gap <= max allowed (overfitting check)
"""
import mlflow
import os

MODEL_NAME    = "gridguard.energy.anomaly_detector"
AUC_THRESHOLD = float(os.getenv("MLFLOW_AUC_THRESHOLD", "0.85"))
MAX_AUC_GAP   = float(os.getenv("MAX_VAL_TEST_AUC_GAP", "0.05"))

os.environ["DATABRICKS_HOST"]  = os.getenv("DATABRICKS_HOST", "")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "")

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()


def test_model_meets_standards():
    # Get champion model
    mv      = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run     = client.get_run(mv.run_id)
    metrics = run.data.metrics

    test_auc = float(metrics.get("auc", 0))
    val_auc  = float(metrics.get("val_auc", 0))
    auc_gap  = abs(val_auc - test_auc)

    print(f"Model:    {MODEL_NAME} v{mv.version}")
    print(f"Val AUC:  {val_auc:.4f}  (used during training)")
    print(f"Test AUC: {test_auc:.4f}  (gate threshold: {AUC_THRESHOLD})")
    print(f"AUC Gap:  {auc_gap:.4f}   (overfitting threshold: {MAX_AUC_GAP})")

    # Gate 1: Test AUC
    assert test_auc >= AUC_THRESHOLD, (
        f"GATE 1 FAILED: Test AUC {test_auc:.4f} < {AUC_THRESHOLD}. Deployment blocked."
    )
    print(f"✓ Gate 1 passed: Test AUC {test_auc:.4f} >= {AUC_THRESHOLD}")

    # Gate 2: Overfitting check
    assert auc_gap <= MAX_AUC_GAP, (
        f"GATE 2 FAILED: Val/Test gap {auc_gap:.4f} > {MAX_AUC_GAP}. Overfitting detected."
    )
    print(f"✓ Gate 2 passed: AUC gap {auc_gap:.4f} <= {MAX_AUC_GAP}")

    # Write version for downstream jobs
    with open(".model_version", "w") as f:
        f.write(mv.version)

    print(f"\n✓ All gates passed. Model v{mv.version} approved for deployment.")


if __name__ == "__main__":
    test_model_meets_standards()
