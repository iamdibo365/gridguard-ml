"""
Model loader — loads XGBoost model from local mounted path.
No MLflow, no Databricks, no S3 calls at runtime.
Model is downloaded locally before container build using download_model.py.
"""
import xgboost as xgb
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Model mounted from local disk — no network calls needed
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")


@lru_cache(maxsize=1)
def load_model():
    """Load XGBoost model from local file. Cached after first load."""
    model_file = os.path.join(MODEL_PATH, "model.xgb")
    logger.info(f"Loading model from: {model_file}")

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}. "
            f"Run download_model.py first, then mount the model/ folder."
        )

    model = xgb.XGBClassifier()
    model.load_model(model_file)
    logger.info("Model loaded successfully")
    return model


def get_model_version() -> str:
    """Read model version from registered_model_meta file."""
    try:
        meta_path = os.path.join(MODEL_PATH, "registered_model_meta")
        with open(meta_path) as f:
            for line in f:
                if "model_version" in line:
                    return line.split(":")[-1].strip().strip("'\"")
        return "1"
    except Exception:
        return "1"
