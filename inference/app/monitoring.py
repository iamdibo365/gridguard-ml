"""
Prediction logger — writes to local JSONL file only.
No S3, no AWS, no external dependencies.
Mount /app/logs as a volume to collect logs externally.
"""
import json
import os
import logging
from datetime import datetime

LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "/app/logs/predictions.jsonl")
logger   = logging.getLogger(__name__)


def log_prediction(features: dict, score: float, prediction_id: str):
    """Append prediction to local JSONL log. Never crashes the API."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        record = {
            "prediction_id": prediction_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "anomaly_score": score,
            **features
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Prediction logging failed (non-fatal): {e}")
