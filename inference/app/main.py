import os
import uuid
import time
import logging
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import EnergyReading, PredictionResponse, HealthResponse
from .model import load_model, get_model_version
from .monitoring import log_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GridGuard ML Inference Service",
    description="Energy anomaly detection for grid substations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
THRESHOLD  = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

FEATURE_ORDER = [
    "energy_kwh", "voltage_v", "current_a", "power_factor", "temperature_c",
    "hour", "dow", "month", "is_weekend", "is_peak",
    "energy_rolling_mean_24h", "energy_rolling_std_24h", "energy_rolling_mean_7d",
    "energy_rolling_max_24h", "voltage_rolling_mean_24h", "energy_lag_1h",
    "energy_lag_24h", "energy_delta_1h", "energy_vs_24h_mean", "z_score_24h"
]


@app.on_event("startup")
async def startup():
    logger.info("Starting GridGuard ML inference service...")
    load_model()
    logger.info("GridGuard inference service ready")


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        load_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=get_model_version(),
            uptime_seconds=round(time.time() - START_TIME, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(reading: EnergyReading):
    try:
        model    = load_model()
        features = pd.DataFrame([reading.dict()])[FEATURE_ORDER]
        proba    = float(model.predict_proba(features)[0][1])

        prediction_id = str(uuid.uuid4())
        response = PredictionResponse(
            station_id=    reading.station_id,
            anomaly_score= round(proba, 4),
            is_anomaly=    proba >= THRESHOLD,
            threshold=     THRESHOLD,
            model_version= get_model_version(),
            prediction_id= prediction_id,
            predicted_at=  datetime.utcnow()
        )

        log_prediction(reading.dict(), proba, prediction_id)

        if response.is_anomaly:
            logger.warning(
                f"ANOMALY station={reading.station_id} "
                f"score={proba:.4f} id={prediction_id}"
            )
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list)
async def predict_batch(readings: list[EnergyReading]):
    if len(readings) > 1000:
        raise HTTPException(status_code=400, detail="Batch limit is 1000 records")
    results = []
    for r in readings:
        results.append(await predict(r))
    return results
