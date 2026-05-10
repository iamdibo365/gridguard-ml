from pydantic import BaseModel, Field, validator
from datetime import datetime


class EnergyReading(BaseModel):
    model_config = {"protected_namespaces": ()}

    station_id:               str   = Field(..., description="Substation identifier")
    energy_kwh:               float = Field(..., ge=0, description="Energy consumption kWh")
    voltage_v:                float = Field(..., ge=0, le=500)   # 500 allows spike detection
    current_a:                float = Field(..., ge=0)
    power_factor:             float = Field(..., ge=0, le=1)
    temperature_c:            float
    hour:                     int   = Field(..., ge=0, le=23)
    dow:                      int   = Field(..., ge=1, le=7)
    month:                    int   = Field(..., ge=1, le=12)
    is_weekend:               int   = Field(..., ge=0, le=1)
    is_peak:                  int   = Field(..., ge=0, le=1)
    energy_rolling_mean_24h:  float
    energy_rolling_std_24h:   float
    energy_rolling_mean_7d:   float
    energy_rolling_max_24h:   float
    voltage_rolling_mean_24h: float
    energy_lag_1h:            float
    energy_lag_24h:           float
    energy_delta_1h:          float
    energy_vs_24h_mean:       float
    z_score_24h:              float

    @validator("energy_kwh")
    def energy_not_extreme(cls, v):
        if v > 100_000:
            raise ValueError("energy_kwh exceeds physical plausibility limit")
        return v


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    station_id:    str
    anomaly_score: float    = Field(..., description="Probability of anomaly [0,1]")
    is_anomaly:    bool     = Field(..., description="True if score >= threshold")
    threshold:     float
    model_version: str
    prediction_id: str
    predicted_at:  datetime


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status:         str
    model_loaded:   bool
    model_version:  str
    uptime_seconds: float
