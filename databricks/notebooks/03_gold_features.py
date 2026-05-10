# Databricks notebook source
# GridGuard ML — Phase 3.3: Gold Feature Engineering

import pyspark.sql.functions as F
from pyspark.sql.window import Window

SILVER_TABLE   = "gridguard.energy.silver_readings"
GOLD_TABLE     = "gridguard.energy.gold_features"
BASELINE_TABLE = "gridguard.energy.feature_baseline"

df = spark.table(SILVER_TABLE)
print(f"Silver rows loaded: {df.count():,}")

# ── TIME FEATURES ──────────────────────────────────────────────────────────
df = (
    df
    .withColumn("hour",       F.hour("timestamp"))
    .withColumn("dow",        F.dayofweek("timestamp"))
    .withColumn("month",      F.month("timestamp"))
    .withColumn("is_weekend", (F.col("dow") >= 6).cast("int"))
    .withColumn("is_peak",
        F.when(
            F.col("hour").between(7, 9) | F.col("hour").between(17, 20), 1
        ).otherwise(0))
)

# ── ROLLING WINDOW FEATURES ────────────────────────────────────────────────
w24  = Window.partitionBy("station_id").orderBy("timestamp").rowsBetween(-23, 0)
w168 = Window.partitionBy("station_id").orderBy("timestamp").rowsBetween(-167, 0)

df = (
    df
    .withColumn("energy_rolling_mean_24h",  F.mean("energy_kwh").over(w24))
    .withColumn("energy_rolling_std_24h",   F.stddev("energy_kwh").over(w24))
    .withColumn("energy_rolling_mean_7d",   F.mean("energy_kwh").over(w168))
    .withColumn("energy_rolling_max_24h",   F.max("energy_kwh").over(w24))
    .withColumn("voltage_rolling_mean_24h", F.mean("voltage_v").over(w24))
)

# ── LAG FEATURES ───────────────────────────────────────────────────────────
w_lag = Window.partitionBy("station_id").orderBy("timestamp")
df = (
    df
    .withColumn("energy_lag_1h",   F.lag("energy_kwh", 1).over(w_lag))
    .withColumn("energy_lag_24h",  F.lag("energy_kwh", 24).over(w_lag))
    .withColumn("energy_delta_1h", F.col("energy_kwh") - F.lag("energy_kwh", 1).over(w_lag))
)

# ── DERIVED FEATURES ───────────────────────────────────────────────────────
df = (
    df
    .withColumn("energy_vs_24h_mean",
        F.col("energy_kwh") / (F.col("energy_rolling_mean_24h") + 0.001))
    .withColumn("z_score_24h",
        (F.col("energy_kwh") - F.col("energy_rolling_mean_24h")) /
        (F.col("energy_rolling_std_24h") + 0.001))
)

# ── DROP WARM-UP ROWS ──────────────────────────────────────────────────────
df_gold = df.dropna(subset=["energy_lag_24h", "energy_rolling_std_24h"])

# ── WRITE GOLD ─────────────────────────────────────────────────────────────
(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)

gold_count = spark.table(GOLD_TABLE).count()
print(f"✓ Gold table rows:  {gold_count:,}")
print(f"✓ Feature columns:  {len(df_gold.columns)}")

# ── SAVE FEATURE BASELINE ──────────────────────────────────────────────────
FEATURE_COLS = [
    "energy_kwh", "voltage_v", "current_a", "power_factor",
    "energy_rolling_mean_24h", "energy_rolling_std_24h",
    "z_score_24h", "energy_delta_1h"
]
baseline = df_gold.select(FEATURE_COLS).sample(0.1)
(
    baseline.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(BASELINE_TABLE)
)
print(f"✓ Feature baseline saved ({baseline.count():,} rows)")
