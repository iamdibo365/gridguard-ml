# Databricks notebook source
# GridGuard ML — Phase 3.1: Bronze Ingestion (Auto Loader)
# FIXES APPLIED:
#   - Removed cloudFiles.useNotifications (requires EventBridge setup)
#   - Replaced input_file_name() with _metadata.file_path (Unity Catalog)
#   - Removed monotonically_increasing_id() (not supported in streaming)
#   - Added header=true option
#   - Uses s3:// path (Unity Catalog external location handles auth)

import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
S3_PATH         = "s3://gridguard-ml-data/raw/energy_readings/"
CHECKPOINT_PATH = "s3://gridguard-ml-data/processed/bronze_checkpoint/"
BRONZE_TABLE    = "gridguard.energy.bronze_readings"

# ── SCHEMA ─────────────────────────────────────────────────────────────────
energy_schema = StructType([
    StructField("timestamp",     StringType(),  True),
    StructField("station_id",    StringType(),  True),
    StructField("energy_kwh",    DoubleType(),  True),
    StructField("voltage_v",     DoubleType(),  True),
    StructField("current_a",     DoubleType(),  True),
    StructField("power_factor",  DoubleType(),  True),
    StructField("temperature_c", DoubleType(),  True),
    StructField("anomaly_label", IntegerType(), True),
    StructField("anomaly_type",  StringType(),  True),
    StructField("ingested_at",   StringType(),  True),
])

# ── AUTO LOADER (directory listing mode — no EventBridge needed) ────────────
df_raw = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", CHECKPOINT_PATH + "schema/")
    .option("cloudFiles.inferColumnTypes", "false")
    .option("header", "true")
    .schema(energy_schema)
    .load(S3_PATH)
)

# ── AUDIT COLUMNS ──────────────────────────────────────────────────────────
df_bronze = (
    df_raw
    .withColumn("_source_file",      F.col("_metadata.file_path"))
    .withColumn("_bronze_loaded_at", F.current_timestamp())
    .withColumn("timestamp",         F.to_timestamp("timestamp"))
)

# ── WRITE TO DELTA ─────────────────────────────────────────────────────────
(
    df_bronze.writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH + "data/")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(BRONZE_TABLE)
)

# ── VALIDATE ───────────────────────────────────────────────────────────────
count = spark.table(BRONZE_TABLE).count()
print(f"✓ Bronze table row count: {count:,}")
spark.sql(f"DESCRIBE HISTORY {BRONZE_TABLE}").show(5, truncate=False)
