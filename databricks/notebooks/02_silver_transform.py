# Databricks notebook source
# GridGuard ML — Phase 3.2: Silver Transform (Validation & Quality Gates)

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import json

BRONZE_TABLE = "gridguard.energy.bronze_readings"
SILVER_TABLE = "gridguard.energy.silver_readings"
DQ_LOG_TABLE = "gridguard.energy.data_quality_log"

# ── LOAD BRONZE ────────────────────────────────────────────────────────────
df             = spark.table(BRONZE_TABLE)
original_count = df.count()
print(f"Bronze rows loaded: {original_count:,}")

# ── DATA QUALITY CHECKS ────────────────────────────────────────────────────
dq_results = {}

null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in ["timestamp", "station_id", "energy_kwh", "voltage_v"]
]).first().asDict()
dq_results["null_check"] = null_counts

invalid_energy  = df.filter(F.col("energy_kwh") < 0).count()
invalid_voltage = df.filter(
    (F.col("voltage_v") < 100) | (F.col("voltage_v") > 350)
).count()
dq_results["range_check"] = {
    "invalid_energy":  invalid_energy,
    "invalid_voltage": invalid_voltage
}

dup_count = df.count() - df.dropDuplicates(["timestamp", "station_id"]).count()
dq_results["duplicate_check"] = {"duplicates": dup_count}
print(f"DQ Results: {json.dumps(dq_results, indent=2)}")

# ── FAIL FAST ──────────────────────────────────────────────────────────────
critical_nulls = null_counts.get("station_id", 0) + null_counts.get("timestamp", 0)
if critical_nulls > original_count * 0.05:
    raise ValueError(
        f"QUALITY GATE FAILED: {critical_nulls} critical nulls exceed 5% threshold"
    )

# ── CLEAN & TRANSFORM ──────────────────────────────────────────────────────
df_silver = (
    df
    .dropna(subset=["timestamp", "station_id", "energy_kwh"])
    .withColumn("voltage_v",
        F.when(F.col("voltage_v").between(100, 350), F.col("voltage_v"))
         .otherwise(F.lit(None)))
    .withColumn("_rank", F.row_number().over(
        Window.partitionBy("timestamp", "station_id")
              .orderBy(F.col("_bronze_loaded_at").desc())
    ))
    .filter(F.col("_rank") == 1)
    .drop("_rank")
    .withColumn("station_id",           F.upper(F.trim(F.col("station_id"))))
    .withColumn("_silver_processed_at", F.current_timestamp())
    .withColumn("_dq_passed",           F.lit(True))
)

# ── UPSERT INTO SILVER ─────────────────────────────────────────────────────
if spark.catalog.tableExists(SILVER_TABLE):
    delta_table = DeltaTable.forName(spark, SILVER_TABLE)
    (
        delta_table.alias("target")
        .merge(
            df_silver.alias("source"),
            "target.timestamp = source.timestamp AND target.station_id = source.station_id"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    df_silver.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)

# ── LOG DQ RESULTS ─────────────────────────────────────────────────────────
silver_count = spark.table(SILVER_TABLE).count()
dq_log = spark.createDataFrame([{
    "run_timestamp": str(F.current_timestamp()),
    "source_table":  BRONZE_TABLE,
    "target_table":  SILVER_TABLE,
    "input_rows":    original_count,
    "output_rows":   silver_count,
    "dropped_rows":  original_count - silver_count,
    "dq_results":    json.dumps(dq_results),
}])
(
    dq_log.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(DQ_LOG_TABLE)
)

print(f"✓ Silver rows: {silver_count:,} | Dropped: {original_count - silver_count:,}")
