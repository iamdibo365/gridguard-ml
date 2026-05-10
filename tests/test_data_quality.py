"""
Data quality tests — run by GitHub Actions as Job 1.
Validates Silver and Gold tables meet quality standards.
"""
import pytest
import os

os.environ["DATABRICKS_HOST"]  = os.getenv("DATABRICKS_HOST", "")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "")


def get_spark():
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.getOrCreate()


def test_silver_table_exists():
    spark = get_spark()
    assert spark.catalog.tableExists("gridguard.energy.silver_readings")


def test_silver_row_count():
    spark = get_spark()
    count = spark.table("gridguard.energy.silver_readings").count()
    assert count > 1000, f"Silver table has only {count} rows"


def test_no_null_station_ids():
    import pyspark.sql.functions as F
    spark     = get_spark()
    null_count = (
        spark.table("gridguard.energy.silver_readings")
        .filter(F.col("station_id").isNull())
        .count()
    )
    assert null_count == 0, f"Found {null_count} null station_ids"


def test_energy_values_non_negative():
    import pyspark.sql.functions as F
    spark     = get_spark()
    neg_count = (
        spark.table("gridguard.energy.silver_readings")
        .filter(F.col("energy_kwh") < 0)
        .count()
    )
    assert neg_count == 0, f"Found {neg_count} negative energy_kwh values"


def test_gold_feature_columns_present():
    spark    = get_spark()
    gold     = spark.table("gridguard.energy.gold_features")
    required = [
        "energy_rolling_mean_24h", "energy_rolling_std_24h",
        "z_score_24h", "energy_lag_24h"
    ]
    missing = [c for c in required if c not in gold.columns]
    assert not missing, f"Missing Gold columns: {missing}"
