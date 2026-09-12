# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup: Import Libraries
# Silver Layer Transformation - Querido Diário Official Gazettes
# This notebook transforms bronze layer data into a cleaned, typed silver layer
# with extracted text content from URLs

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, DateType, TimestampType, ArrayType
import requests
import hashlib
from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Configuration: Table Names
# Configuration: Define catalog, schema, and table names
BRONZE_TABLE = "workspace.tcc_rag.bronze_querido_diario"
SILVER_TABLE = "workspace.tcc_rag.silver_querido_diario"

print(f"Source (Bronze): {BRONZE_TABLE}")
print(f"Target (Silver): {SILVER_TABLE}")

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
# Load bronze data with basic filtering
# Only include records with successfully downloaded text content

bronze_df = (
    spark.table(BRONZE_TABLE)
    .filter(
        F.col("territory_id").isNotNull() &
        F.col("date").isNotNull() &
        F.col("txt_raw").isNotNull() &
        (F.col("status_download") == F.lit("success"))
    )
)

record_count = bronze_df.count()
print(f"Loaded {record_count:,} records from bronze table (with text content)")
print(f"Schema: {len(bronze_df.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Apply Silver Transformations
# Apply silver layer transformations
# Text content comes from txt_raw column (already downloaded in Bronze layer)

silver_df = (
    bronze_df
    
    # 1. Deduplicate on key fields (territory_id, date, edition)
    .dropDuplicates(["territory_id", "date", "edition"])
    
    # 2. Type conversions: date fields from STRING to proper types
    .withColumn("publication_date", F.to_date(F.col("date")))
    .withColumn("scraped_at_ts", F.to_timestamp(F.col("scraped_at")))
    .withColumn("_bronze_ingestion_timestamp", F.to_timestamp(F.col("_ingestion_timestamp")))
    
    # 3. Boolean conversion: is_extra_edition from STRING to BOOLEAN
    .withColumn("is_extra_edition_bool", 
                F.when(F.lower(F.col("is_extra_edition")) == "true", True)
                 .when(F.lower(F.col("is_extra_edition")) == "false", False)
                 .otherwise(None).cast("boolean"))
    
    # 4. Generate surrogate key: SHA256 hash of territory_id + date + edition
    .withColumn("gazette_id", 
                F.sha2(F.col("txt_url"), 256))
    
    # 5. Process text content from txt_raw (already downloaded in Bronze)
    .withColumn("text_content", F.col("txt_raw"))
    .withColumn("has_content", F.col("txt_raw").isNotNull())
    .withColumn("text_length", F.length(F.col("txt_raw")))
    .withColumn("text_preview", F.substring(F.col("txt_raw"), 1, 500))
    .withColumn("word_count", 
                F.when(F.col("txt_raw").isNotNull(),
                       F.size(F.split(F.trim(F.col("txt_raw")), "\\s+")))
                 .otherwise(0))
    
    # 6. Add processing timestamp
    .withColumn("_silver_processed_at", F.current_timestamp())
    
    # 7. Select and rename columns for final silver schema
    .select(
        # Primary identifiers
        "gazette_id",
        "territory_id",
        "territory_name",
        "state_code",
        
        # Temporal data (properly typed)
        "publication_date",
        "edition",
        F.col("is_extra_edition_bool").alias("is_extra_edition"),
        F.col("scraped_at_ts").alias("scraped_at"),
        
        # Source URLs
        F.col("url").alias("pdf_url"),
        "txt_url",
        
        # Text content fields (from Bronze txt_raw)
        "text_content",
        "text_length",
        "text_preview",
        "word_count",
        "has_content",
        
        # Search hints
        F.col("excerpts").alias("search_excerpts"),
        
        # Lineage
        "_bronze_ingestion_timestamp",
        "_silver_processed_at"
    )
)

print(f"✓ Silver transformations applied")
print(f"Silver DataFrame: {silver_df.count():,} records after deduplication")
print(f"Text content loaded from Bronze layer (txt_raw column)")

# COMMAND ----------

# DBTITLE 1,Write to Silver Table
# Write to silver table
print("Writing to silver table...")

(
    silver_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(SILVER_TABLE)
)

final_count = spark.table(SILVER_TABLE).count()
print(f"✓ Write complete. Total records: {final_count:,}")
print(f"Text content loaded from Bronze layer (txt_raw column)")

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Data quality checks and sample results

silver_table = spark.table(SILVER_TABLE)

# 1. Overall statistics
print("=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

total_records = silver_table.count()
print(f"\n1. Total records: {total_records:,}")

# 2. Content availability
print("\n2. Text Content Availability:")
content_counts = (
    silver_table
    .groupBy("has_content")
    .count()
    .orderBy(F.desc("count"))
)
display(content_counts)

# 3. Content rate
with_content = silver_table.filter(F.col("has_content") == True).count()
content_rate = (with_content / total_records * 100) if total_records > 0 else 0
print(f"\n3. Content Rate: {content_rate:.2f}% ({with_content:,} / {total_records:,})")

# 4. Text content statistics (for successful extractions)
print("\n4. Text Content Statistics (successful extractions):")
text_stats = (
    silver_table
    .filter(F.col("has_content") == True)
    .select(
        F.avg("text_length").alias("avg_text_length"),
        F.min("text_length").alias("min_text_length"),
        F.max("text_length").alias("max_text_length"),
        F.avg("word_count").alias("avg_word_count")
    )
)
display(text_stats)

# 5. Records by state
print("\n5. Top 10 States by Record Count:")
state_counts = (
    silver_table
    .groupBy("state_code")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
)
display(state_counts)

# 6. Sample records with text preview
print("\n6. Sample Records (with text preview):")
sample_records = (
    silver_table
    .select(
        "gazette_id",
        "territory_name",
        "state_code",
        "publication_date",
        "has_content",
        "text_length",
        "word_count",
        "text_preview"
    )
    .limit(5)
)
display(sample_records)

print("\n" + "=" * 60)
print("✓ Data quality checks completed")
print("=" * 60)