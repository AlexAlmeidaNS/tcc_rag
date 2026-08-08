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

# DBTITLE 1,UDF: Text Content Fetcher
# UDF to fetch text content from URLs with comprehensive error handling
from pyspark.sql.types import StructType, StructField, StringType

# Define return schema for the UDF
fetch_result_schema = StructType([
    StructField("content", StringType(), True),
    StructField("status", StringType(), True),
    StructField("error_message", StringType(), True)
])

@F.udf(returnType=fetch_result_schema)
def fetch_text_content(url):
    """
    Fetch text content from a URL with error handling.
    
    Returns:
        tuple: (content_text, status, error_message)
            - content_text: The fetched text or None
            - status: 'success', 'http_error', 'timeout', 'connection_error', 'unknown_error'
            - error_message: Error details or None
    """
    if not url:
        return (None, "invalid_url", "URL is null or empty")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Successfully fetched content
        return (response.text, "success", None)
        
    except requests.exceptions.HTTPError as e:
        # HTTP error (4xx, 5xx)
        return (None, "http_error", f"HTTP {e.response.status_code}: {str(e)}")
        
    except requests.exceptions.Timeout:
        # Request timed out
        return (None, "timeout", "Request timed out after 30 seconds")
        
    except requests.exceptions.ConnectionError as e:
        # Connection error (DNS failure, refused connection, etc.)
        return (None, "connection_error", f"Connection failed: {str(e)}")
        
    except Exception as e:
        # Catch-all for unexpected errors
        return (None, "unknown_error", f"Unexpected error: {str(e)}")

print("✓ UDF 'fetch_text_content' registered successfully")

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
# Load bronze data with basic filtering
# Remove records with null values in critical fields

bronze_df = (
    spark.table(BRONZE_TABLE)
    .filter(
        F.col("territory_id").isNotNull() &
        F.col("date").isNotNull() &
        F.col("txt_url").isNotNull()
    )
)

record_count = bronze_df.count()
print(f"Loaded {record_count:,} records from bronze table")
print(f"Schema: {len(bronze_df.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Apply Silver Transformations
# Apply all silver layer transformations

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
                F.sha2(F.concat_ws("||", 
                                    F.col("territory_id"), 
                                    F.col("date"), 
                                    F.coalesce(F.col("edition"), F.lit(""))), 256))
    
    # 5. Fetch text content from txt_url using UDF
    .withColumn("fetch_result", fetch_text_content(F.col("txt_url")))
    
    # 6. Extract fields from UDF result
    .withColumn("text_content", F.col("fetch_result.content"))
    .withColumn("content_extraction_status", F.col("fetch_result.status"))
    .withColumn("content_error_message", F.col("fetch_result.error_message"))
    .drop("fetch_result")
    
    # 7. Calculate content metrics
    .withColumn("has_content", F.col("text_content").isNotNull())
    .withColumn("text_length", F.length(F.col("text_content")))
    .withColumn("text_preview", F.substring(F.col("text_content"), 1, 500))
    .withColumn("word_count", 
                F.when(F.col("text_content").isNotNull(),
                       F.size(F.split(F.trim(F.col("text_content")), "\\s+")))
                 .otherwise(0))
    
    # 8. Add processing timestamps
    .withColumn("content_extracted_at", F.current_timestamp())
    .withColumn("_silver_processed_at", F.current_timestamp())
    
    # 9. Select and rename columns for final silver schema
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
        
        # Text content (core for RAG)
        "text_content",
        "text_length",
        "text_preview",
        "word_count",
        
        # Data quality
        "content_extraction_status",
        "content_error_message",
        "has_content",
        "content_extracted_at",
        
        # Search hints
        F.col("excerpts").alias("search_excerpts"),
        
        # Lineage
        "_bronze_ingestion_timestamp",
        "_silver_processed_at"
    )
)

print(f"✓ Silver transformations applied successfully")
print(f"Silver DataFrame: {silver_df.count():,} records after deduplication")

# COMMAND ----------

# DBTITLE 1,Write to Silver Table
# Write to silver table with proper schema enforcement
# Using merge mode for idempotent updates based on gazette_id

from delta.tables import DeltaTable

# Check if silver table exists by querying catalog
try:
    spark.table(SILVER_TABLE)
    table_exists = True
except:
    table_exists = False

if table_exists:
    print(f"Table {SILVER_TABLE} exists. Performing MERGE operation...")
    
    # Create DeltaTable reference
    delta_table = DeltaTable.forName(spark, SILVER_TABLE)
    
    # Merge logic: update existing records or insert new ones based on gazette_id
    # Explicitly map columns to avoid schema mismatch with old table versions
    merge_result = (
        delta_table.alias("target")
        .merge(
            silver_df.alias("source"),
            "target.gazette_id = source.gazette_id"
        )
        .whenMatchedUpdate(set={
            "territory_id": "source.territory_id",
            "territory_name": "source.territory_name",
            "state_code": "source.state_code",
            "publication_date": "source.publication_date",
            "edition": "source.edition",
            "is_extra_edition": "source.is_extra_edition",
            "scraped_at": "source.scraped_at",
            "pdf_url": "source.pdf_url",
            "txt_url": "source.txt_url",
            "text_content": "source.text_content",
            "text_length": "source.text_length",
            "text_preview": "source.text_preview",
            "word_count": "source.word_count",
            "content_extraction_status": "source.content_extraction_status",
            "content_error_message": "source.content_error_message",
            "has_content": "source.has_content",
            "content_extracted_at": "source.content_extracted_at",
            "search_excerpts": "source.search_excerpts",
            "_bronze_ingestion_timestamp": "source._bronze_ingestion_timestamp",
            "_silver_processed_at": "source._silver_processed_at"
        })
        .whenNotMatchedInsert(values={
            "gazette_id": "source.gazette_id",
            "territory_id": "source.territory_id",
            "territory_name": "source.territory_name",
            "state_code": "source.state_code",
            "publication_date": "source.publication_date",
            "edition": "source.edition",
            "is_extra_edition": "source.is_extra_edition",
            "scraped_at": "source.scraped_at",
            "pdf_url": "source.pdf_url",
            "txt_url": "source.txt_url",
            "text_content": "source.text_content",
            "text_length": "source.text_length",
            "text_preview": "source.text_preview",
            "word_count": "source.word_count",
            "content_extraction_status": "source.content_extraction_status",
            "content_error_message": "source.content_error_message",
            "has_content": "source.has_content",
            "content_extracted_at": "source.content_extracted_at",
            "search_excerpts": "source.search_excerpts",
            "_bronze_ingestion_timestamp": "source._bronze_ingestion_timestamp",
            "_silver_processed_at": "source._silver_processed_at"
        })
        .execute()
    )
    
    print(f"✓ MERGE completed successfully")
    
else:
    print(f"Table {SILVER_TABLE} does not exist. Creating new table...")
    
    # Create new table with explicit properties
    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .option("delta.autoOptimize.optimizeWrite", "true")
        .option("delta.autoOptimize.autoCompact", "true")
        .saveAsTable(SILVER_TABLE)
    )
    
    # Add table comment
    spark.sql(f"""
        COMMENT ON TABLE {SILVER_TABLE} IS 
        'Silver layer: cleaned and enriched official gazettes with extracted text content'
    """)
    
    print(f"✓ Table {SILVER_TABLE} created successfully")

# Verify write
final_count = spark.table(SILVER_TABLE).count()
print(f"\nFinal record count in {SILVER_TABLE}: {final_count:,}")

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

# 2. Content extraction status breakdown
print("\n2. Content Extraction Status:")
status_counts = (
    silver_table
    .groupBy("content_extraction_status")
    .count()
    .orderBy(F.desc("count"))
)
display(status_counts)

# 3. Success rate
success_count = silver_table.filter(F.col("content_extraction_status") == "success").count()
success_rate = (success_count / total_records * 100) if total_records > 0 else 0
print(f"\n3. Success Rate: {success_rate:.2f}% ({success_count:,} / {total_records:,})")

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
        "text_preview",
        "content_extraction_status"
    )
    .limit(5)
)
display(sample_records)

print("\n" + "=" * 60)
print("✓ Data quality checks completed")
print("=" * 60)