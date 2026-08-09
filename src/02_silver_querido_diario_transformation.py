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
# Apply silver layer transformations - METADATA ONLY (no text fetching yet)
# Text content will be fetched and updated incrementally in a separate step

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
    
    # 5. Initialize text content fields as NULL (will be populated later)
    .withColumn("text_content", F.lit(None).cast("string"))
    .withColumn("content_extraction_status", F.lit("pending"))
    .withColumn("content_error_message", F.lit(None).cast("string"))
    .withColumn("has_content", F.lit(False))
    .withColumn("text_length", F.lit(0).cast("int"))
    .withColumn("text_preview", F.lit(None).cast("string"))
    .withColumn("word_count", F.lit(0).cast("int"))
    .withColumn("content_extracted_at", F.lit(None).cast("timestamp"))
    
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
        
        # Text content fields (initially NULL/empty)
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

print(f"✓ Silver transformations applied (metadata only)")
print(f"Silver DataFrame: {silver_df.count():,} records after deduplication")
print(f"Note: Text content fields initialized as NULL/pending - will be fetched incrementally")

# COMMAND ----------

# DBTITLE 1,Write to Silver Table
from delta.tables import DeltaTable

# Write metadata to silver table (FAST - no text fetching)
print("Writing metadata to silver table...")

table_exists = spark.catalog.tableExists(SILVER_TABLE)

if table_exists:
    print(f"Table {SILVER_TABLE} exists. Checking for new records...")
    
    # Identify new records by anti-joining against existing gazette_ids
    existing_ids = spark.table(SILVER_TABLE).select("gazette_id")
    new_records_df = silver_df.join(existing_ids, "gazette_id", "left_anti")
    new_count = new_records_df.count()
    
    if new_count == 0:
        print("No new records to insert.")
    else:
        print(f"Found {new_count:,} new records. Appending...")
        (
            new_records_df
            .write
            .format("delta")
            .mode("append")
            .saveAsTable(SILVER_TABLE)
        )
        print(f"✓ Inserted {new_count:,} new records successfully")

else:
    print(f"Table {SILVER_TABLE} does not exist. Creating new table...")
    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .option("delta.autoOptimize.optimizeWrite", "true")
        .option("delta.autoOptimize.autoCompact", "true")
        .saveAsTable(SILVER_TABLE)
    )
    spark.sql(f"""
        COMMENT ON TABLE {SILVER_TABLE} IS 
        'Silver layer: cleaned and enriched official gazettes with extracted text content'
    """)
    print(f"✓ Table {SILVER_TABLE} created successfully")

final_count = spark.table(SILVER_TABLE).count()
print(f"\n✓ Metadata write complete. Total records: {final_count:,}")
print(f"Note: Text content not yet fetched (status='pending'). Run next cell to fetch incrementally.")

# COMMAND ----------

# DBTITLE 1,Fetch and Update Text Content Incrementally
from delta.tables import DeltaTable
import time

# Configuration for incremental text fetching
BATCH_SIZE = 50  # Process 50 URLs at a time
MAX_BATCHES = None  # Set to a number to limit batches (e.g., 5 for testing), or None for all

print("Starting incremental text content fetching...")
print(f"Batch size: {BATCH_SIZE} records")
print("-" * 60)

# Count records that need text content
pending_count = (
    spark.table(SILVER_TABLE)
    .filter(F.col("content_extraction_status") == "pending")
    .count()
)

if pending_count == 0:
    print("✓ No pending records. All text content already fetched.")
else:
    print(f"Found {pending_count:,} records with pending text content")
    
    total_batches = (pending_count + BATCH_SIZE - 1) // BATCH_SIZE
    if MAX_BATCHES:
        total_batches = min(total_batches, MAX_BATCHES)
        print(f"Processing first {total_batches} batches (limit set)")
    
    delta_table = DeltaTable.forName(spark, SILVER_TABLE)
    
    batch_num = 0
    while batch_num < total_batches:
        batch_num += 1
        batch_start = time.time()
        
        print(f"\nBatch {batch_num}/{total_batches}:")
        
        # Select next batch of pending records
        pending_batch = (
            spark.table(SILVER_TABLE)
            .filter(F.col("content_extraction_status") == "pending")
            .select("gazette_id", "txt_url")
            .limit(BATCH_SIZE)
        )
        
        batch_count = pending_batch.count()
        if batch_count == 0:
            print("  No more pending records.")
            break
        
        print(f"  Fetching text from {batch_count} URLs...")
        
        # Fetch text content using the UDF
        updates_df = (
            pending_batch
            .withColumn("fetch_result", fetch_text_content(F.col("txt_url")))
            .withColumn("text_content", F.col("fetch_result.content"))
            .withColumn("content_extraction_status", F.col("fetch_result.status"))
            .withColumn("content_error_message", F.col("fetch_result.error_message"))
            .drop("fetch_result")
            .withColumn("has_content", F.col("text_content").isNotNull())
            .withColumn("text_length", F.length(F.col("text_content")))
            .withColumn("text_preview", F.substring(F.col("text_content"), 1, 500))
            .withColumn("word_count", 
                        F.when(F.col("text_content").isNotNull(),
                               F.size(F.split(F.trim(F.col("text_content")), "\\s+")))
                         .otherwise(0))
            .withColumn("content_extracted_at", F.current_timestamp())
        )
        
        # Merge updates back into the table
        (
            delta_table.alias("target")
            .merge(
                updates_df.alias("source"),
                "target.gazette_id = source.gazette_id"
            )
            .whenMatchedUpdate(
                set = {
                    "text_content": "source.text_content",
                    "content_extraction_status": "source.content_extraction_status",
                    "content_error_message": "source.content_error_message",
                    "has_content": "source.has_content",
                    "text_length": "source.text_length",
                    "text_preview": "source.text_preview",
                    "word_count": "source.word_count",
                    "content_extracted_at": "source.content_extracted_at"
                }
            )
            .execute()
        )
        
        batch_duration = time.time() - batch_start
        print(f"  ✓ Updated {batch_count} records in {batch_duration:.1f}s ({batch_duration/batch_count:.2f}s per record)")
    
    # Final summary
    print("\n" + "=" * 60)
    remaining = (
        spark.table(SILVER_TABLE)
        .filter(F.col("content_extraction_status") == "pending")
        .count()
    )
    completed = pending_count - remaining
    print(f"✓ Text fetching complete: {completed:,} records processed")
    if remaining > 0:
        print(f"  {remaining:,} records still pending (run this cell again to continue)")
    print("=" * 60)

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