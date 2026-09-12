# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Overview
# MAGIC %md
# MAGIC # Silver Layer - Quality Variants Generator
# MAGIC
# MAGIC Gera múltiplas variantes de qualidade de cada diário oficial para experimentos de TCC. Cada diário da camada Bronze é processado em várias versões: uma baseline limpa e outras com distúrbios controlados de qualidade.

# COMMAND ----------

# DBTITLE 1,Setup: Import Libraries
# Silver Layer - Quality Variants for Controlled Experiments
# Generates multiple quality versions of each official gazette for TCC research

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from delta.tables import DeltaTable
import random
import re
from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Configuration: Table Names and Variants
# Configuration: Define catalog, schema, and table names
BRONZE_TABLE = "workspace.tcc_rag.bronze_querido_diario"
SILVER_TABLE = "workspace.tcc_rag.silver_quality_variants"

# Define which quality variants to generate
# Add new variants here without changing core logic
ENABLED_VARIANTS = [
    "baseline",      # Clean version
    "ocr_noise",     # Simulated OCR errors
    "duplicate",     # Duplicate content
    "missing_fields" # Removed content
]

# Disturbance intensity parameters (0.0 to 1.0)
DISTURBANCE_PARAMS = {
    "ocr_noise": {"char_error_rate": 0.02},      # 2% of characters affected
    "duplicate": {"duplication_rate": 0.15},     # 15% of text duplicated
    "missing_fields": {"removal_rate": 0.10}     # 10% of text removed
}

print(f"Source (Bronze): {BRONZE_TABLE}")
print(f"Target (Silver): {SILVER_TABLE}")
print(f"Enabled variants: {', '.join(ENABLED_VARIANTS)}")

# COMMAND ----------

# DBTITLE 1,Function: Baseline Text Cleaning
def clean_text_baseline(text):
    """
    Baseline text cleaning: normalizes encoding and removes redundant
    whitespace/control characters without altering semantic content.
    
    Args:
        text (str): Raw text from Bronze layer
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Normalize unicode encoding
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize whitespace: collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks: collapse multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

print("✓ Baseline cleaning function defined")

# COMMAND ----------

# DBTITLE 1,Function: OCR Noise Injection
def inject_ocr_noise(text, char_error_rate=0.02):
    """
    Simulates common OCR errors by replacing visually similar characters
    and randomly removing characters.
    
    Args:
        text (str): Clean text
        char_error_rate (float): Proportion of characters to affect (0.0-1.0)
        
    Returns:
        str: Text with OCR noise
    """
    if not text:
        return text
    
    # Common OCR confusions (visually similar characters)
    ocr_substitutions = {
        '0': 'O',
        'O': '0',
        '1': 'l',
        'l': '1',
        'I': '1',
        '5': 'S',
        'S': '5',
        '8': 'B',
        'B': '8',
        'rn': 'm',
        'vv': 'w',
        'cl': 'd'
    }
    
    chars = list(text)
    num_errors = int(len(chars) * char_error_rate)
    
    # Randomly select positions to corrupt
    error_positions = random.sample(range(len(chars)), min(num_errors, len(chars)))
    
    for pos in error_positions:
        char = chars[pos]
        
        # 70% chance of substitution, 30% chance of deletion
        if random.random() < 0.7:
            # Try to substitute with OCR confusion
            if char in ocr_substitutions:
                chars[pos] = ocr_substitutions[char]
            # Otherwise insert a random similar character
            elif char.isalpha():
                chars[pos] = random.choice(['o', '0', 'l', '1', 'i', 'I'])
        else:
            # Delete character
            chars[pos] = ''
    
    return ''.join(chars)

print("✓ OCR noise injection function defined")

# COMMAND ----------

# DBTITLE 1,Function: Content Duplication
def inject_duplicates(text, duplication_rate=0.15):
    """
    Duplicates random chunks of text to simulate data quality issues.
    
    Args:
        text (str): Clean text
        duplication_rate (float): Proportion of text to duplicate (0.0-1.0)
        
    Returns:
        str: Text with duplicated sections
    """
    if not text or len(text) < 100:
        return text
    
    # Split into sentences or paragraphs
    chunks = re.split(r'([.!?]\s+|\n\n)', text)
    
    if len(chunks) < 3:
        return text
    
    # Calculate how many chunks to duplicate
    num_duplicates = max(1, int(len(chunks) * duplication_rate))
    
    # Randomly select chunks to duplicate
    duplicate_indices = random.sample(range(len(chunks)), min(num_duplicates, len(chunks)))
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append(chunk)
        
        # Duplicate this chunk if selected
        if i in duplicate_indices:
            result.append(chunk)  # Insert duplicate immediately after
    
    return ''.join(result)

print("✓ Duplication injection function defined")

# COMMAND ----------

# DBTITLE 1,Function: Missing Fields Injection
def inject_missing_fields(text, removal_rate=0.10):
    """
    Randomly removes chunks of text to simulate missing or corrupted fields.
    
    Args:
        text (str): Clean text
        removal_rate (float): Proportion of text to remove (0.0-1.0)
        
    Returns:
        str: Text with sections removed
    """
    if not text or len(text) < 100:
        return text
    
    # Split into sentences or paragraphs
    chunks = re.split(r'([.!?]\s+|\n\n)', text)
    
    if len(chunks) < 3:
        return text
    
    # Calculate how many chunks to remove
    num_removals = max(1, int(len(chunks) * removal_rate))
    
    # Randomly select chunks to remove
    removal_indices = set(random.sample(range(len(chunks)), min(num_removals, len(chunks))))
    
    # Keep only non-removed chunks
    result = [chunk for i, chunk in enumerate(chunks) if i not in removal_indices]
    
    return ''.join(result)

print("✓ Missing fields injection function defined")

# COMMAND ----------

# DBTITLE 1,Variant Processing Registry
# Registry mapping variant names to processing functions
# To add a new variant:
# 1. Define its function above
# 2. Add entry here
# 3. Add to ENABLED_VARIANTS and DISTURBANCE_PARAMS in config

VARIANT_PROCESSORS = {
    "baseline": lambda text, params: clean_text_baseline(text),
    "ocr_noise": lambda text, params: inject_ocr_noise(
        clean_text_baseline(text), 
        params.get("char_error_rate", 0.02)
    ),
    "duplicate": lambda text, params: inject_duplicates(
        clean_text_baseline(text), 
        params.get("duplication_rate", 0.15)
    ),
    "missing_fields": lambda text, params: inject_missing_fields(
        clean_text_baseline(text), 
        params.get("removal_rate", 0.10)
    )
}

print("✓ Variant processing registry configured")
print(f"  Available processors: {list(VARIANT_PROCESSORS.keys())}")

# COMMAND ----------

# DBTITLE 1,UDF: Generate Quality Variants
from pyspark.sql.types import ArrayType

# Define schema for variant output
variant_schema = StructType([
    StructField("quality_variant", StringType(), False),
    StructField("texto", StringType(), True)
])

@F.udf(returnType=ArrayType(variant_schema))
def generate_variants(text, seed_value):
    """
    Generate all enabled quality variants for a single text.
    
    Args:
        text (str): Raw text from Bronze layer
        seed_value (int): Seed for reproducible randomness
        
    Returns:
        list: Array of (quality_variant, texto) tuples
    """
    if not text:
        return [(variant, "") for variant in ENABLED_VARIANTS]
    
    # Set seed for reproducibility
    random.seed(seed_value)
    
    results = []
    
    for variant in ENABLED_VARIANTS:
        if variant in VARIANT_PROCESSORS:
            # Get parameters for this variant
            params = DISTURBANCE_PARAMS.get(variant, {})
            
            # Apply processing function
            processed_text = VARIANT_PROCESSORS[variant](text, params)
            
            results.append((variant, processed_text))
        else:
            # Unknown variant - store original
            results.append((variant, text))
    
    return results

print("✓ UDF 'generate_variants' registered successfully")

# COMMAND ----------

# DBTITLE 1,Load Bronze Data
# Load bronze data with filtering for valid text content
bronze_df = (
    spark.table(BRONZE_TABLE)
    .filter(
        F.col("txt_url").isNotNull() &
        F.col("txt_raw").isNotNull() &
        (F.length(F.col("txt_raw")) > 100)  # At least 100 chars
    )
    .select(
        F.sha2(F.col("txt_url"), 256).alias("gazette_id"),  # Use SHA-256 hash of txt_url as gazette_id
        "txt_raw"
    )
)

record_count = bronze_df.count()
print(f"Loaded {record_count:,} records from bronze table")
print(f"Ready to generate {len(ENABLED_VARIANTS)} variants per record")
print(f"Total output records expected: {record_count * len(ENABLED_VARIANTS):,}")

# COMMAND ----------

# DBTITLE 1,Generate Quality Variants
# Generate all quality variants for each diario
print("Generating quality variants...")
print(f"This may take a few minutes for {record_count:,} records...")

variants_df = (
    bronze_df
    # Create a seed for reproducible randomness based on gazette_id
    .withColumn("seed", F.abs(F.hash(F.col("gazette_id"))))
    
    # Generate all variants using UDF
    .withColumn("variants", generate_variants(F.col("txt_raw"), F.col("seed")))
    
    # Explode variants array into separate rows
    .withColumn("variant_data", F.explode(F.col("variants")))
    
    # Extract variant fields
    .select(
        "gazette_id",
        F.col("variant_data.quality_variant").alias("quality_variant"),
        F.col("variant_data.texto").alias("texto"),
        F.current_timestamp().alias("data_processamento")
    )
    )

variant_count = variants_df.count()
print(f"✓ Generated {variant_count:,} variant records")
print(f"  = {record_count:,} gazettes × {len(ENABLED_VARIANTS)} variants")

# Show sample distribution
print("\nVariant distribution:")
variants_df.groupBy("quality_variant").count().orderBy("quality_variant").show()

# COMMAND ----------

# DBTITLE 1,Write to Silver Table (Idempotent Merge)
# Write to Silver table (recreate due to schema change from diario_id to gazette_id)
print("Writing to silver table...")

# Drop existing table to accommodate schema change (diario_id -> gazette_id)
try:
    spark.sql(f"DROP TABLE IF EXISTS {SILVER_TABLE}")
    print(f"Dropped existing table {SILVER_TABLE} to recreate with new schema")
except Exception as e:
    print(f"Table {SILVER_TABLE} does not exist, will create new")

# Create table with new schema
(
    variants_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(SILVER_TABLE)
)
print("✓ Table creation completed with gazette_id column")

# Final count
final_count = spark.table(SILVER_TABLE).count()
print(f"\n✓ Silver table now contains: {final_count:,} total records")

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Data quality checks and validation
silver_table = spark.table(SILVER_TABLE)

print("=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

# 1. Total records and unique gazettes
total_records = silver_table.count()
unique_gazettes = silver_table.select("gazette_id").distinct().count()
print(f"\n1. Total records: {total_records:,}")
print(f"   Unique gazettes: {unique_gazettes:,}")
print(f"   Variants per gazette: {total_records / unique_gazettes:.1f} (expected: {len(ENABLED_VARIANTS)})")

# 2. Records by variant
print("\n2. Distribution by Quality Variant:")
variant_dist = (
    silver_table
    .groupBy("quality_variant")
    .agg(
        F.count("*").alias("count"),
        F.avg(F.length("texto")).alias("avg_text_length"),
        F.min(F.length("texto")).alias("min_length"),
        F.max(F.length("texto")).alias("max_length")
    )
    .orderBy("quality_variant")
)
display(variant_dist)

# 3. Completeness check
print("\n3. Completeness Check:")
completeness = (
    silver_table
    .groupBy("gazette_id")
    .agg(F.countDistinct("quality_variant").alias("variant_count"))
    .groupBy("variant_count")
    .count()
    .orderBy("variant_count")
)
display(completeness)

# 4. Sample records
print("\n4. Sample Records (one per variant):")
for variant in ENABLED_VARIANTS:
    print(f"\n--- {variant.upper()} ---")
    sample = (
        silver_table
        .filter(F.col("quality_variant") == variant)
        .select(
            "gazette_id",
            F.substring("texto", 1, 200).alias("text_preview"),
            F.length("texto").alias("text_length")
        )
        .limit(1)
    )
    display(sample)

print("\n" + "=" * 60)
print("✓ Data quality checks completed")
print("=" * 60)