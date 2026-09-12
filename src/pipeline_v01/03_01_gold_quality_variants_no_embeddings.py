# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Gold Layer: Quality Variants with Vector Embeddings
# MAGIC %md
# MAGIC # Gold Layer: Quality Variants Chunking & Vector Embeddings
# MAGIC
# MAGIC ## Purpose
# MAGIC Transform silver layer quality variants into semantically meaningful chunks with vector embeddings optimized for RAG retrieval experiments.
# MAGIC
# MAGIC ## Architecture
# MAGIC - **Input**: `workspace.tcc_rag.silver_quality_variants` (all quality variants)
# MAGIC - **Output**: `workspace.tcc_rag.gold_quality_variants_chunks` (chunks + embeddings)
# MAGIC - **Index**: Single Vector Search index covering all variants with `quality_variant` as filterable metadata
# MAGIC
# MAGIC ## Chunking Strategy
# MAGIC - **Method**: Recursive character splitting with gazette-specific separators
# MAGIC - **Chunk size**: 800 tokens (~3200 chars)
# MAGIC - **Overlap**: 200 tokens (~800 chars, 25% overlap to preserve context)
# MAGIC - **Consistency**: Identical chunking logic applied to all variants (only text content differs)
# MAGIC
# MAGIC ## Embedding Generation
# MAGIC - **Model**: `databricks-bge-large-en` (1024 dimensions)
# MAGIC - **Batch size**: 50-100 chunks per inference call
# MAGIC - **Parallelization**: pandas_udf for distributed processing
# MAGIC
# MAGIC ## Idempotency
# MAGIC - **Strategy**: MERGE INTO (upsert) keyed by `(gazette_id, quality_variant, chunk_id)`
# MAGIC - **Benefit**: Reprocessing only computes new/changed chunks, skips existing embeddings

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
# Imports
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType, 
    FloatType, TimestampType, DateType
)
from delta.tables import DeltaTable
import json

# Table configuration
SILVER_TABLE = "workspace.tcc_rag.silver_quality_variants"
GOLD_TABLE = "workspace.tcc_rag.gold_quality_variants_chunks"

# Chunking parameters (consistent with baseline)
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 200  # tokens (25% overlap)
AVG_CHARS_PER_TOKEN = 4  # Portuguese text approximation

CHUNK_SIZE_CHARS = CHUNK_SIZE * AVG_CHARS_PER_TOKEN  # ~3200 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * AVG_CHARS_PER_TOKEN  # ~800 chars

# Embedding configuration
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"
EMBEDDING_BATCH_SIZE = 75  # chunks per inference call
EMBEDDING_DIMENSION = 1024  # bge-large-en output dimension

# Vector Search configuration
VECTOR_SEARCH_ENDPOINT = "querido_diario_endpoint"
VECTOR_INDEX_NAME = "workspace.tcc_rag.quality_variants_vector_index"

print(f"✓ Configuration loaded")
print(f"  Source: {SILVER_TABLE}")
print(f"  Target: {GOLD_TABLE}")
print(f"  Chunk size: {CHUNK_SIZE} tokens (~{CHUNK_SIZE_CHARS} chars)")
print(f"  Overlap: {CHUNK_OVERLAP} tokens (~{CHUNK_OVERLAP_CHARS} chars)")
print(f"  Embedding model: {EMBEDDING_MODEL_ENDPOINT} ({EMBEDDING_DIMENSION}D)")
print(f"  Batch size: {EMBEDDING_BATCH_SIZE} chunks/call")

# COMMAND ----------

# DBTITLE 1,Deterministic Chunking Function
def chunk_text_deterministic(text, gazette_id, quality_variant, 
                             chunk_size_chars=CHUNK_SIZE_CHARS, 
                             overlap_chars=CHUNK_OVERLAP_CHARS):
    """
    Split text into overlapping chunks using recursive character splitting.
    Identical logic applied to all quality variants - only input text differs.
    
    Generates deterministic chunk_ids: {gazette_id}_{quality_variant}_chunk_{index}
    
    Strategy:
    1. Try to split on gazette-specific separators (articles, sections)
    2. Fall back to paragraph/sentence boundaries
    3. Apply sliding window with overlap
    
    Returns: List of (chunk_id, chunk_text, chunk_index, chunk_char_count)
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Gazette-specific separators (hierarchical, try in order)
    separators = [
        "\n\n\n",        # Multiple blank lines (section breaks)
        "Art. ",         # Article markers
        "Artigo ",       # Alternative article markers  
        "\n\n",          # Paragraph breaks
        ". \n",          # Sentence + newline
        ".\n",           # Sentence + newline (no space)
        "\n",            # Line breaks
        ". ",            # Sentences
        " ",             # Words
    ]
    
    MAX_CHUNKS_PER_DOC = 2000  # Safety limit
    min_step = max(chunk_size_chars - overlap_chars, 1)  # Guaranteed forward progress
    
    chunks = []
    chunk_index = 0
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size_chars, text_len)
        raw_chunk = text[start:end]
        
        # Try to break at a good boundary near the end
        best_break = None
        if end < text_len:
            search_start = max(0, len(raw_chunk) - int(chunk_size_chars * 0.2))
            for sep in separators:
                sep_pos = raw_chunk.rfind(sep, search_start)
                if sep_pos > 0:
                    best_break = sep_pos + len(sep)
                    break
        
        chunk_text = raw_chunk[:best_break] if best_break else raw_chunk
        chunk_text_stripped = chunk_text.strip()
        
        if len(chunk_text_stripped) > 50:  # Minimum chunk size filter
            # Deterministic chunk_id includes quality_variant
            chunk_id = f"{gazette_id}_{quality_variant}_chunk_{chunk_index}"
            chunks.append((
                chunk_id,
                chunk_text_stripped,
                chunk_index,
                len(chunk_text_stripped)
            ))
            chunk_index += 1
        
        # Advance with guaranteed minimum step
        consumed = len(chunk_text) if len(chunk_text) > 0 else chunk_size_chars
        step = max(consumed - overlap_chars, min_step)
        start += step
        
        # Safety valve
        if chunk_index >= MAX_CHUNKS_PER_DOC:
            print(f"⚠ {gazette_id}_{quality_variant} hit MAX_CHUNKS_PER_DOC ({MAX_CHUNKS_PER_DOC})")
            break
    
    return chunks

print("✓ Deterministic chunking function defined")

# COMMAND ----------

# DBTITLE 1,Batch Embedding Generation with pandas_udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import numpy as np
from mlflow.deployments import get_deploy_client

# Initialize deployment client (once, outside UDF)
client = get_deploy_client("databricks")

@pandas_udf(ArrayType(FloatType()))
def generate_embeddings_batch(texts: pd.Series) -> pd.Series:
    """
    Generate embeddings for a batch of texts using databricks-bge-large-en.
    
    Uses pandas_udf to parallelize across Spark partitions.
    Each partition processes texts in sub-batches of EMBEDDING_BATCH_SIZE.
    
    Args:
        texts: pandas Series of text strings
    
    Returns:
        pandas Series of embedding arrays (each array has EMBEDDING_DIMENSION floats)
    """
    def generate_batch(batch_texts):
        """Helper to call model endpoint with a batch of texts"""
        try:
            response = client.predict(
                endpoint=EMBEDDING_MODEL_ENDPOINT,
                inputs={"input": batch_texts}
            )
            # Response format: {"data": [{"embedding": [...]}, ...]}
            embeddings = [item["embedding"] for item in response["data"]]
            return embeddings
        except Exception as e:
            print(f"⚠ Error generating embeddings: {str(e)}")
            # Return zero vectors on error to avoid data loss
            return [[0.0] * EMBEDDING_DIMENSION] * len(batch_texts)
    
    # Process in sub-batches
    all_embeddings = []
    texts_list = texts.tolist()
    
    for i in range(0, len(texts_list), EMBEDDING_BATCH_SIZE):
        batch = texts_list[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = generate_batch(batch)
        all_embeddings.extend(batch_embeddings)
    
    return pd.Series(all_embeddings)

print("✓ Batch embedding UDF registered")
print(f"  Model endpoint: {EMBEDDING_MODEL_ENDPOINT}")
print(f"  Batch size: {EMBEDDING_BATCH_SIZE} chunks per call")
print(f"  Output dimension: {EMBEDDING_DIMENSION}D vectors")

# COMMAND ----------

# DBTITLE 1,Load Silver Variants and Apply Chunking
# Load silver quality variants
silver_df = spark.table(SILVER_TABLE)

record_count = silver_df.count()
variant_types = silver_df.select("quality_variant").distinct().count()
gazette_count = silver_df.select("gazette_id").distinct().count()

print(f"✓ Loaded {record_count:,} records from silver variants")
print(f"  Unique gazettes: {gazette_count}")
print(f"  Quality variants: {variant_types}")
print(f"\nVariant distribution:")
silver_df.groupBy("quality_variant").count().orderBy("quality_variant").show()

# Register chunking UDF
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

chunk_schema = ArrayType(
    StructType([
        StructField("chunk_id", StringType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_char_count", IntegerType(), False)
    ])
)

@F.udf(chunk_schema)
def chunk_udf(text, gazette_id, quality_variant):
    """Spark UDF wrapper for chunking function"""
    return chunk_text_deterministic(text, gazette_id, quality_variant)

print("\n✓ Chunking UDF registered")

# Apply chunking transformation (distributed)
print("\nApplying chunking to all variants...")
chunked_df = (
    silver_df
    .withColumn(
        "chunks",
        chunk_udf(
            F.col("texto"),
            F.col("gazette_id"),
            F.col("quality_variant")
        )
    )
    .withColumn("chunk", F.explode(F.col("chunks")))
    .select(
        F.col("gazette_id"),
        F.col("quality_variant"),
        F.col("chunk.chunk_id").alias("chunk_id"),
        F.col("chunk.chunk_text").alias("chunk_text"),
        F.col("chunk.chunk_index").alias("chunk_index"),
        F.col("chunk.chunk_char_count").alias("chunk_char_count"),
        (F.col("chunk.chunk_char_count") / F.lit(AVG_CHARS_PER_TOKEN)).alias("chunk_token_count"),
        F.current_timestamp().alias("data_processamento")
    )
)

chunk_count = chunked_df.count()
print(f"\n✓ Generated {chunk_count:,} chunks across all variants")
print(f"  Average chunks per gazette-variant: {chunk_count/record_count:.1f}")

# Show sample
print("\nSample chunks:")
chunked_df.select(
    "chunk_id", 
    "quality_variant", 
    "chunk_token_count"
).show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Write Chunks to Table (No Embeddings Yet)
# Step 1: Write chunks WITHOUT embeddings (this is fast - just text storage)
# Step 2 (next cell): Add embeddings in controlled batches

import time

print(f"Writing {chunk_count:,} chunks to {GOLD_TABLE} (WITHOUT embeddings yet)...")
print("This should take < 1 minute\n")

start = time.time()

# Add a NULL embedding column placeholder
chunks_no_embeddings = chunked_df.withColumn(
    "embedding",
    F.lit(None).cast(ArrayType(FloatType()))
)

# Write all chunks at once (no embeddings = fast)
chunks_no_embeddings.write.format("delta").mode("overwrite").saveAsTable(GOLD_TABLE)

elapsed = time.time() - start

print(f"✓ Wrote {chunk_count:,} chunks in {elapsed:.1f} seconds")

# Verify
final_count = spark.table(GOLD_TABLE).count()
print(f"\n✓ Table {GOLD_TABLE} contains {final_count:,} chunks")

print(f"\nDistribution by variant:")
spark.table(GOLD_TABLE).groupBy("quality_variant").count().orderBy("quality_variant").show()

print(f"\nSample (embeddings are NULL for now):")
spark.table(GOLD_TABLE).select(
    "chunk_id", 
    "quality_variant", 
    "chunk_token_count",
    "embedding"
).show(5, truncate=False)

print(f"\n{'='*70}")
print("✓ Chunks written successfully!")
print("⚠ Next: Run Cell 8 to add embeddings in controlled batches")
print(f"{'='*70}")