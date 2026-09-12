# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Gold Layer: Chunking for RAG
# MAGIC %md
# MAGIC # Gold Layer: Querido Diário Chunking for RAG
# MAGIC
# MAGIC ## Purpose
# MAGIC Transform silver layer gazette data into semantically meaningful chunks optimized for RAG retrieval.
# MAGIC
# MAGIC ## Chunking Strategy
# MAGIC - **Method**: Recursive character splitting with gazette-specific separators
# MAGIC - **Chunk size**: 800 tokens (~600 words)
# MAGIC - **Overlap**: 200 tokens (25% overlap to preserve context at boundaries)
# MAGIC - **Metadata preservation**: All silver layer metadata attached to each chunk
# MAGIC
# MAGIC ## Input
# MAGIC - **Table**: `workspace.tcc_rag.silver_querido_diario`
# MAGIC - **Filters**: Only gazettes with extracted text content
# MAGIC
# MAGIC ## Output
# MAGIC - **Table**: `workspace.tcc_rag.gold_querido_diario_chunks`
# MAGIC - **Schema**: Chunks with metadata for Vector Search indexing

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
# Imports
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, TimestampType
from datetime import datetime
import uuid
import re

# Configuration
SILVER_TABLE = "workspace.tcc_rag.silver_querido_diario"
GOLD_TABLE = "workspace.tcc_rag.gold_querido_diario_chunks"

# Chunking parameters (as per chunking_strategies_guide.md)
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 200  # tokens (25% overlap)
AVG_CHARS_PER_TOKEN = 4  # Approximation for Portuguese text

# Convert token sizes to character counts for implementation
CHUNK_SIZE_CHARS = CHUNK_SIZE * AVG_CHARS_PER_TOKEN  # ~3200 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * AVG_CHARS_PER_TOKEN  # ~800 chars

print(f"✓ Configuration loaded")
print(f"  Source: {SILVER_TABLE}")
print(f"  Target: {GOLD_TABLE}")
print(f"  Chunk size: {CHUNK_SIZE} tokens (~{CHUNK_SIZE_CHARS} chars)")
print(f"  Overlap: {CHUNK_OVERLAP} tokens (~{CHUNK_OVERLAP_CHARS} chars)")

# COMMAND ----------

# DBTITLE 1,Chunking Function with Overlap
def chunk_text_with_overlap(text, gazette_id, chunk_size_chars=CHUNK_SIZE_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS):
    """
    Split text into overlapping chunks using recursive character splitting.
    
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
    
    chunks = []
    chunk_index = 0
    
    # Simple sliding window implementation
    start = 0
    while start < len(text):
        # Calculate end position
        end = start + chunk_size_chars
        
        # If this is not the first chunk, move start back by overlap amount
        if start > 0:
            start = max(0, start - overlap_chars)
            end = start + chunk_size_chars
        
        # Don't exceed text length
        chunk_text = text[start:min(end, len(text))]
        
        # Try to break at a good boundary (prefer separators)
        if end < len(text):  # Not the last chunk
            # Look for a separator near the end
            best_break = None
            for sep in separators:
                # Look in the last 20% of the chunk for a good break point
                search_start = len(chunk_text) - int(chunk_size_chars * 0.2)
                sep_pos = chunk_text.rfind(sep, search_start)
                if sep_pos > 0:
                    best_break = sep_pos + len(sep)
                    break
            
            if best_break:
                chunk_text = chunk_text[:best_break]
        
        # Clean and add chunk
        chunk_text = chunk_text.strip()
        if len(chunk_text) > 50:  # Minimum chunk size (filter very small chunks)
            chunk_id = f"{gazette_id}_chunk_{chunk_index}"
            chunks.append((
                chunk_id,
                chunk_text,
                chunk_index,
                len(chunk_text)
            ))
            chunk_index += 1
        
        # Move to next chunk
        start = start + len(chunk_text)
        
        # Safety: prevent infinite loop
        if start >= len(text) or len(chunk_text) == 0:
            break
    
    return chunks

print("✓ Chunking function defined")

# COMMAND ----------

# DBTITLE 1,Load Silver Data
# Load silver layer data
# Filter to only gazettes with successfully extracted text content
silver_df = (
    spark.table(SILVER_TABLE)
    .filter(F.col("has_content") == True)
    .filter(F.col("text_content").isNotNull())
    .filter(F.length(F.col("text_content")) > 100)  # Minimum text length
)

silver_count = silver_df.count()
print(f"✓ Loaded {silver_count:,} gazettes from silver layer with text content")

# Show sample
print("\nSample silver records:")
silver_df.select(
    "gazette_id",
    "territory_name", 
    "publication_date",
    "text_length",
    "word_count"
).show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Apply Chunking Transformation
def chunk_text_with_overlap(text, gazette_id, chunk_size_chars=CHUNK_SIZE_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS):
    """
    Split text into overlapping chunks using recursive character splitting.

    Strategy:
    1. Try to split on gazette-specific separators (articles, sections)
    2. Fall back to paragraph/sentence boundaries
    3. Apply sliding window with overlap, guaranteeing forward progress

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

    # Safety cap: prevents runaway chunk generation for pathological documents
    # (e.g. dense OCR text that keeps triggering early separator matches)
    MAX_CHUNKS_PER_DOC = 2000

    # Guaranteed minimum forward step per iteration, regardless of how short
    # a boundary-trimmed chunk turns out to be. This is what actually
    # prevents the stall/near-zero-progress bug.
    min_step = max(chunk_size_chars - overlap_chars, 1)

    chunks = []
    chunk_index = 0
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size_chars, text_len)
        raw_chunk = text[start:end]

        # Try to break at a good boundary near the end of the window
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
            chunk_id = f"{gazette_id}_chunk_{chunk_index}"
            chunks.append((
                chunk_id,
                chunk_text_stripped,
                chunk_index,
                len(chunk_text_stripped)
            ))
            chunk_index += 1

        # Advance from a single source of truth (start), with a guaranteed
        # minimum step. This is the key fix: the old code re-derived start
        # from an already-advanced value and could double-subtract overlap,
        # letting net progress go to zero or negative on short trimmed chunks.
        consumed = len(chunk_text) if len(chunk_text) > 0 else chunk_size_chars
        step = max(consumed - overlap_chars, min_step)
        start += step

        # Hard safety valve: if a document somehow still produces an
        # excessive number of chunks, stop instead of exhausting memory.
        if chunk_index >= MAX_CHUNKS_PER_DOC:
            print(f"⚠ gazette_id={gazette_id} hit MAX_CHUNKS_PER_DOC "
                  f"({MAX_CHUNKS_PER_DOC}); text may be malformed or "
                  f"chunk_size_chars/overlap_chars may need tuning.")
            break

    return chunks


print("✓ Chunking function defined")

# COMMAND ----------

# DBTITLE 1,Write to Gold Table
from delta.tables import DeltaTable

# Drop existing table to start completely fresh (avoids schema conflicts)
if spark.catalog.tableExists(GOLD_TABLE):
    spark.sql(f"DROP TABLE {GOLD_TABLE}")
    print(f"✓ Dropped existing table {GOLD_TABLE}")

table_exists = False  # Will be created on first batch

# Process gazettes one at a time to avoid OOM
BATCH_SIZE = 1  # Process one gazette at a time
total_gazettes = 118 # From silver_df count

print(f"Processing {total_gazettes} gazettes one at a time to avoid OOM...")
print("This may take a while but will be reliable.\n")

# Process using monotonic ID assignment in SQL
for batch_num in range(total_gazettes):
    print(f"Processing gazette {batch_num + 1}/{total_gazettes}...", end=" ")

    # if (batch_num + 1) == 25:
    #     print(f"Skipping gazette {batch_num + 1} (known problematic)...")
    #     continue
    
    # Get one gazette using limit and offset directly in SQL
    batch_df = silver_df.orderBy("gazette_id").offset(batch_num).limit(1)
    
    # Apply chunking
    # Collect the single row, apply chunking function, then convert back to DataFrame
    gazette_row = batch_df.collect()[0]
    gazette_id = gazette_row.gazette_id
    text_content = gazette_row.text_content
    
    # Apply chunking function (returns list of tuples)
    chunks_list = chunk_text_with_overlap(text_content, gazette_id)
    
    # Convert to DataFrame with all metadata from silver layer
    if len(chunks_list) > 0:
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
        
        # Define the COMPLETE schema upfront with all columns
        complete_schema = StructType([
            StructField("chunk_id", StringType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_char_count", IntegerType(), False),
            StructField("gazette_id", StringType(), False),
            StructField("territory_name", StringType(), True),
            StructField("state_code", StringType(), True),
            StructField("publication_date", DateType(), True),
            StructField("chunk_token_count", DoubleType(), False)
        ])
        
        # Build complete rows with all columns at once
        complete_rows = [
            (
                chunk_id,
                chunk_text,
                chunk_index,
                chunk_char_count,
                gazette_id,
                gazette_row.territory_name,
                gazette_row.state_code,
                gazette_row.publication_date,
                float(chunk_char_count) / float(AVG_CHARS_PER_TOKEN)
            )
            for chunk_id, chunk_text, chunk_index, chunk_char_count in chunks_list
        ]
        
        # Create DataFrame with complete schema in one shot
        chunked_batch = spark.createDataFrame(complete_rows, schema=complete_schema)
    else:
        # Skip if no chunks produced
        print(f"  ⚠ No chunks produced, skipping")
        continue
    
    # Write to table (always append or create if doesn't exist)
    if not table_exists:
        # First batch - create table
        (
            chunked_batch.write
            .format("delta")
            .mode("overwrite")
            .option("delta.enableChangeDataFeed", "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact", "true")
            .saveAsTable(GOLD_TABLE)
        )
        table_exists = True
        print("  ✓ Created table with first batch")
        
        # Add table comment
        spark.sql(f"""
            COMMENT ON TABLE {GOLD_TABLE} IS 
            'Gold layer: Semantically chunked gazette text optimized for RAG retrieval with Vector Search'
        """)
    else:


        # chunk_count = chunked_batch.count()
        # print(f"  ✓ Pre Append {chunk_count} chunks")

        print(f"  ✓ Pre Append")

        # Append to existing table
        (
            chunked_batch.write
            .format("delta")
            .mode("append")
            .saveAsTable(GOLD_TABLE)
        )
        # chunk_count = chunked_batch.count()
        print(f"  ✓ Appended")


print(f"\n{'='*60}")
print("✓ All batches processed successfully")
print(f"{'='*60}")

# Verify write
final_count = spark.table(GOLD_TABLE).count()
gazette_count = spark.table(GOLD_TABLE).select("gazette_id").distinct().count()
print(f"\nFinal Statistics:")
print(f"  Total chunks: {final_count:,}")
print(f"  Unique gazettes: {gazette_count:,}")
print(f"  Avg chunks per gazette: {final_count/gazette_count:.1f}")

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Load gold table for validation
from pyspark.sql.functions import col

gold_df = spark.table(GOLD_TABLE)

print("=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# 1. Chunk size distribution
print("\n1. Chunk Token Count Distribution:")
chunk_dist = (
    gold_df
    .withColumn(
        "token_bucket",
        F.when(col("chunk_token_count") < 400, "< 400")
        .when(col("chunk_token_count") < 600, "400-600")
        .when(col("chunk_token_count") < 800, "600-800")
        .when(col("chunk_token_count") < 1000, "800-1000")
        .otherwise("> 1000")
    )
    .groupBy("token_bucket")
    .count()
    .orderBy("token_bucket")
)
display(chunk_dist)

# 2. Summary statistics
print("\n2. Chunk Statistics:")
gold_df.select(
    F.count("*").alias("total_chunks"),
    F.countDistinct("gazette_id").alias("unique_gazettes"),
    F.avg("chunk_token_count").alias("avg_tokens_per_chunk"),
    F.min("chunk_token_count").alias("min_tokens"),
    F.max("chunk_token_count").alias("max_tokens"),
    F.avg("chunk_char_count").alias("avg_chars_per_chunk")
).show(truncate=False)

# 3. Territory distribution
print("\n3. Chunks by Territory (Top 10):")
territory_dist = (
    gold_df
    .groupBy("territory_name", "state_code")
    .agg(
        F.count("*").alias("chunk_count"),
        F.countDistinct("gazette_id").alias("gazette_count")
    )
    .orderBy(F.desc("chunk_count"))
    .limit(10)
)
display(territory_dist)

# 4. Check for chunks with unusual sizes
print("\n4. Quality Checks:")
outliers = gold_df.filter(
    (col("chunk_token_count") < 50) | (col("chunk_token_count") > 1200)
).count()

print(f"  Chunks < 50 tokens or > 1200 tokens: {outliers}")

if outliers > 0:
    print("  ⚠️  Warning: Found outlier chunks. Review chunking logic.")
    gold_df.filter(
        (col("chunk_token_count") < 50) | (col("chunk_token_count") > 1200)
    ).select(
        "chunk_id", "chunk_token_count", "chunk_char_count"
    ).show(10)
else:
    print("  ✓ All chunks within expected size range")

# 5. Sample chunks for manual inspection
print("\n5. Sample Chunks (random 3):")
sample_chunks = gold_df.sample(0.01).limit(3)
for row in sample_chunks.collect():
    print(f"\nChunk ID: {row.chunk_id}")
    print(f"Territory: {row.territory_name}, {row.state_code}")
    print(f"Date: {row.publication_date}")
    print(f"Tokens: {row.chunk_token_count}")
    print(f"Preview: {row.chunk_text[:200]}...")
    print("-" * 60)

print("\n✓ Data quality checks completed")

# COMMAND ----------

# DBTITLE 1,Next Steps: Vector Search Setup
# MAGIC %md
# MAGIC ## 🚀 Next Steps: Vector Search Setup
# MAGIC
# MAGIC Now that the gold layer is ready with chunked data, the next phase is to set up Vector Search:
# MAGIC
# MAGIC ### Phase 2: Vector Search Index Creation
# MAGIC
# MAGIC 1. **Create Vector Search Endpoint**
# MAGIC    ```python
# MAGIC    from databricks.vector_search.client import VectorSearchClient
# MAGIC    
# MAGIC    vsc = VectorSearchClient()
# MAGIC    vsc.create_endpoint(
# MAGIC        name="querido_diario_endpoint",
# MAGIC        endpoint_type="STANDARD"
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 2. **Create Vector Search Index**
# MAGIC    ```python
# MAGIC    index = vsc.create_delta_sync_index(
# MAGIC        endpoint_name="querido_diario_endpoint",
# MAGIC        source_table_name="workspace.tcc_rag.gold_querido_diario_chunks",
# MAGIC        index_name="workspace.tcc_rag.querido_diario_vector_index",
# MAGIC        pipeline_type="TRIGGERED",
# MAGIC        primary_key="chunk_id",
# MAGIC        embedding_source_column="chunk_text",
# MAGIC        embedding_model_endpoint_name="databricks-bge-large-en"
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 3. **Test Retrieval**
# MAGIC    ```python
# MAGIC    results = index.similarity_search(
# MAGIC        query_text="decreto lei salário mínimo",
# MAGIC        columns=["chunk_id", "chunk_text", "territory_name", "publication_date"],
# MAGIC        num_results=5
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC ### Key Configuration Decisions
# MAGIC
# MAGIC * **Embedding Model**: `databricks-bge-large-en` (1024 dimensions, good for Portuguese)
# MAGIC * **Primary Key**: `chunk_id` (unique identifier for each chunk)
# MAGIC * **Source Column**: `chunk_text` (the text to embed)
# MAGIC * **Pipeline Type**: `TRIGGERED` (manual sync) or `CONTINUOUS` (auto-sync)
# MAGIC
# MAGIC ### Metadata Filtering
# MAGIC
# MAGIC Vector Search supports filtering by metadata columns:
# MAGIC ```python
# MAGIC results = index.similarity_search(
# MAGIC     query_text="licitação pública",
# MAGIC     filters={"state_code": "DF", "publication_date": {">": "2024-01-01"}},
# MAGIC     num_results=10
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC This enables hybrid search: semantic similarity + structured filters!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **📚 Reference**: See `chunking_strategies_guide.md` for detailed chunking rationale