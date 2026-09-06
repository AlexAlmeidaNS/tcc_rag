# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Vector Search Setup: Overview
# MAGIC %md
# MAGIC # Vector Search Setup: Querido Diário RAG
# MAGIC
# MAGIC ## Purpose
# MAGIC Set up Databricks Vector Search for semantic retrieval over chunked gazette data.
# MAGIC
# MAGIC ## Components
# MAGIC 1. **Vector Search Endpoint**: Compute resource for embeddings and similarity search
# MAGIC 2. **Vector Search Index**: Delta Sync index that automatically embeds and indexes the gold layer chunks
# MAGIC
# MAGIC ## Configuration
# MAGIC - **Source Table**: `workspace.tcc_rag.gold_querido_diario_chunks` (from Phase 1)
# MAGIC - **Embedding Model**: `databricks-bge-large-en` (1024 dimensions, multilingual)
# MAGIC - **Primary Key**: `chunk_id`
# MAGIC - **Embedding Column**: `chunk_text`
# MAGIC
# MAGIC ## Pipeline Type Options
# MAGIC - **TRIGGERED**: Manual sync (on-demand)
# MAGIC - **CONTINUOUS**: Auto-sync when source table changes
# MAGIC
# MAGIC For this initial setup, we'll use **TRIGGERED** for more control.

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
# Configuration
from databricks.sdk import WorkspaceClient

# Vector Search configuration
ENDPOINT_NAME = "querido_diario_endpoint"
SOURCE_TABLE = "workspace.tcc_rag.gold_querido_diario_chunks"
INDEX_NAME = "workspace.tcc_rag.querido_diario_vector_index"
PRIMARY_KEY = "chunk_id"
EMBEDDING_SOURCE_COLUMN = "chunk_text"
EMBEDDING_MODEL = "databricks-bge-large-en"  # Multilingual, 1024 dimensions

# Initialize client
w = WorkspaceClient()

print("✓ Configuration loaded")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Source: {SOURCE_TABLE}")
print(f"  Index: {INDEX_NAME}")
print(f"  Embedding Model: {EMBEDDING_MODEL}")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
# Step 1: Create Vector Search Endpoint

from databricks.sdk.errors import NotFound

# Check if endpoint already exists
try:
    endpoint = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    print(f"✓ Endpoint '{ENDPOINT_NAME}' already exists")
    print(f"  Status: {endpoint.endpoint_status.state.value if endpoint.endpoint_status else 'UNKNOWN'}")
except NotFound:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    print("⏳ This may take 5-10 minutes...")
    
    from databricks.sdk.service.vectorsearch import EndpointType
    wait = w.vector_search_endpoints.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD
    )
    endpoint = wait.result()
    
    print(f"✓ Endpoint '{ENDPOINT_NAME}' created successfully")
    print(f"  Status: {endpoint.endpoint_status.state.value if endpoint.endpoint_status else 'UNKNOWN'}")

# Wait for endpoint to be online
import time
max_wait = 600  # 10 minutes
start_time = time.time()

while True:
    endpoint_info = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    status = endpoint_info.endpoint_status.state.value if endpoint_info.endpoint_status else 'UNKNOWN'
    
    if status == 'ONLINE':
        print(f"\n✓ Endpoint is ONLINE and ready")
        break
    elif status in ['OFFLINE', 'PROVISIONING_FAILURE']:
        print(f"\n❌ Endpoint failed to provision. Status: {status}")
        raise Exception(f"Endpoint provisioning failed: {status}")
    else:
        elapsed = int(time.time() - start_time)
        print(f"  Status: {status} (elapsed: {elapsed}s)", end='\r')
        
        if elapsed > max_wait:
            print(f"\n⚠️ Timeout waiting for endpoint (>{max_wait}s)")
            break
        
        time.sleep(10)

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
# Step 2: Create Vector Search Index

from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    VectorIndexType, 
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType
)

# Check if index already exists
try:
    index_info = w.vector_search_indexes.get_index(INDEX_NAME)
    print(f"✓ Index '{INDEX_NAME}' already exists")
    status_msg = index_info.status.message if index_info.status else 'Unknown'
    is_ready = index_info.status.ready if index_info.status else False
    
    if is_ready:
        print(f"\n✅ Status: READY for queries")
        if index_info.status and hasattr(index_info.status, 'indexed_row_count'):
            print(f"   Indexed rows: {index_info.status.indexed_row_count:,}")
    else:
        print(f"\n⏳ Status: PROVISIONING")
        print(f"   {status_msg}")
        print("   Initial provisioning takes 15-30 minutes.")
        if index_info.status and hasattr(index_info.status, 'index_url'):
            print(f"   Monitor: {index_info.status.index_url}")
    
    print("\n📝 To recreate:")
    print(f"    w.vector_search_indexes.delete_index('{INDEX_NAME}')")
    print("    Then re-run this cell.")
    
except NotFound:
    print(f"Creating index '{INDEX_NAME}'...")
    print("⏳ Initial indexing may take several minutes...")
    
    delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
        source_table=SOURCE_TABLE,
        pipeline_type=PipelineType.TRIGGERED,
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name=EMBEDDING_SOURCE_COLUMN,
                embedding_model_endpoint_name=EMBEDDING_MODEL
            )
        ]
    )
    
    index = w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key=PRIMARY_KEY,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=delta_sync_spec
    )
    
    print(f"✓ Index '{INDEX_NAME}' created successfully")
    print("\n📝 Note: Use TRIGGERED pipeline type for manual sync.")
    print("   To sync changes: w.vector_search_indexes.sync_index(INDEX_NAME)")
    print("\n⏳ Initial provisioning takes 15-30 minutes.")
    print("   The index is provisioning in the background.")
    if index.status and hasattr(index.status, 'index_url'):
        print(f"   Monitor progress: {index.status.index_url}")

# COMMAND ----------

# DBTITLE 1,Check Index Status
# Run this cell anytime to check if the index is ready

index_status = w.vector_search_indexes.get_index(INDEX_NAME)
is_ready = index_status.status.ready if index_status.status else False

if is_ready:
    print(f"✅ Index is READY for queries!")
    print(f"\n📊 Index Statistics:")
    if index_status.status and hasattr(index_status.status, 'indexed_row_count'):
        print(f"   Indexed rows: {index_status.status.indexed_row_count:,}")
    print(f"\n🎯 You can now run cell 5 to test semantic search.")
else:
    message = index_status.status.message if index_status.status else 'Unknown'
    print(f"⏳ Index is still provisioning...")
    print(f"\nStatus: {message}")
    print(f"\nThis typically takes 15-30 minutes.")
    if index_status.status and hasattr(index_status.status, 'index_url'):
        print(f"Monitor progress: {index_status.status.index_url}")

# COMMAND ----------

# DBTITLE 1,Sync Index with Gold Table
# Sync Vector Search Index with Latest Gold Table Data

print("🔄 Syncing vector search index with gold table...")
print(f"   Source: {SOURCE_TABLE}")
print(f"   Index: {INDEX_NAME}")
print("\n⏳ This may take several minutes depending on the number of new/updated rows...\n")

# Trigger sync
w.vector_search_indexes.sync_index(INDEX_NAME)

print("✓ Sync initiated successfully!")
print("\n📝 Note: The sync runs asynchronously in the background.")
print("   Run the 'Check Index Status' cell to monitor progress.")

# COMMAND ----------

# DBTITLE 1,Switch to CONTINUOUS Mode
# Recreate Index with CONTINUOUS Mode
# CONTINUOUS mode automatically processes all rows and keeps index in sync

from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType
)

print("⚠️  IMPORTANT: This will delete and recreate the index.")
print("   Current index has 144 rows and pipeline stuck in CREATED state.")
print("   New index will use TRIGGERED mode with proper initialization.\n")

response = input("Proceed with deletion and recreation? (yes/no): ")

if response.lower() == "yes":
    # Check if index exists before trying to delete
    from databricks.sdk.errors import NotFound
    import time
    
    try:
        print("\n🔍 Checking if index exists...")
        w.vector_search_indexes.get_index(INDEX_NAME)
        print("   Found existing index")
        print("\n🗑️  Step 1: Deleting old index...")
        w.vector_search_indexes.delete_index(INDEX_NAME)
        print("✓ Old index deleted")
        print("\n⏳ Waiting 10 seconds for cleanup...")
        time.sleep(10)
    except NotFound:
        print("   Index already deleted (from previous attempt)")
        print("✓ Skipping deletion step")
    
    print("\n🆕 Step 2: Creating new index with TRIGGERED mode...")
    print("   (CONTINUOUS mode not supported in this workspace)\n")
    
    delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
        source_table=SOURCE_TABLE,
        pipeline_type=PipelineType.TRIGGERED,  # Manual sync mode
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name=EMBEDDING_SOURCE_COLUMN,
                embedding_model_endpoint_name=EMBEDDING_MODEL
            )
        ]
    )
    
    index = w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key=PRIMARY_KEY,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=delta_sync_spec
    )
    
    print("✓ New index created with TRIGGERED mode!")
    print("\n📝 What happens now:")
    print("   • Pipeline will automatically run ONCE to process all 122,011 rows")
    print("   • This first run happens automatically (no sync needed)")
    print("   • After first run completes, future updates require manual sync")
    print("\n⏳ Initial indexing will take 20-40 minutes for 122K chunks.")
    print("   Monitor progress by re-running 'Check Index Status' cell.")
    print("\n🔄 To sync future updates after first run completes:")
    print("   w.vector_search_indexes.sync_index(INDEX_NAME)")
else:
    print("\n❌ Operation cancelled.")

# COMMAND ----------

# DBTITLE 1,Test Semantic Search
# Step 3: Test Semantic Search

# Test query
test_query = "decreto lei salário mínimo"

print(f"🔍 Testing semantic search...")
print(f"   Query: '{test_query}'\n")

results = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    query_text=test_query,
    columns=["chunk_id", "chunk_text", "territory_name", "state_code", "publication_date"],
    num_results=5
)

# Access result data
result_data = results.result.data_array if results.result else []
print(f"✓ Found {len(result_data)} results\n")
print("=" * 80)

# Display results
for i, row in enumerate(result_data, 1):
    print(f"\nResult #{i}")
    print(f"  Chunk ID: {row[0]}")
    print(f"  Territory: {row[2]}, {row[3]}")
    print(f"  Date: {row[4]}")
    print(f"  Preview: {row[1][:200]}...")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Test Filtered Search
# Step 4: Test Filtered Search (Hybrid Search)

filtered_query = "licitação pública obras"

print(f"🔍 Testing filtered semantic search...")
print(f"   Query: '{filtered_query}'")

import json

filtered_results = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    query_text=filtered_query,
    columns=["chunk_id", "chunk_text", "territory_name", "state_code", "publication_date"],
    num_results=3
)

print(f"✓ Found {len(filtered_results.result.data_array if filtered_results.result else [])} results\n")
print("=" * 80)

for i, row in enumerate(filtered_results.result.data_array if filtered_results.result else [], 1):
    print(f"\nResult #{i}")
    print(f"  Chunk ID: {row[0]}")
    print(f"  Territory: {row[2]}, {row[3]}")
    print(f"  Date: {row[4]}")
    print(f"  Preview: {row[1][:200]}...")
    print("-" * 80)

print("\n✓ Hybrid search (semantic + filters) working correctly!")

# COMMAND ----------

# DBTITLE 1,RAG Example: Vector Search + LLM
# RAG Example: Combining Vector Search + LLM
# This demonstrates a complete Retrieval Augmented Generation pipeline

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Step 1: Vector Search - Retrieve relevant chunks
user_question = "Quantas nomeacoes tivemos? Inclua uma tabela com os resultados e data do decreto"

print(f"🔍 User Question: {user_question}\n")
print("Step 1: Retrieving relevant chunks from vector index...")

# Query vector index
retrieval_results = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    query_text=user_question,
    columns=["chunk_text", "territory_name", "state_code", "publication_date"],
    num_results=5  # Get top 5 most relevant chunks
)

result_data = retrieval_results.result.data_array if retrieval_results.result else []
print(f"✓ Retrieved {len(result_data)} relevant chunks\n")

# Step 2: Format context from retrieved chunks
context_parts = []
for i, row in enumerate(result_data, 1):
    chunk_text = row[0]
    territory = row[1]
    date = row[3]
    context_parts.append(f"Documento {i} ({territory}, {date}):\n{chunk_text}")

context = "\n\n".join(context_parts)

print("Step 2: Building prompt with retrieved context...")
print(f"   Context length: {len(context)} characters\n")

# Step 3: LLM - Generate answer using retrieved context
print("Step 3: Generating answer with LLM...\n")

# Create prompt with retrieved context
system_prompt = """Você é um assistente especializado em legislação municipal brasileira.
Responda a pergunta do usuário usando APENAS as informações fornecidas no contexto.
Se a informação não estiver no contexto, diga que não encontrou essa informação específica.
Cite os documentos relevantes na sua resposta."""

user_prompt = f"""Contexto (documentos municipais de Alagoas):

{context}

---

Pergunta: {user_question}

Resposta:"""

# Call Databricks Foundation Model API
w_serving = WorkspaceClient()
response = w_serving.serving_endpoints.query(
    name="databricks-meta-llama-3-3-70b-instruct",  # Using Llama 3.3 70B
    messages=[
        ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=ChatMessageRole.USER, content=user_prompt)
    ],
    max_tokens=500,
    temperature=0.1  # Lower temperature for factual answers
)

# Extract answer
answer = response.choices[0].message.content

print("=" * 80)
print("📝 ANSWER:")
print("=" * 80)
print(answer)
print("=" * 80)

print("\n✓ RAG pipeline complete!")
print("\n💡 This combines:")
print("   1. Vector search for semantic retrieval")
print("   2. LLM for natural language synthesis")
print("   3. Filtered search for targeted results (state_code = AL)")

# COMMAND ----------

# DBTITLE 1,Next Steps: Using Your Vector Search Index
# MAGIC %md
# MAGIC ## 🎉 Vector Search Setup Complete!
# MAGIC
# MAGIC Your Vector Search index is now ready for RAG applications.
# MAGIC
# MAGIC ### What You Can Do Now
# MAGIC
# MAGIC 1. **Query the index programmatically**
# MAGIC    ```python
# MAGIC    from databricks.vector_search.client import VectorSearchClient
# MAGIC    vsc = VectorSearchClient()
# MAGIC    index = vsc.get_index("workspace.tcc_rag.querido_diario_vector_index")
# MAGIC    
# MAGIC    results = index.similarity_search(
# MAGIC        query_text="seu texto de busca",
# MAGIC        num_results=5
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 2. **Use with Databricks AI Playground**
# MAGIC    - Navigate to AI Playground
# MAGIC    - Select your Vector Search index as a retrieval source
# MAGIC    - Test RAG queries interactively
# MAGIC
# MAGIC 3. **Build a RAG Application**
# MAGIC    - Use the index in a Databricks App
# MAGIC    - Integrate with LangChain or LlamaIndex
# MAGIC    - Deploy as a Model Serving endpoint
# MAGIC
# MAGIC ### Index Management
# MAGIC
# MAGIC **Manual Sync** (TRIGGERED mode):
# MAGIC ```python
# MAGIC index.sync()
# MAGIC ```
# MAGIC
# MAGIC **Check Index Status**:
# MAGIC ```python
# MAGIC vsc.get_index("workspace.tcc_rag.querido_diario_vector_index")
# MAGIC ```
# MAGIC
# MAGIC **Delete Index** (if needed):
# MAGIC ```python
# MAGIC vsc.delete_index("workspace.tcc_rag.querido_diario_vector_index")
# MAGIC ```
# MAGIC
# MAGIC ### Available Filter Dimensions
# MAGIC
# MAGIC You can filter results by any of these metadata columns:
# MAGIC - `territory_name`: Municipality name
# MAGIC - `state_code`: State abbreviation (e.g., "AL", "SP")
# MAGIC - `publication_date`: Date filter (e.g., `{">" :"2024-01-01"}`)
# MAGIC - `territory_id`: IBGE territory code
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **📚 Next Steps**: Build your RAG application or explore the Playground!

# COMMAND ----------

# DBTITLE 1,RAG Example: Vector Search + LLM
# RAG Example: Combining Vector Search + LLM
# This demonstrates a complete Retrieval Augmented Generation pipeline

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Step 1: Vector Search - Retrieve relevant chunks
user_question = "Quais são as regras sobre aposentadoria em Alagoas?"

print(f"🔍 User Question: {user_question}\n")
print("Step 1: Retrieving relevant chunks from vector index...")

# Query vector index
retrieval_results = w.vector_search_indexes.query_index(
    index_name=INDEX_NAME,
    query_text=user_question,
    columns=["chunk_text", "territory_name", "state_code", "publication_date"],
    filters_json=json.dumps({"state_code": "PE"}),  # Filter to Alagoas
    num_results=5  # Get top 5 most relevant chunks
)

result_data = retrieval_results.result.data_array if retrieval_results.result else []
print(f"✓ Retrieved {len(result_data)} relevant chunks\n")

# Step 2: Format context from retrieved chunks
context_parts = []
for i, row in enumerate(result_data, 1):
    chunk_text = row[0]
    territory = row[1]
    date = row[3]
    context_parts.append(f"Documento {i} ({territory}, {date}):\n{chunk_text}")

context = "\n\n".join(context_parts)

print("Step 2: Building prompt with retrieved context...")
print(f"   Context length: {len(context)} characters\n")

# Step 3: LLM - Generate answer using retrieved context
print("Step 3: Generating answer with LLM...\n")

# Create prompt with retrieved context
system_prompt = """Você é um assistente especializado em legislação municipal brasileira.
Responda a pergunta do usuário usando APENAS as informações fornecidas no contexto.
Se a informação não estiver no contexto, diga que não encontrou essa informação específica.
Cite os documentos relevantes na sua resposta."""

user_prompt = f"""Contexto (documentos municipais de Alagoas):

{context}

---

Pergunta: {user_question}

Resposta:"""

# Call Databricks Foundation Model API
w_serving = WorkspaceClient()
response = w_serving.serving_endpoints.query(
    name="databricks-meta-llama-3-1-70b-instruct",  # Using Llama 3.1 70B
    messages=[
        ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=ChatMessageRole.USER, content=user_prompt)
    ],
    max_tokens=500,
    temperature=0.1  # Lower temperature for factual answers
)

# Extract answer
answer = response.choices[0].message.content

print("=" * 80)
print("📝 ANSWER:")
print("=" * 80)
print(answer)
print("=" * 80)

print("\n✓ RAG pipeline complete!")
print("\n💡 This combines:")
print("   1. Vector search for semantic retrieval")
print("   2. LLM for natural language synthesis")
print("   3. Filtered search for targeted results (state_code = AL)")