# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Avaliação de Qualidade RAG com RAGAS
# MAGIC %md
# MAGIC # Avaliação de Aderência das Respostas RAG por Cenário de Qualidade de Dados
# MAGIC
# MAGIC ## Objetivo
# MAGIC Avaliar comparativamente a qualidade das respostas geradas por um pipeline RAG (Retrieval-Augmented Generation) 
# MAGIC sobre dados do portal **Querido Diário**, considerando quatro cenários de qualidade de dados:
# MAGIC
# MAGIC | Cenário | Descrição |
# MAGIC | --- | --- |
# MAGIC | **baseline** | Dados originais, como importados da camada bronze |
# MAGIC | **ocr_noise** | Simulação de erros de OCR (caracteres trocados, palavras corrompidas) |
# MAGIC | **duplicate** | Simulação de dados duplicados (trechos repetidos no texto) |
# MAGIC | **missing_fields** | Simulação de exclusão de campos chaves do texto |
# MAGIC
# MAGIC ## Metodologia
# MAGIC 1. **Recuperação**: Para cada pergunta-teste, recuperar chunks via Vector Search filtrando por `quality_variant`
# MAGIC 2. **Geração**: Gerar respostas com LLM (Llama 3.3 70B) usando o contexto recuperado
# MAGIC 3. **Avaliação**: Aplicar métricas do **RAGAS** (faithfulness, answer relevance, context precision, context recall)
# MAGIC 4. **Comparação**: Gerar relatório comparativo entre os quatro cenários
# MAGIC
# MAGIC ## Métricas RAGAS
# MAGIC - **Faithfulness (Fidelidade)**: O quanto a resposta é fiel ao contexto recuperado
# MAGIC - **Answer Relevancy (Relevância da Resposta)**: O quanto a resposta é relevante para a pergunta
# MAGIC - **Context Precision (Precisão do Contexto)**: Se os itens relevantes estão bem ranqueados na recuperação
# MAGIC - **Context Recall (Revocação do Contexto)**: Se toda a informação necessária foi recuperada

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install "ragas==0.1.16" langchain-databricks pandas matplotlib seaborn --quiet
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup and Configuration
# Configuration
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# -- Vector Search configuration (mesma do notebook 04) --
ENDPOINT_NAME = "querido_diario_endpoint"
INDEX_NAME = "workspace.tcc_rag.querido_diario_vector_index"
SOURCE_TABLE = "workspace.tcc_rag.gold_quality_variants_chunks"
PRIMARY_KEY = "chunk_id"
EMBEDDING_SOURCE_COLUMN = "chunk_text"
EMBEDDING_MODEL = "databricks-bge-large-en"

# -- LLM for answer generation --
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# -- Quality variants to test --
QUALITY_VARIANTS = ["baseline", "ocr_noise", "duplicate", "missing_fields"]

# -- RAG parameters --
NUM_RETRIEVAL_RESULTS = 5  # Top-K chunks to retrieve
MAX_TOKENS = 500
TEMPERATURE = 0.1  # Low temperature for factual answers

# Initialize Databricks client
w = WorkspaceClient()

print("✓ Configuration loaded")
print(f"  Index: {INDEX_NAME}")
print(f"  LLM: {LLM_ENDPOINT}")
print(f"  Variants: {QUALITY_VARIANTS}")
print(f"  Top-K retrieval: {NUM_RETRIEVAL_RESULTS}")

# COMMAND ----------

# DBTITLE 1,Define Test Questions and Ground Truth
# Define test questions with ground truth answers based on the baseline gazette data
# These questions have factual answers that can be verified from the Recife official gazette

TEST_QUESTIONS = [
    {
        "question": "Quantos empregos formais foram criados em Recife em fevereiro de 2025?",
        "ground_truth": "Foram criados 3.351 novos postos de trabalho (empregos formais) em Recife em fevereiro de 2025, segundo dados do Novo Caged do Ministério do Trabalho e Emprego."
    },
    {
        "question": "Qual foi o saldo positivo de empregos no setor de serviços?",
        "ground_truth": "O setor de serviços teve um saldo positivo de 2.991 empregos, resultado de 13.847 contratações."
    },
    {
        "question": "Qual o número total de empregos na cidade de Recife após fevereiro de 2025?",
        "ground_truth": "O estoque de empregos na cidade de Recife aumentou para 565.312 após as contratações de fevereiro de 2025."
    },
    {
        "question": "Qual o número de telefone do Conecta Zap para atualização do CadÚnico?",
        "ground_truth": "O número de telefone do Conecta Zap, conta oficial e verificada da Prefeitura do Recife no WhatsApp, é 81 99117-1407."
    },
    {
        "question": "Quantas admissões e desligamentos foram registrados no período?",
        "ground_truth": "Foram registradas 20.978 admissões e 17.627 desligamentos, resultando em 3.351 novos postos de trabalho."
    },
    {
        "question": "Quantas pessoas serão alcançadas na primeira fase da campanha do CadÚnico pelo WhatsApp?",
        "ground_truth": "Na primeira fase, as mensagens serão enviadas exclusivamente para beneficiários do Bolsa Família, alcançando cerca de 150 mil pessoas."
    }
]

print(f"✓ {len(TEST_QUESTIONS)} test questions defined")
for i, q in enumerate(TEST_QUESTIONS, 1):
    print(f"  Q{i}: {q['question']}")

# COMMAND ----------

# DBTITLE 1,Vector Search Retrieval Function
# Vector Search retrieval function
# Queries the Vector Search index filtered by quality_variant

def retrieve_chunks(query_text: str, quality_variant: str, num_results: int = NUM_RETRIEVAL_RESULTS) -> list:
    """
    Retrieve chunks from the Vector Search index filtered by quality_variant.
    
    Args:
        query_text: The search query
        quality_variant: One of 'baseline', 'ocr_noise', 'duplicate', 'missing_fields'
        num_results: Number of chunks to retrieve
    
    Returns:
        List of dictionaries with chunk_id, chunk_text, gazette_id, and similarity score
    """
    filters = json.dumps({"quality_variant": quality_variant})
    
    results = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        query_text=query_text,
        columns=["chunk_id", "chunk_text", "gazette_id", "quality_variant"],
        filters_json=filters,
        num_results=num_results
    )
    
    data_array = results.result.data_array if results.result else []
    
    chunks = []
    for row in data_array:
        chunks.append({
            "chunk_id": row[0],
            "chunk_text": row[1],
            "gazette_id": row[2],
            "quality_variant": row[3],
            "score": row[4] if len(row) > 4 else None
        })
    
    return chunks

# Quick test: retrieve from baseline
print("Testing retrieval from baseline variant...")
test_chunks = retrieve_chunks("empregos formais Recife", "baseline", num_results=2)
print(f"✓ Retrieved {len(test_chunks)} chunks")
for c in test_chunks:
    print(f"  chunk_id: {c['chunk_id'][:50]}... | score: {c['score']:.4f}")
    print(f"  preview: {c['chunk_text'][:100]}...")

# COMMAND ----------

# DBTITLE 1,LLM Answer Generation Function
# LLM answer generation function
# Generates an answer using retrieved context chunks

def generate_answer(question: str, chunks: list) -> str:
    """
    Generate an answer using retrieved context chunks.
    
    Args:
        question: The user question
        chunks: List of chunk dictionaries with 'chunk_text'
    
    Returns:
        The LLM-generated answer
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"Documento {i} (Gazette: {chunk['gazette_id']}, Variant: {chunk['quality_variant']}):\n{chunk['chunk_text']}")
    
    context = "\n\n".join(context_parts)
    
    # System prompt
    system_prompt = """Você é um assistente especializado em legislação municipal brasileira.
Responda a pergunta do usuário usando APENAS as informações fornecidas no contexto.
Se a informação não estiver no contexto, diga que não encontrou essa informação específica.
Cite os documentos relevantes na sua resposta."""
    
    # User prompt with context
    user_prompt = f"""Contexto (documentos municipais):

{context}

---

Pergunta: {question}

Resposta:"""
    
    # Call LLM
    response = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=user_prompt)
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    
    return response.choices[0].message.content

print("✓ Answer generation function defined")

# COMMAND ----------

# DBTITLE 1,Run RAG Pipeline for All Variants
# Run RAG pipeline for all question × variant combinations
# Collects results into a list for RAGAS evaluation

print("=" * 80)
print("Running RAG pipeline for all variants...")
print("=" * 80)

rag_results = []

total_combinations = len(TEST_QUESTIONS) * len(QUALITY_VARIANTS)
current = 0

for q_idx, test_q in enumerate(TEST_QUESTIONS, 1):
    question = test_q["question"]
    ground_truth = test_q["ground_truth"]
    
    print(f"\n{'─' * 80}")
    print(f"Q{q_idx}: {question}")
    print(f"{'─' * 80}")
    
    for variant in QUALITY_VARIANTS:
        current += 1
        print(f"\n  [{current}/{total_combinations}] Variant: {variant}...")
        
        # Step 1: Retrieve chunks
        try:
            chunks = retrieve_chunks(question, variant)
            print(f"    ✓ Retrieved {len(chunks)} chunks")
        except Exception as e:
            print(f"    ❌ Retrieval error: {e}")
            continue
        
        if not chunks:
            print(f"    ⚠️ No chunks retrieved, skipping")
            continue
        
        # Step 2: Generate answer
        try:
            answer = generate_answer(question, chunks)
            print(f"    ✓ Answer generated ({len(answer)} chars)")
        except Exception as e:
            print(f"    ❌ Generation error: {e}")
            continue
        
        # Step 3: Record result
        rag_results.append({
            "question": question,
            "question_id": q_idx,
            "quality_variant": variant,
            "answer": answer,
            "contexts": [c["chunk_text"] for c in chunks],
            "ground_truth": ground_truth,
            "retrieval_scores": [c["score"] for c in chunks if c["score"] is not None]
        })
        
        # Show answer preview
        print(f"    Preview: {answer[:120]}...")

print(f"\n{'=' * 80}")
print(f"✓ Pipeline complete: {len(rag_results)} results collected")
print(f"{'=' * 80}")

# COMMAND ----------

# DBTITLE 1,RAGAS Evaluation
# RAGAS Evaluation
# Evaluate RAG quality across all variants using RAGAS 0.1.16 API

import pandas as pd
import ragas
print(f"RAGAS version: {ragas.__version__}")

# Configure RAGAS to use Databricks LLM and embeddings
from langchain_databricks import ChatDatabricks, DatabricksEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from ragas.run_config import RunConfig

# Create LLM and embeddings for RAGAS evaluation
# Use a more capable model as the RAGAS judge (separate from answer generation LLM)
RAGAS_JUDGE_LLM = "databricks-llama-4-maverick"  # More capable than Llama 3.3 70B
print(f"Configuring RAGAS with judge LLM: {RAGAS_JUDGE_LLM}...")
eval_llm = ChatDatabricks(endpoint=RAGAS_JUDGE_LLM, temperature=0)
eval_embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_MODEL)

ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

# Prepare evaluation dataset
# RAGAS 0.1.x expects a HuggingFace Dataset with: question, answer, contexts, ground_truth
from datasets import Dataset
eval_data = Dataset.from_pandas(pd.DataFrame(rag_results)[["question", "answer", "contexts", "ground_truth"]])

print(f"\nRunning RAGAS evaluation...")
print(f"  Samples: {len(eval_data)}")
print(f"  Metrics: faithfulness, answer_relevancy, context_precision, context_recall")
print(f"  LLM Judge: {RAGAS_JUDGE_LLM}")
print(f"  Embeddings: {EMBEDDING_MODEL}")
print()

# Run evaluation with higher timeout to avoid LLM timeouts
# Default timeout is 180s; increased to 600s with more retries
run_config = RunConfig(timeout=600, max_retries=20, max_wait=120, max_workers=4)

result = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
)

# Convert results to DataFrame
ragas_df = result.to_pandas()

# Add metadata columns for grouping
ragas_df["quality_variant"] = [r["quality_variant"] for r in rag_results]
ragas_df["question_id"] = [r["question_id"] for r in rag_results]

print("✓ RAGAS evaluation complete!")
print(f"  Results: {len(ragas_df)} evaluations")
print(f"  Columns: {list(ragas_df.columns)}")
print()

# Display full results
display(ragas_df)

# COMMAND ----------

# DBTITLE 1,Comparative Report: Aggregate Metrics
# Comparative Report: Aggregate Metrics by Quality Variant

# Define metric columns (RAGAS output column names may vary)
metric_cols = [c for c in ragas_df.columns if c in [
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "faithfulness_score", "answer_relevancy_score", 
    "context_precision_score", "context_recall_score"
]]

# If metric columns have different names, find them
if len(metric_cols) < 4:
    # Try alternative naming patterns
    possible_metrics = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
    }
    metric_cols = []
    for metric_name, possible_names in possible_metrics.items():
        for col in ragas_df.columns:
            if any(name in col.lower() for name in possible_names):
                metric_cols.append(col)
                break

print("Metric columns detected:", metric_cols)
print()

# Aggregate by quality variant
agg_df = ragas_df.groupby("quality_variant")[metric_cols].agg(["mean", "std"]).round(4)

print("=" * 80)
print("📊 RELATÓRIO COMPARATIVO: Métricas RAGAS por Cenário de Qualidade")
print("=" * 80)
print()

# Display mean scores in a clean table
mean_df = ragas_df.groupby("quality_variant")[metric_cols].mean().round(4)
mean_df = mean_df.reindex(QUALITY_VARIANTS)  # Maintain consistent ordering

print("\nMédias por cenário:")
print("-" * 80)
display(mean_df)

print("\nDesvio padrão por cenário:")
print("-" * 80)
std_df = ragas_df.groupby("quality_variant")[metric_cols].std().round(4)
std_df = std_df.reindex(QUALITY_VARIANTS)
display(std_df)

# COMMAND ----------

# DBTITLE 1,Visualization: Comparative Charts
# Visualização: Gráficos comparativos

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Avaliação RAGAS por Cenário de Qualidade de Dados", fontsize=16, fontweight="bold")

colors = {"baseline": "#2ecc71", "ocr_noise": "#e74c3c", "duplicate": "#f39c12", "missing_fields": "#3498db"}

for idx, metric in enumerate(metric_cols):
    ax = axes[idx // 2][idx % 2]
    
    # Prepare data for this metric
    plot_data = ragas_df.groupby("quality_variant")[metric].mean().reindex(QUALITY_VARIANTS)
    
    # Bar chart
    bars = ax.bar(
        range(len(QUALITY_VARIANTS)),
        plot_data.values,
        color=[colors[v] for v in QUALITY_VARIANTS],
        edgecolor="black",
        linewidth=0.5
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, plot_data.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )
    
    # Add error bars (standard deviation)
    std_data = ragas_df.groupby("quality_variant")[metric].std().reindex(QUALITY_VARIANTS)
    ax.errorbar(
        range(len(QUALITY_VARIANTS)),
        plot_data.values,
        yerr=std_data.values,
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5
    )
    
    ax.set_xticks(range(len(QUALITY_VARIANTS)))
    ax.set_xticklabels(QUALITY_VARIANTS, rotation=15, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n✓ Gráficos gerados")

# COMMAND ----------

# DBTITLE 1,Heatmap: Metrics by Variant
# Heatmap: Métricas por cenário (visão consolidada)

fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = mean_df.T  # Transpose for better layout

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Score"}
)

ax.set_title("Heatmap: Métricas RAGAS por Cenário de Qualidade", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Métrica", fontsize=11)
ax.set_xlabel("Cenário de Qualidade", fontsize=11)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Per-Question Breakdown
# Detailed per-question breakdown

print("=" * 80)
print("DETALHAMENTO: Métricas por Pergunta e Cenário")
print("=" * 80)

for q_id in sorted(ragas_df["question_id"].unique()):
    question_row = ragas_df[ragas_df["question_id"] == q_id].iloc[0]
    print(f"\n{'─' * 80}")
    print(f"Q{q_id}: {question_row['question']}")
    print(f"{'─' * 80}")
    
    q_data = ragas_df[ragas_df["question_id"] == q_id].set_index("quality_variant")
    q_data = q_data.reindex(QUALITY_VARIANTS)
    
    display(q_data[metric_cols].round(4))

# COMMAND ----------

# DBTITLE 1,Save Results to Delta Table
# Save results to a Delta table for persistence and further analysis

results_to_save = ragas_df[[
    "question_id", "question", "quality_variant", "answer", "ground_truth"
] + metric_cols].copy()

# Add contexts as JSON string for storage
results_to_save["contexts"] = [json.dumps(r["contexts"]) for r in rag_results]

# Save to Unity Catalog
OUTPUT_TABLE = "workspace.tcc_rag.rag_quality_evaluation_results"

# Check if table exists; create or append accordingly
try:
    spark.sql(f"DESCRIBE TABLE {OUTPUT_TABLE}")
    table_exists = True
    print(f"Table {OUTPUT_TABLE} already exists. Appending new results.")
    print(f"  To start fresh, drop the table first: DROP TABLE {OUTPUT_TABLE}")
except Exception:
    table_exists = False
    print(f"Table {OUTPUT_TABLE} does not exist. Creating new table.")

# Write with append mode (creates table if it doesn't exist)
spark.createDataFrame(results_to_save).write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE)

print(f"\n✓ Results saved to {OUTPUT_TABLE}")
print(f"  Rows written: {len(results_to_save)}")
print(f"  Columns: {list(results_to_save.columns)}")

# COMMAND ----------

# DBTITLE 1,Conclusions and Next Steps
# MAGIC %md
# MAGIC ## Conclusões e Próximos Passos
# MAGIC
# MAGIC ### Análise dos Resultados
# MAGIC
# MAGIC O relatório comparativo acima evidencia o impacto da qualidade dos dados sobre o desempenho do pipeline RAG:
# MAGIC
# MAGIC 1. **Baseline**: Serve como referência (controle) — dados originais sem degradação
# MAGIC 2. **OCR Noise**: Avalia o impacto de erros de OCR (caracteres trocados, palavras corrompidas)
# MAGIC 3. **Duplicate**: Avalia o impacto de conteúdo duplicado nos chunks recuperados
# MAGIC 4. **Missing Fields**: Avalia o impacto de campos removidos no contexto enviado ao LLM
# MAGIC
# MAGIC ### Métricas Interpretadas
# MAGIC
# MAGIC | Métrica | O que mede |
# MAGIC | --- | --- |
# MAGIC | **Faithfulness** | Fidelidade da resposta ao contexto recuperado (0-1) |
# MAGIC | **Answer Relevancy** | Relevância da resposta em relação à pergunta (0-1) |
# MAGIC | **Context Precision** | Precisão do ranking de recuperação (0-1) |
# MAGIC | **Context Recall** | Cobertura da informação necessária no contexto (0-1) |
# MAGIC
# MAGIC ### Próximos Passos
# MAGIC - Ampliar o conjunto de perguntas-teste para maior robustez estatística
# MAGIC - Avaliar diferentes modelos LLM (ex: Llama 3.1, Claude) como geradores e juízes
# MAGIC - Experimentar diferentes valores de Top-K e estratégias de chunking
# MAGIC - Considerar métricas adicionais do RAGAS (ex: context_entity_recall, noise_sensitivity)
# MAGIC - Analisar casos individuais onde cada cenário degrada significativamente a resposta

# COMMAND ----------

