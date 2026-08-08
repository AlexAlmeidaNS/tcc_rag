# Databricks notebook source
# MAGIC %md
# MAGIC # Ingestão de teste — API Querido Diário -> Delta Lake (Databricks)
# MAGIC
# MAGIC Objetivo: puxar diários oficiais via API pública do Querido Diário,
# MAGIC persistir em Delta (camada Bronze) e registrar métricas básicas de
# MAGIC qualidade de dados úteis para o TCC (completude, duplicidade, atraso de publicação).
# MAGIC
# MAGIC API docs: https://api.queridodiario.ok.org.br/docs

# COMMAND ----------

# MAGIC %md ## 1. Configuração

# COMMAND ----------

import requests
import time
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType, ArrayType
)

BASE_URL = "https://api.queridodiario.ok.org.br"
ENDPOINT = "/gazettes"

# Exemplo: alguns municípios (código IBGE) para o teste piloto.
# Vale escolher municípios de portes diferentes para já observar
# variação de qualidade/cobertura no seu TCC.
TERRITORY_IDS = [
    "2700706",  # Batalha - AL
    "3550308",  # São Paulo - SP
    "2304400",  # Fortaleza - CE
]

PUBLISHED_SINCE = "2024-01-01"
PUBLISHED_UNTIL = "2024-12-31"
PAGE_SIZE = 50

CATALOG = "workspace"           # ajuste para seu catálogo Unity Catalog
SCHEMA = "tcc_rag"         # ajuste conforme seu schema
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_querido_diario"

# COMMAND ----------

# MAGIC %md ## 2. Função de extração (paginada) por município

# COMMAND ----------

def fetch_gazettes(territory_id, published_since, published_until, page_size=50, max_pages=None):
    """
    Consulta a API do Querido Diário para um município e intervalo de datas,
    paginando os resultados. Retorna lista de dicts (registros brutos da API).
    """
    results = []
    offset = 0
    page = 0

    while True:
        params = {
            "territory_ids": territory_id,
            "published_since": published_since,
            "published_until": published_until,
            "size": page_size,
            "offset": offset,
        }

        resp = requests.get(f"{BASE_URL}{ENDPOINT}", params=params, timeout=30)

        if resp.status_code != 200:
            # Registrar falha explicitamente em vez de silenciar —
            # relevante para sua discussão de qualidade/observabilidade
            print(f"[ERRO] territory_id={territory_id} offset={offset} status={resp.status_code}")
            break

        data = resp.json()
        gazettes = data.get("gazettes", [])

        if not gazettes:
            break

        for g in gazettes:
            g["_ingestion_territory_id"] = territory_id
            g["_ingestion_timestamp"] = datetime.utcnow().isoformat()
            results.append(g)

        offset += page_size
        page += 1

        if max_pages and page >= max_pages:
            break

        # Cortesia com a API pública — evita rate limit
        time.sleep(0.3)

    return results

# COMMAND ----------

# MAGIC %md ## 3. Executar extração para os municípios de teste

# COMMAND ----------

all_records = []

for tid in TERRITORY_IDS:
    print(f"Extraindo diários de {tid}...")
    records = fetch_gazettes(
        territory_id=tid,
        published_since=PUBLISHED_SINCE,
        published_until=PUBLISHED_UNTIL,
        page_size=PAGE_SIZE,
        max_pages=5,  # limite para teste piloto; remover em execução completa
    )
    print(f"  -> {len(records)} registros")
    all_records.extend(records)

print(f"\nTotal geral: {len(all_records)} registros")

# COMMAND ----------

# MAGIC %md ## 4. Carregar em DataFrame Spark e persistir camada Bronze (Delta)

# COMMAND ----------

if all_records:
    # schema explícito necessário porque excerpts pode ser array vazia
    schema = StructType([
        StructField("territory_id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("scraped_at", StringType(), True),
        StructField("url", StringType(), True),
        StructField("territory_name", StringType(), True),
        StructField("state_code", StringType(), True),
        StructField("excerpts", ArrayType(StringType()), True),
        StructField("edition", StringType(), True),
        StructField("is_extra_edition", StringType(), True),
        StructField("txt_url", StringType(), True),
        StructField("_ingestion_territory_id", StringType(), True),
        StructField("_ingestion_timestamp", StringType(), True),
    ])
    
    sdf = spark.createDataFrame(all_records, schema=schema)

    sdf = sdf.withColumn("_ingestion_date", F.current_date())

    (
        sdf.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(BRONZE_TABLE)
    )

    print(f"Gravado em {BRONZE_TABLE}: {sdf.count()} linhas")
else:
    print("Nenhum registro retornado — verifique parâmetros/territory_ids/janela de datas.")

# COMMAND ----------

# MAGIC %md ## 5. Checagens rápidas de qualidade (para embasar o TCC)

# COMMAND ----------

df_bronze = spark.table(BRONZE_TABLE)

print("Total de registros:", df_bronze.count())

# Completude por campo-chave
for col in ["territory_name", "date", "file_url", "excerpts"]:
    if col in df_bronze.columns:
        nulls = df_bronze.filter(F.col(col).isNull()).count()
        print(f"  Nulos em '{col}': {nulls}")

# Duplicidade (mesmo diário reprocessado)
if "file_url" in df_bronze.columns:
    dup = (
        df_bronze.groupBy("file_url")
        .count()
        .filter("count > 1")
        .count()
    )
    print("URLs de arquivo duplicadas:", dup)

# Distribuição por município (cobertura desigual)
if "_ingestion_territory_id" in df_bronze.columns:
    df_bronze.groupBy("_ingestion_territory_id").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximos passos sugeridos
# MAGIC 1. Camada **Silver**: parsing/normalização do texto extraído, deduplicação, tipagem de datas.
# MAGIC 2. Camada **Gold**: chunking do texto + geração de embeddings (ex.: Databricks Vector Search).
# MAGIC 3. Instrumentar testes de qualidade (Delta Live Tables expectations ou Great Expectations)
# MAGIC    diretamente entre Bronze -> Silver, para depois correlacionar falhas de qualidade
# MAGIC    com métricas de RAG (faithfulness, context precision) na etapa experimental do TCC.