# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
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

# DBTITLE 1,Setup: Configuration
import requests
import time
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType, ArrayType
)
import random  # adicionar ao topo, junto dos outros imports

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MAX_PER_TERRITORY = 120

BASE_URL = "https://api.queridodiario.ok.org.br"
ENDPOINT = "/gazettes"

# Exemplo: alguns municípios (código IBGE) para o teste piloto.
# Vale escolher municípios de portes diferentes para já observar
# variação de qualidade/cobertura no seu TCC.
# TERRITORY_IDS = [
#     "3550308",  # São Paulo - SP
#     "2304400",  # Fortaleza - CE
#     "2611606",  # Recife - PE
#     "2927408",  # Salvador - BA
#     "3106200",  # Belo Horizonte - MG
#     "4106902",  # Curitiba - PR
#     "4314902",  # Porto Alegre - RS
#     "1302603",  # Manaus - AM
#     "5208707",  # Goiânia - GO
#     "2700706",  # Batalha - AL
# ]

TERRITORY_IDS = [
    "2611606",  # Recife - PE
]

PUBLISHED_SINCE = "2025-01-01"
PUBLISHED_UNTIL = "2025-12-31"
PAGE_SIZE = 50

CATALOG = "workspace"           # ajuste para seu catálogo Unity Catalog
SCHEMA = "tcc_rag"         # ajuste conforme seu schema
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_querido_diario"

# Configurações de download de texto bruto
DOWNLOAD_TIMEOUT = 30  # segundos por requisição
MAX_RETRIES = 3       # tentativas por URL
RETRY_BACKOFF = 1     # segundos (multiplicado pelo número da tentativa)



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

# DBTITLE 1,Cell 7
all_records = []

for tid in TERRITORY_IDS:
    print(f"Extraindo diários de {tid}...")
    try:
        records = fetch_gazettes(
            territory_id=tid,
            published_since=PUBLISHED_SINCE,
            published_until=PUBLISHED_UNTIL,
            page_size=PAGE_SIZE
        )
    except Exception as e:
        print(f"  [AVISO] Falha ao conectar à API ({type(e).__name__}): {e}")
        print(f"  Os metadados existentes na tabela Bronze serão preservados.")
        break
    
    # NOVO: aplica o teto por município
    if len(records) > MAX_PER_TERRITORY:
        records = random.sample(records, MAX_PER_TERRITORY)

    print(f"  -> {len(records)} registros (após teto de {MAX_PER_TERRITORY})")
    all_records.extend(records)

print(f"\nTotal geral: {len(all_records)} registros")

# COMMAND ----------

# MAGIC %md ## 4. Carregar em DataFrame Spark e persistir camada Bronze (Delta)

# COMMAND ----------

# DBTITLE 1,Persistir camada Bronze (Delta)
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
        # Novas colunas para texto bruto baixado da fonte
        StructField("txt_raw", StringType(), True),
        StructField("status_download", StringType(), True),
    ])
    
    sdf = spark.createDataFrame(all_records, schema=schema)

    sdf = sdf.withColumn("_ingestion_date", F.current_date())
    sdf = sdf.withColumn("status_download", F.lit("pending"))

    from delta.tables import DeltaTable

    if spark.catalog.tableExists(BRONZE_TABLE):
        # Garantir que as novas colunas existam na tabela existente
        existing_cols = spark.table(BRONZE_TABLE).columns
        if "txt_raw" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (txt_raw STRING)")
        if "status_download" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (status_download STRING)")
        if "_ingestion_date" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (_ingestion_date DATE)")

        # MERGE: preserva registros existentes (com texto já baixado),
        # insere apenas novos registros — garante idempotência
        delta_table = DeltaTable.forName(spark, BRONZE_TABLE)
        (
            delta_table.alias("target")
            .merge(
                sdf.alias("source"),
                "target.territory_id = source.territory_id "
                "AND target.date = source.date "
                "AND target.edition = source.edition"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
        print(f"MERGE em {BRONZE_TABLE}: novos registros inseridos (textos já baixados são preservados)")
    else:
        (
            sdf.write
            .format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .saveAsTable(BRONZE_TABLE)
        )
        print(f"Tabela {BRONZE_TABLE} criada com {sdf.count()} linhas")

    total = spark.table(BRONZE_TABLE).count()
    print(f"Total de registros na tabela: {total}")
else:
    # Garantir que as colunas de texto existam mesmo sem novos registros
    if spark.catalog.tableExists(BRONZE_TABLE):
        existing_cols = spark.table(BRONZE_TABLE).columns
        if "txt_raw" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (txt_raw STRING)")
        if "status_download" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (status_download STRING)")
        if "_ingestion_date" not in existing_cols:
            spark.sql(f"ALTER TABLE {BRONZE_TABLE} ADD COLUMNS (_ingestion_date DATE)")
        print("Nenhum registro novo — tabela Bronze preservada (colunas de texto adicionadas se necessário).")
    else:
        print("Nenhum registro retornado — verifique parâmetros/territory_ids/janela de datas.")

# COMMAND ----------

# DBTITLE 1,Download incremental de texto bruto
# MAGIC %md
# MAGIC ## 4.1. Download incremental do texto bruto
# MAGIC
# MAGIC Para cada registro ingerido, baixa o conteúdo textual a partir do campo `txt_url`.
# MAGIC O texto é armazenado **sem nenhum tratamento** na coluna `texto_bruto`.
# MAGIC O status do download é rastreado em `status_download` (`pending` → `success` | `error`).
# MAGIC Registros já baixados com sucesso são **ignorados** em execuções subsequentes (idempotente).

# COMMAND ----------

# DBTITLE 1,Função: Baixar texto bruto
import requests
import time

def download_text_content(url, timeout=DOWNLOAD_TIMEOUT, max_retries=MAX_RETRIES):
    """
    Baixa o conteúdo textual de um diário a partir da URL informada.
    Retorna o texto bruto exatamente como retornado pela fonte, sem qualquer tratamento.

    Retorna:
        tuple: (txt_raw, status_download, erro_msg)
            - txt_raw: texto bruto ou None
            - status_download: 'success' ou 'error'
            - erro_msg: mensagem de erro ou None
    """
    erro_msg = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text, "success", None
            else:
                erro_msg = f"HTTP {resp.status_code}"
        except Exception as e:
            erro_msg = str(e)

        if attempt < max_retries - 1:
            time.sleep(RETRY_BACKOFF * (attempt + 1))

    return None, "error", erro_msg

# COMMAND ----------

# DBTITLE 1,Executar download incremental
from delta.tables import DeltaTable

# Ler registros que ainda não tiveram o texto baixado com sucesso
bronze_df = spark.table(BRONZE_TABLE)

pending_df = (
    bronze_df
    .filter(
        (F.col("status_download").isNull() | (F.col("status_download") != "success"))
        & F.col("txt_url").isNotNull()
    )
)

pending_count = pending_df.count()
print(f"Registros pendentes de download: {pending_count}")

if pending_count > 0:
    # Coletar registros pendentes para o driver
    pending_records = (
        pending_df
        .select("territory_id", "date", "edition", "txt_url")
        .collect()
    )

    download_results = []
    success_count = 0
    error_count = 0

    for row in pending_records:
        txt_raw, status, erro = download_text_content(row["txt_url"])

        download_results.append({
            "territory_id": row["territory_id"],
            "date": row["date"],
            "edition": row["edition"],
            "txt_raw": txt_raw,
            "status_download": status,
        })

        if status == "success":
            success_count += 1
        else:
            error_count += 1
            print(f"  [ERRO] territory_id={row['territory_id']} date={row['date']} edition={row['edition']}: {erro}")

    print(f"\nDownloads concluídos: {success_count} sucesso, {error_count} erros")

    # Atualizar a tabela Bronze com os textos baixados
    if download_results:
        results_schema = StructType([
            StructField("territory_id", StringType(), True),
            StructField("date", StringType(), True),
            StructField("edition", StringType(), True),
            StructField("txt_raw", StringType(), True),
            StructField("status_download", StringType(), True),
        ])

        results_df = spark.createDataFrame(download_results, schema=results_schema)

        delta_table = DeltaTable.forName(spark, BRONZE_TABLE)
        (
            delta_table.alias("target")
            .merge(
                results_df.alias("source"),
                "target.territory_id = source.territory_id "
                "AND target.date = source.date "
                "AND target.edition = source.edition"
            )
            .whenMatchedUpdate(set={
                "txt_raw": "source.txt_raw",
                "status_download": "source.status_download",
            })
            .execute()
        )

        print(f"Tabela {BRONZE_TABLE} atualizada com {len(download_results)} registros de texto")
else:
    print("Nenhum registro pendente de download — todos já foram processados.")

# COMMAND ----------

# MAGIC %md ## 5. Checagens rápidas de qualidade (para embasar o TCC)

# COMMAND ----------

# DBTITLE 1,Checagens rápidas de qualidade
df_bronze = spark.table(BRONZE_TABLE)

print("Total de registros:", df_bronze.count())

# Completude por campo-chave
for col in ["territory_name", "date", "txt_url", "excerpts"]:
    if col in df_bronze.columns:
        nulls = df_bronze.filter(F.col(col).isNull()).count()
        print(f"  Nulos em '{col}': {nulls}")

# Duplicidade (mesmo diário reprocessado)
if "txt_url" in df_bronze.columns:
    dup = (
        df_bronze.groupBy("txt_url")
        .count()
        .filter("count > 1")
        .count()
    )
    print("URLs de arquivo duplicadas:", dup)

# Distribuição por município (cobertura desigual)
if "_ingestion_territory_id" in df_bronze.columns:
    df_bronze.groupBy("_ingestion_territory_id").count().show()

# Status de download de texto bruto
if "status_download" in df_bronze.columns:
    print("\nStatus de download de texto:")
    df_bronze.groupBy("status_download").count().orderBy(F.desc("count")).show()

    with_text = df_bronze.filter(F.col("txt_raw").isNotNull()).count()
    total = df_bronze.count()
    pct = (100 * with_text / total) if total > 0 else 0
    print(f"Registros com texto bruto: {with_text}/{total} ({pct:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximos passos sugeridos
# MAGIC 1. Camada **Silver**: parsing/normalização do texto extraído, deduplicação, tipagem de datas.
# MAGIC 2. Camada **Gold**: chunking do texto + geração de embeddings (ex.: Databricks Vector Search).
# MAGIC 3. Instrumentar testes de qualidade (Delta Live Tables expectations ou Great Expectations)
# MAGIC    diretamente entre Bronze -> Silver, para depois correlacionar falhas de qualidade
# MAGIC    com métricas de RAG (faithfulness, context precision) na etapa experimental do TCC.