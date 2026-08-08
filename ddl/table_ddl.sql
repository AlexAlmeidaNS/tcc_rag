-- ============================================================
-- Querido Diário - Table DDL Statements
-- Generated: 2026-08-05 00:46:04.401204
-- ============================================================

-- ============================================================
-- Bronze Layer: Raw API Data
-- ============================================================

CREATE TABLE workspace.tcc_rag.bronze_querido_diario (
  territory_id STRING COLLATE UTF8_BINARY,
  date STRING COLLATE UTF8_BINARY,
  scraped_at STRING COLLATE UTF8_BINARY,
  url STRING COLLATE UTF8_BINARY,
  territory_name STRING COLLATE UTF8_BINARY,
  state_code STRING COLLATE UTF8_BINARY,
  excerpts ARRAY<STRING COLLATE UTF8_BINARY>,
  edition STRING COLLATE UTF8_BINARY,
  is_extra_edition STRING COLLATE UTF8_BINARY,
  txt_url STRING COLLATE UTF8_BINARY,
  _ingestion_territory_id STRING COLLATE UTF8_BINARY,
  _ingestion_timestamp STRING COLLATE UTF8_BINARY,
  _ingestion_date DATE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd',
  'delta.parquet.format.version' = '2.12.0',
  'delta.parquet.format.version.afe.internal' = '2.12.0')
;


-- ============================================================
-- Silver Layer: Cleaned and Enriched with Text Content
-- ============================================================

CREATE TABLE workspace.tcc_rag.silver_querido_diario (
  gazette_id STRING COLLATE UTF8_BINARY NOT NULL COMMENT 'Surrogate key: hash of territory_id + date + edition',
  territory_id STRING COLLATE UTF8_BINARY NOT NULL,
  territory_name STRING COLLATE UTF8_BINARY,
  state_code STRING COLLATE UTF8_BINARY,
  publication_date DATE NOT NULL,
  edition STRING COLLATE UTF8_BINARY,
  is_extra_edition BOOLEAN,
  scraped_at TIMESTAMP,
  pdf_url STRING COLLATE UTF8_BINARY COMMENT 'Original PDF URL',
  txt_url STRING COLLATE UTF8_BINARY COMMENT 'Extracted text URL',
  text_content STRING COLLATE UTF8_BINARY COMMENT 'Full extracted text from txt_url',
  text_length INT COMMENT 'Character count of text_content',
  text_preview STRING COLLATE UTF8_BINARY COMMENT 'First 500 characters',
  word_count INT,
  content_extraction_status STRING COLLATE UTF8_BINARY COMMENT 'success, failed, pending',
  content_error_message STRING COLLATE UTF8_BINARY,
  has_content BOOLEAN,
  is_valid_url BOOLEAN,
  content_extracted_at TIMESTAMP,
  search_excerpts ARRAY<STRING COLLATE UTF8_BINARY> COMMENT 'Pre-indexed excerpts from API',
  _bronze_ingestion_timestamp TIMESTAMP,
  _silver_processed_at TIMESTAMP NOT NULL)
USING delta
COMMENT 'Silver layer: cleaned and enriched official gazettes with extracted text content'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.enableDeletionVectors' = 'true',
  'delta.enableRowTracking' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.changeDataFeed' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.domainMetadata' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.feature.rowTracking' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd',
  'delta.parquet.format.version' = '2.12.0',
  'delta.parquet.format.version.afe.internal' = '2.12.0')
;
