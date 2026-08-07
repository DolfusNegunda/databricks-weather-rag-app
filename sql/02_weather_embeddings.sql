-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the
-- embedding notebook. Idempotent -- safe to run more than once.
-- Replace {{EMBEDDING_DIM}} with your model's output dimension if you swap
-- models (see notebooks/ingest_weather_embeddings.py's model/dim table):
--   - sentence-transformers/all-MiniLM-L6-v2 (default here): 384
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768

-- Enable pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. document_id -> weather_documents.id, cascading on
-- delete so removing a document (e.g. an expired alert cleanup) removes its
-- vectors in the same statement rather than leaving orphans.
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id          TEXT PRIMARY KEY,         -- "{document_id}:{chunk_index}"
    document_id TEXT NOT NULL REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT weather_embeddings_document_chunk_unique
        UNIQUE (document_id, chunk_index)
);

-- HNSW for fast approximate cosine search. Requires at least one row to
-- build against, so this may need to be (re)run after the first embedding
-- batch lands if your Postgres version builds the index eagerly and empty --
-- IF NOT EXISTS makes re-running it a no-op either way.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document
    ON weather_embeddings (document_id);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
