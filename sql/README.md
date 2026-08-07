# SQL setup files for Lakebase

Run these manually against your Lakebase Postgres database before running the
app or `notebooks/ingest_weather_embeddings.py`.

## Setup order

### 1. Run `01_weather_documents.sql`

Creates `weather_documents`, the raw document store harvested from the
National Weather Service API. `app.py`'s `ensure_weather_tables()` also
creates this table on first use, so running it by hand is optional but makes
the schema visible up front.

### 2. Run `02_weather_embeddings.sql`

Creates `weather_embeddings` with a pgvector column.

**IMPORTANT:** replace `{{EMBEDDING_DIM}}` with your model's dimension:

* `sentence-transformers/all-MiniLM-L6-v2` (the default used throughout this
  project): **384**
* `sentence-transformers/all-mpnet-base-v2`: 768
* `BAAI/bge-small-en-v1.5`: 384
* `BAAI/bge-base-en-v1.5`: 768

If you change `EMBEDDING_MODEL_NAME` in the ingestion notebook, update this
file's `VECTOR({{EMBEDDING_DIM}})` to match, or every insert will fail with a
dimension mismatch.
