# Databricks notebook source
# MAGIC %md
# MAGIC # Weather RAG -- Embedding Ingestion
# MAGIC
# MAGIC Reads `weather_documents` rows that don't have an embedding yet
# MAGIC (`embedded_at IS NULL`), chunks their `narrative_text`, embeds every
# MAGIC chunk with `sentence-transformers/all-MiniLM-L6-v2`, and writes the
# MAGIC vectors into `weather_embeddings` -- ready for `<=>` cosine search the
# MAGIC instant this notebook finishes, no manual follow-up step required.
# MAGIC
# MAGIC Idempotent and safe to re-run: `lakebase.ensure_weather_schema()` is
# MAGIC applied first, already-embedded documents are excluded by the
# MAGIC `WHERE embedded_at IS NULL` filter, and the embeddings insert is an
# MAGIC `ON CONFLICT ... DO UPDATE`, so re-running after a document's text
# MAGIC changed replaces its chunks instead of erroring or duplicating.
# MAGIC
# MAGIC If you swap `EMBEDDING_MODEL_NAME`, update `EMBEDDING_DIM` to match and
# MAGIC re-apply `sql/02_weather_embeddings.sql`'s `VECTOR({{EMBEDDING_DIM}})`
# MAGIC column, or every insert below will fail with a dimension mismatch:
# MAGIC
# MAGIC | Model | Dim |
# MAGIC | --- | --- |
# MAGIC | `sentence-transformers/all-MiniLM-L6-v2` (default here) | 384 |
# MAGIC | `sentence-transformers/all-mpnet-base-v2` | 768 |
# MAGIC | `BAAI/bge-small-en-v1.5` | 384 |
# MAGIC | `BAAI/bge-base-en-v1.5` | 768 |
# MAGIC
# MAGIC This file is plain Databricks notebook source -- it also runs standalone
# MAGIC as `python notebooks/ingest_weather_embeddings.py`. Every widget below
# MAGIC has a working `os.environ` default, and all `dbutils` use is guarded so
# MAGIC it never executes outside an actual notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: install dependencies
# MAGIC
# MAGIC A notebook's attached cluster does **not** automatically have this repo's
# MAGIC `requirements.txt` installed -- that only happens automatically for the
# MAGIC already-deployed Databricks App's own container, which is a completely
# MAGIC separate environment from a notebook's cluster. Without this cell, the
# MAGIC very next `import lakebase` below fails with `ModuleNotFoundError:
# MAGIC No module named 'sqlalchemy'` (or `psycopg2`, or `sentence_transformers`,
# MAGIC depending on what the cluster's base image happens to already have).
# MAGIC
# MAGIC The relative path below (`../requirements.txt`) resolves against this
# MAGIC notebook's own location inside the Git folder / Repo -- that resolution
# MAGIC is a Databricks Repos feature and needs this notebook to actually be
# MAGIC inside one (which it is, per this project's own deploy instructions). If
# MAGIC this cell errors that it cannot find the file, install the same packages
# MAGIC directly instead:
# MAGIC ```
# MAGIC %pip install sqlalchemy psycopg2-binary sentence-transformers numpy databricks-sdk torch --extra-index-url https://download.pytorch.org/whl/cpu
# MAGIC ```
# MAGIC Skipped automatically when this file runs as a plain script
# MAGIC (`python notebooks/ingest_weather_embeddings.py`), since `%pip` is a
# MAGIC notebook-only magic command -- that path is expected to already be
# MAGIC running inside a venv where `pip install -r requirements.txt` was run
# MAGIC ahead of time (see this repo's README).

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# Restart so the packages just installed are actually importable -- %pip
# install alone does not refresh an already-running Python process's loaded
# modules. Guarded the same way every other dbutils use in this file is:
# running as a plain script has no dbutils at all, and needs no restart since
# its venv already has everything installed.
if "dbutils" in globals():
    dbutils.library.restartPython()  # noqa: F821 -- only defined inside an actual notebook run

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Same env var names and defaults as `embedder.py` uses for
# MAGIC `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM`, so this notebook and the
# MAGIC running Flask app can never silently disagree about which model or
# MAGIC dimension is in play.

# COMMAND ----------

import os
import sys
import time
from pathlib import Path

_NOTEBOOK_START = time.time()

# `python notebooks/ingest_weather_embeddings.py` puts the *notebooks/* dir on
# sys.path, not the repo root -- so `import embedder` / `import lakebase`
# below would fail standalone without this. `__file__` isn't defined when
# Databricks runs this as actual notebook cells, where the repo root is
# already on sys.path anyway, hence the guard.
if "__file__" in globals():
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)


def _param(name: str, default: str) -> str:
    """Resolve a config value from a notebook widget if `dbutils` is actually
    defined (i.e. we're really running inside Databricks), else fall back to
    `os.environ` -- so `python notebooks/ingest_weather_embeddings.py` works
    standalone with no `dbutils` at all, using the exact same default.
    """
    if "dbutils" in globals():
        try:
            dbutils.widgets.text(name, os.environ.get(name, default))
            return dbutils.widgets.get(name)
        except Exception:
            pass
    return os.environ.get(name, default)


EMBEDDING_MODEL_NAME = _param("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(_param("EMBEDDING_DIM", "384"))

# Not env/widget-configurable on purpose: embedder.py's own internal batch
# size (_BATCH_SIZE) isn't read from the environment either, so exposing this
# as an override here would let it print a value that embed_texts() doesn't
# actually use -- the exact "silently disagree" trap this config cell exists
# to avoid.
BATCH_SIZE = 32

# embedder.py resolves EMBEDDING_MODEL_NAME/EMBEDDING_DIM from os.environ at
# *import* time. Setting them here before importing it (rather than after)
# means a widget override actually changes which model gets used below, not
# just what this cell prints.
os.environ["EMBEDDING_MODEL_NAME"] = EMBEDDING_MODEL_NAME
os.environ["EMBEDDING_DIM"] = str(EMBEDDING_DIM)

import embedder
import lakebase
from psycopg2.extras import execute_values

print(f"EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME}")
print(f"EMBEDDING_DIM        = {EMBEDDING_DIM}")
print(f"BATCH_SIZE           = {BATCH_SIZE}")

if embedder.MODEL_NAME != EMBEDDING_MODEL_NAME or embedder.EMBEDDING_DIM != EMBEDDING_DIM:
    # Only possible if embedder was already imported earlier in this same
    # interpreter session with a different value -- module-level constants
    # don't re-evaluate on a second import, so surface it instead of quietly
    # embedding with the wrong model.
    print(
        "WARNING: embedder.py already had a different model/dim resolved "
        f"({embedder.MODEL_NAME}/{embedder.EMBEDDING_DIM}) before this cell "
        "ran. Restart the Python/notebook kernel to pick up the new value."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: make sure the schema exists
# MAGIC
# MAGIC `ensure_weather_schema()` applies `sql/01_weather_documents.sql` and
# MAGIC `sql/02_weather_embeddings.sql`. Both are `IF NOT EXISTS` / `ON CONFLICT`
# MAGIC guarded, so calling this here is a no-op if someone already ran the
# MAGIC `.sql` files by hand -- it just makes this notebook safe to run first.

# COMMAND ----------

schema_result = lakebase.ensure_weather_schema(embedding_dim=EMBEDDING_DIM)
print(f"ensure_weather_schema -> {schema_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: find documents that still need embedding
# MAGIC
# MAGIC `embedded_at IS NULL` is the "needs (re-)embedding" marker on
# MAGIC `weather_documents`. An empty result here is normal (everything is
# MAGIC already embedded) -- the remaining cells cascade harmlessly over an
# MAGIC empty list rather than needing a special early exit.

# COMMAND ----------

unembedded_rows = lakebase.run_query(
    "SELECT id, narrative_text FROM weather_documents WHERE embedded_at IS NULL"
)
print(f"Documents needing embedding: {len(unembedded_rows)}")

if not unembedded_rows:
    print("Nothing to embed -- every weather_documents row already has embedded_at set.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: chunk each document's narrative text
# MAGIC
# MAGIC `embedder.chunk_text()` slides a `CHUNK_SIZE`/`CHUNK_OVERLAP` character
# MAGIC window over `narrative_text` and returns `[]` for empty/whitespace text.
# MAGIC Alerts (~484 measured chars) and forecast periods (~111 measured chars)
# MAGIC are short enough to almost always come back as a single chunk; an area
# MAGIC forecast discussion (~9275 measured chars) is exactly the case chunking
# MAGIC exists for -- it should split into several. Documents that produce zero
# MAGIC chunks are skipped for embedding but still get `embedded_at` stamped in
# MAGIC Step 6, so an empty document isn't retried forever.

# COMMAND ----------

processed_document_ids = []
zero_chunk_document_ids = []
embedding_rows = []  # (document_id, chunk_index, chunk_text)
multi_chunk_document_count = 0

for row in unembedded_rows:
    document_id = row["id"]
    chunks = embedder.chunk_text(row["narrative_text"])

    if not chunks:
        zero_chunk_document_ids.append(document_id)
        continue

    if len(chunks) > 1:
        multi_chunk_document_count += 1

    processed_document_ids.append(document_id)
    for chunk_index, chunk in enumerate(chunks):
        embedding_rows.append((document_id, chunk_index, chunk))

chunk_texts = [chunk for _, _, chunk in embedding_rows]

print(f"Total chunks produced: {len(chunk_texts)}")
print(
    f"Documents producing more than one chunk: {multi_chunk_document_count} "
    f"(chunk_size={embedder.CHUNK_SIZE}, chunk_overlap={embedder.CHUNK_OVERLAP} -- "
    "expect this near zero for short alerts/forecast periods and meaningfully "
    "nonzero once discussion-length narratives are in the batch)"
)
print(f"Documents skipped (empty/whitespace narrative_text, zero chunks): {len(zero_chunk_document_ids)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: embed every chunk
# MAGIC
# MAGIC One call to `embedder.embed_texts()` -- it already batches internally,
# MAGIC so there's no manual batching loop here.

# COMMAND ----------

embed_start = time.time()
vectors = embedder.embed_texts(chunk_texts)
embed_elapsed = time.time() - embed_start

print(f"Embedded {len(chunk_texts)} chunk(s) in {embed_elapsed:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: write vectors into weather_embeddings
# MAGIC
# MAGIC The insert casts straight to `::vector` in the template
# MAGIC (`"(%s, %s, %s, %s, %s::vector, %s)"` via `execute_values`), with
# MAGIC `embedder.to_vector_literal()` supplying that value as a bracketed
# MAGIC pgvector literal string. That is deliberate: a related pipeline for the
# MAGIC analogous ticker-news source writes embeddings as a plain
# MAGIC `%s::double precision[]` array and then tells the user to separately run
# MAGIC a manual `UPDATE ... SET embedding = embedding::vector` afterward -- and
# MAGIC if that manual step is ever skipped, every later similarity search
# MAGIC silently returns nothing, with no error anywhere pointing at why. Casting
# MAGIC in the same statement means a row is queryable via `<=>` the instant this
# MAGIC transaction commits, with no follow-up step, ever.
# MAGIC
# MAGIC `ON CONFLICT (document_id, chunk_index) DO UPDATE` makes re-running this
# MAGIC notebook after a document's text changed replace its old chunks instead
# MAGIC of erroring or duplicating them.

# COMMAND ----------

_UPSERT_EMBEDDINGS_SQL = """
INSERT INTO weather_embeddings (id, document_id, chunk_index, chunk_text, embedding, model_name)
VALUES %s
ON CONFLICT (document_id, chunk_index) DO UPDATE SET
    chunk_text = EXCLUDED.chunk_text,
    embedding = EXCLUDED.embedding,
    model_name = EXCLUDED.model_name,
    created_at = now()
"""

insert_rows = [
    (
        f"{document_id}:{chunk_index}",
        document_id,
        chunk_index,
        chunk_text,
        embedder.to_vector_literal(vector),
        embedder.MODEL_NAME,
    )
    for (document_id, chunk_index, chunk_text), vector in zip(embedding_rows, vectors)
]

if insert_rows:
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                _UPSERT_EMBEDDINGS_SQL,
                insert_rows,
                template="(%s, %s, %s, %s, %s::vector, %s)",
            )
        conn.commit()
    print(f"Wrote {len(insert_rows)} embedding row(s) to weather_embeddings")
else:
    print("No embedding rows to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: stamp embedded_at
# MAGIC
# MAGIC Marks every document just processed -- including the zero-chunk ones,
# MAGIC which have nothing to embed but still shouldn't be re-queried by Step 2
# MAGIC on every future run -- so this notebook only ever does new work.

# COMMAND ----------

stamp_ids = processed_document_ids + zero_chunk_document_ids

if stamp_ids:
    stamped_count = lakebase.run_write(
        "UPDATE weather_documents SET embedded_at = now() WHERE id = ANY(%s)",
        (stamp_ids,),
    )
else:
    stamped_count = 0

print(
    f"Stamped embedded_at on {stamped_count} document(s) "
    f"({len(zero_chunk_document_ids)} had empty/whitespace narrative_text and "
    "zero chunks, but are stamped too so they aren't retried forever)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

total_elapsed = time.time() - _NOTEBOOK_START

print("=" * 60)
print("Weather embedding ingestion summary")
print(f"  Documents processed : {len(unembedded_rows)}")
print(f"  Chunks embedded     : {len(chunk_texts)}")
print(f"  Elapsed time        : {total_elapsed:.2f}s")
print("=" * 60)
