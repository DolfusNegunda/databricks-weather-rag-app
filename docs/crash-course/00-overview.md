# Weather RAG Crash Course: Overview

This is the map. Everything else in `docs/crash-course/` is a runnable notebook
that zooms into one box below. Read this first, then work through
`01_embeddings.py` through `05_hnsw_benchmark.py` in order — each one assumes
you understand the stage before it.

The one-sentence version of the whole project: **turn free-text weather
reports into searchable meaning, so a question in plain English can find the
right sentence out of thousands, and (optionally) a language model can turn
that sentence into a written answer.**

## The pipeline, end to end

```
                          api.weather.gov
                                 |
                                 |  GET /alerts/active, /gridpoints/.../forecast,
                                 |  /products/{id}  (weather_client.py)
                                 v
                    +-------------------------+
                    |   raw JSON per source   |
                    |  alert / forecast /     |
                    |  discussion             |
                    +-------------------------+
                                 |
                                 |  normalize_alert() / normalize_forecast() /
                                 |  normalize_discussion()  (weather_client.py)
                                 v
                    +-------------------------+
                    |    weather_documents    |   <- ONE row per alert /
                    |  narrative_text (free   |      forecast fetch / discussion,
                    |  text), location,       |      raw payload kept for
                    |  severity, payload...   |      provenance
                    +-------------------------+
                                 |
                                 |  chunk_text()  (embedder.py)
                                 |  slide an 800-char / 100-char-overlap
                                 |  window over narrative_text
                                 v
                    +-------------------------+
                    |     text chunks         |   <- a short alert stays
                    |  (1 or more per doc)    |      ONE chunk; a 9275-char
                    +-------------------------+      discussion becomes ~14
                                 |
                                 |  embed_texts()  (embedder.py)
                                 |  sentence-transformers/all-MiniLM-L6-v2
                                 v
                    +-------------------------+
                    |   384-float vectors      |  <- one per chunk, same
                    |   (one per chunk)         |     space every time
                    +-------------------------+
                                 |
                                 |  INSERT ... ::vector  (ingestion notebook)
                                 v
                    +-------------------------+
                    |   weather_embeddings     |  <- chunk_text + embedding
                    |   VECTOR(384) column,    |     VECTOR(384), HNSW index
                    |   HNSW index             |     on embedding
                    +-------------------------+
                                 |
        user types a question   |
        "flash flood risk       |
        this weekend"           |
                |                |
                v                |
      embed_query()  (embedder.py, SAME model)
                |                |
                +------ both are now points in the same 384-dim space ------+
                                 |
                                 v
                    ORDER BY embedding <=> query_vector   (app.py, pgvector)
                                 |
                                 v
                    +-------------------------+
                    |   top-k nearest chunks   |  <- ranked by cosine
                    |   + similarity scores    |     similarity
                    +-------------------------+
                                 |
                                 |  (optional) build a prompt from ONLY
                                 |  the retrieved chunks  (app.py:
                                 |  _summarize_results)
                                 v
                    +-------------------------+
                    |  Foundation Model call   |  <- "answer using only
                    |  -> grounded answer      |     the documents below"
                    +-------------------------+
```

Every arrow above is a real function call in this repo, not a metaphor —
`docs/crash-course/01` through `05` run each stage in isolation so you can
see it happen.

## Why each stage exists, and what breaks if you skip it

**Harvest (`weather_client.py`'s `get_alerts` / `get_forecast` /
`get_latest_discussion`).** The NWS API is a *live* API, not a document
store — asking it the same question twice means two more network round
trips. This stage exists to turn "an API you'd have to re-query for every
user question" into a corpus you can search instantly and repeatedly. Skip
it, and every search becomes three live calls to api.weather.gov per
question (alerts, gridpoint forecast, AFD lookup), each with its own
15-second timeout and its own chance of a 403 if the User-Agent header is
missing — for a question someone might ask fifty times a day.

**Normalize into `weather_documents` (`normalize_alert` /
`normalize_forecast` / `normalize_discussion`).** The three NWS source types
have three unrelated JSON shapes: an alert's usable text is spread across
`properties.description` and `properties.instruction`; a forecast's text is
buried inside `properties.periods[i].detailedForecast` for 14 separate
periods; a discussion's text is one long `productText` string. This stage
funnels all three into the same shape — one `narrative_text` column — so
every later stage (chunking, embedding, search) has exactly one code path
instead of three. Skip it, and `chunk_text`/`embed_texts` would need
per-source-type branches, and the actual `/weather/search` query in
`app.py` — which searches alerts, forecasts, and discussions together with a
single `ORDER BY ... <=> ...` — would instead need three separate queries
unioned back together by hand.

**Chunking (`embedder.chunk_text`, called from
`notebooks/ingest_weather_embeddings.py`).** This is the stage with the
biggest real cost to skipping, and the project has the measurements to prove
it. A real Area Forecast Discussion we fetched during planning ran **9,275
characters** — a single narrative covering a synoptic overview, a
weekend flash-flood-risk paragraph, and an overnight-low-temperature
paragraph, all in one string. `all-MiniLM-L6-v2` compresses *however much
text you hand it* into exactly one fixed 384-number vector. Embed that whole
discussion as a single chunk and you get one vector that is a blended
average of every sub-topic in it — a query about "flash flood risk" now has
to compete against unrelated overnight-temperature content diluting that one
vector's similarity score. This project's default (`CHUNK_SIZE=800`,
`CHUNK_OVERLAP=100`, both from `embedder.py`) turns that one discussion into
roughly a dozen focused chunks instead, each embeddable and rankable on its
own topic. Contrast that with a single NWS alert, whose combined
description+instruction text measured only **620 characters** (484 + 136)
during planning — well under the 800-char chunk size, so it almost always
stays exactly one chunk. Chunking barely does anything to alerts or forecast
periods; it is entirely earning its keep on discussions.

**Embedding (`embedder.embed_texts` / `get_model`).** This is what makes
search *semantic* instead of *lexical*. The model was trained so that two
sentences with the same meaning land near each other in 384-dimensional
space even if they share almost no words. Skip it (fall back to keyword or
`ILIKE` text search), and a query like "flash flood risk this weekend" would
only find documents that literally contain those words — a differently
worded discussion paragraph that says "rapid water-level rises are possible
given saturated soils" would be invisible to keyword search while being
exactly the right answer semantically. `01_embeddings.py` proves this
concretely with two very differently worded flood-warning sentences.

**Store vectors in `weather_embeddings` (`sql/02_weather_embeddings.sql`,
the `VECTOR(384)` column + HNSW index).** pgvector adds a native vector type
and distance operators directly inside the same Postgres database that
already holds `weather_documents` — so there is no second system (a
dedicated vector database) to keep in sync, and a similarity search can be a
plain SQL `JOIN` back to `location` / `severity` / `headline` instead of a
separate lookup. Skip it, and you'd need to load every embedding into
Python memory on every app start and brute-force the cosine math yourself —
which is exactly what `05_hnsw_benchmark.py` measures the cost of at scale.

**Embed the query the same way (`embedder.embed_query`, called from
`app.py`'s `_run_search`).** Cosine similarity is only meaningful when both
vectors came from the same model — comparing a query embedded with a
different model, or the same model with different settings, to a corpus
embedded earlier would land it in a different 384-dim space where distances
are geometrically meaningless. Skip this discipline and the failure is
silent: the query still returns *some* rows, they're just not actually the
nearest ones, with no error anywhere to say so.

**Nearest-neighbor search by cosine distance (`app.py`'s
`_WEATHER_SEARCH_SQL`, the `<=>` operator).** This is the payoff: instead of
a live, multi-second round trip to three NWS endpoints, a search against an
already-embedded corpus with an HNSW index returns in milliseconds, ranked
by actual semantic closeness. `03_pgvector.py` and `05_hnsw_benchmark.py`
dig into what the index buys you over a full sequential scan.

**RAG: hand the retrieved text to a language model (`app.py`'s
`_summarize_results`).** The model is instructed to answer "using only the
weather documents provided below... do not invent facts that are not
present in them." That grounding is the entire point — a general-purpose
model has no idea what alert is active in Austin *this morning*, so asking
it directly would either get a refusal or a fluent, confident, wrong guess.
Retrieval-then-summarize gives the model real, just-synced text to work
from. Skip retrieval and ask the model directly, and you've turned a
grounded answer into a hallucination risk. `04_retrieval_rag.py` also shows
the failure mode in the *other* direction: retrieval can hand the model
correct-but-incomplete text if a chunk boundary splits a hazard from its
instruction.

## Map: pipeline stage -> file in this repo

| Stage | Implemented in |
| --- | --- |
| Harvest raw NWS JSON | `weather_client.py` (`get_alerts`, `get_forecast`, `get_latest_discussion`, `resolve_point`) |
| Normalize into a common document shape | `weather_client.py` (`normalize_alert`, `normalize_forecast`, `normalize_discussion`) |
| Raw document storage | `sql/01_weather_documents.sql` (`weather_documents` table); written by `app.py`'s `_upsert_weather_document` |
| Chunk long narrative text | `embedder.py` (`chunk_text`); driven by `notebooks/ingest_weather_embeddings.py` (Step 3) |
| Embed chunks into 384-dim vectors | `embedder.py` (`embed_texts`, `get_model`); driven by `notebooks/ingest_weather_embeddings.py` (Step 4) |
| Store vectors in a pgvector column | `sql/02_weather_embeddings.sql` (`weather_embeddings` table, `VECTOR(384)`, HNSW index); written by `notebooks/ingest_weather_embeddings.py` (Step 5) |
| Embed the user's query the same way | `embedder.py` (`embed_query`); called from `app.py` (`_run_search`) |
| Nearest-neighbor search by cosine distance | `app.py` (`_WEATHER_SEARCH_SQL`, the `<=>` operator), routes `POST/GET /weather/search` |
| RAG: grounded answer from retrieved text | `app.py` (`_summarize_results`), triggered by `"summarize": true` |
| Connection pooling / auth shared by every stage above | `lakebase.py` (`get_connection`, `run_query`, `run_write`, `ensure_weather_schema`) |

## How to use the rest of this crash course

- `01_embeddings.py` — what a 384-float vector actually is, and why two
  differently worded sentences about the same flood warning end up close
  together while an unrelated sentence doesn't. Also proves, in plain
  Python, that `1 - cosine_distance` is the same number as `cosine_similarity`
  — the fact the SQL in later files leans on.
- `02_chunking.py` — runs the project's real `chunk_text` on a real (or
  realistic fallback) discussion, shows the literal overlapping substring
  between two chunks, and constructs a deliberate example of a sentence
  getting cut in half when `chunk_overlap` is 0.
- `03_pgvector.py` — the `VECTOR` column type, the three distance operators
  (`<->`, `<#>`, `<=>`), and HNSW vs IVFFlat vs a sequential scan.
- `04_retrieval_rag.py` — runs (or, without a live database, prints) the
  actual retrieval query from `app.py`, then walks through building the
  same RAG prompt `_summarize_results` builds, plus a worked example of bad
  chunking causing a retrieved answer to miss the actual instruction.
- `05_hnsw_benchmark.py` — a real latency (and, where practical, recall)
  benchmark of the HNSW index against a forced sequential scan.

Every `.py` file in this folder is Databricks notebook source (`# COMMAND
----------` cell markers) and also runs standalone as a plain script. None
of them require a live Lakebase connection or an installed
`sentence-transformers`/`torch` to produce useful output — where either is
missing, they fall back to clearly labeled hand-computed or hand-written
examples instead of crashing.
