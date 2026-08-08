# Weather RAG — homework writeup

## Data source and why

This project builds its RAG pipeline over the **National Weather Service
API** (`api.weather.gov`) instead of the reference app's stock-ticker-news
source. It needs **no API key** — every endpoint used here
(`/points`, `/alerts/active`, `/gridpoints/.../forecast`,
`/products/types/AFD/locations/...`, `/products/{id}`) is open and keyless,
which is exactly why the bootcamp brief recommends it as a source to build
against: nothing to provision, nothing to rotate, nothing that can 403 for
lack of a credential (only for a missing `User-Agent`, which
`weather_client.py` always sends).

The pipeline deliberately harvests **three** different NWS source types, not
just one, and the numbers measured while planning this project are the
reason:

- **Alerts** (`source_type = 'alert'`): at the time of checking, there were
  **214 active alerts nationwide** — but **zero** active alerts for the
  entire state of Texas. Alerts are real-time and important when present,
  but a pipeline built on alerts alone would have had **nothing to retrieve
  at all** for a large fraction of the country on a calm day. That is not a
  hypothetical edge case; it is the literal measured state of one whole
  state during planning.
- **Forecasts** (`source_type = 'forecast'`): always available for every
  point NWS covers, regardless of weather. A single fetch returns 14
  periods, averaging roughly 111 characters of narrative each
  (`detailedForecast`) — short, structured, and reliable filler for the
  calm-day case alerts can't cover.
- **Area forecast discussions** (`source_type = 'discussion'`): a real one
  measured during planning ran **9,275 characters** of free-text
  meteorologist narrative in a single `productText`. Alerts and forecast
  periods are short enough that chunking barely does anything to them
  (most produce exactly one chunk at the pipeline's 800-character chunk
  size); a discussion is the case chunking exists for, routinely splitting
  into several hundred-plus-character chunks with meaningful overlap
  between them.

Together, the three source types mean retrieval always has *something*
useful to return (forecasts cover the calm days alerts can't), the pipeline
has at least one source substantial enough to make chunking do real work
(discussions), and the `source_type` filter on `POST /weather/search` /
`GET /weather/search` (a stretch goal) has a genuine reason to exist —
alerts, forecasts, and discussions are different enough in tone, length, and
urgency that filtering by one is a real, useful distinction, not a
decorative parameter with nothing to differentiate.

## Schema decisions

Two tables, one-to-many:

- **`weather_documents`** — one row per harvested item (one alert, one
  forecast fetch, one discussion). Holds the raw NWS payload (`payload`
  `JSONB`, kept for provenance) alongside the flattened fields the UI and
  search results actually read (`location`, `source_type`, `event`,
  `headline`, `severity`, timestamps) and the free text that gets embedded
  (`narrative_text`).
- **`weather_embeddings`** — one row per *chunk* of a document's
  `narrative_text`. `document_id REFERENCES weather_documents(id) ON DELETE
  CASCADE`, so deleting a document (say, pruning an expired alert) removes
  all of its vectors in the same statement — no orphaned embedding rows to
  clean up separately.

Both tables live in their own Postgres schema (`LAKEBASE_SCHEMA`, default
`weather`), not `public`. This matters in practice: this is the second
bootcamp homework in this workspace, and reusing the same Lakebase instance
(and the same connection-string secret) from Day 1 is the natural thing to
do rather than provisioning a second instance just to run a second app. A
shared `public` schema would have worked today — nothing here happens to
share a table name with Day 1's `watchlist_items`/`price_snapshots` — but
that's luck, not a guarantee, and a shared namespace also makes it harder to
tear one project's data out without touching the other's. `lakebase.py`
creates the schema automatically and points every connection at it via `SET
search_path` (not the libpq `options` parameter, which Lakebase's proxy
silently drops), which is also why nothing in `app.py`, the ingestion
notebook, or any of the three verification scripts needed to change to get
this isolation — they all reference `weather_documents`/`weather_embeddings`
unqualified already.

### `content_hash` + `embedded_at`

Beyond what the reference ticker-news pipeline needed, `weather_documents`
carries `content_hash` (a `sha1` of `narrative_text`) and `embedded_at`
(`NULL` until a document's chunks have been embedded). `POST /weather/sync`'s
upsert compares the *incoming* hash against the *previously stored* hash
(via a `WITH previous AS (...)` CTE, since a plain `RETURNING` on an
`INSERT ... ON CONFLICT` can't see the pre-write value) and only clears
`embedded_at` back to `NULL` when the hash actually changed:

```sql
embedded_at = CASE
    WHEN weather_documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
    THEN NULL
    ELSE weather_documents.embedded_at
END
```

This distinction matters most concretely for **alerts**: an alert's `id` is
the NWS URN taken verbatim, so it stays the *same* row across every re-sync
while the alert is active — and the scheduled resync job (see below) hits
`POST /weather/sync` with `source_types=["alert"]` every 15 minutes. Most of
those re-syncs will find identical text (NWS hasn't updated the alert since
last check), and the hash comparison means that case leaves `embedded_at`
alone, so `notebooks/ingest_weather_embeddings.py`'s `WHERE embedded_at IS
NULL` query correctly skips it — no needless re-embedding of unchanged text
every 15 minutes, forever, for as long as an alert stays active. If NWS
*does* revise that same alert's wording, the hash changes, `embedded_at` is
cleared automatically, and the next notebook run re-embeds it with no human
having to notice or decide anything.

Forecasts and discussions get this same mechanism, but it shows up
differently in practice: `normalize_forecast` derives its document `id` from
a hash of `(location, updated)` and `normalize_discussion` prefixes its `id`
with the NWS product's own UUID — so a *newly issued* forecast run or AFD
almost always arrives as a brand-new row (new `id`, `embedded_at` already
`NULL`) rather than an in-place update to an existing one. The practical
effect is the same — new content ends up queued for embedding automatically,
no manual bookkeeping required — but it is the fresh-row path doing that
work for forecasts/discussions, while it is genuinely the hash comparison
doing it for alerts, which are the one source type whose `id` is stable
across repeated syncs of the same real-world item.

### Chunking and embedding

`embedder.py`'s actual defaults (confirmed by reading the file, not assumed):

- **Chunk size 800 characters, overlap 100 characters** — a sliding
  character window (`WEATHER_CHUNK_SIZE` / `WEATHER_CHUNK_OVERLAP`), matching
  the reference pipeline's convention.
- **Model `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional
  output** (`EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM`), matching
  `sql/02_weather_embeddings.sql`'s `VECTOR(384)` column.

Matching the reference pipeline's model choice specifically — not just
picking any 384-dim (or any) sentence-transformer model — matters because it
keeps this pipeline's vectors in the *same embedding space* as the rest of
the bootcamp material built on that model. Two different models can both
output 384 floats and still place semantically similar text at very
different points in their respective spaces; staying on the same model
means the cosine-distance (`<=>`) conventions this whole bootcamp uses stay
comparable across projects, rather than this pipeline quietly becoming a
second, incompatible embedding space that happens to fit in the same column
type.

## Running the pipeline end to end

1. **Clean checkout and environment:**

   ```bash
   git clone <this-repo-url> && cd lakebase-weather-rag
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   cp env.example .env
   ```

   Edit `.env`, set `LAKEBASE_URL` to a real Lakebase Postgres connection
   string, then start the app — `app.py` calls `load_dotenv()` before
   importing anything that reads the environment, so `.env` is picked up
   automatically:

   ```bash
   python app.py
   ```

2. **Harvest data** — sync a couple of locations across all three source
   types:

   ```bash
   curl -X POST http://localhost:8000/weather/sync \
     -H "Content-Type: application/json" \
     -d '{
           "locations": ["Chicago, IL", "Austin, TX"],
           "source_types": ["alert", "forecast", "discussion"]
         }'
   ```

   This upserts into `weather_documents`. Nothing is embedded yet —
   `embedded_at` stays `NULL` on every new row.

3. **Embed** — run the ingestion notebook. Two equivalent forms:

   - **Standalone**, from the repo root, against the same `.env`-derived
     environment as step 1:

     ```bash
     python notebooks/ingest_weather_embeddings.py
     ```

   - **As an actual Databricks notebook**: open
     `notebooks/ingest_weather_embeddings.py` in the workspace Git folder
     and run all cells. Same code, same output, `dbutils` widgets standing
     in for the `os.environ` defaults used standalone.

   Either way it prints the number of documents needing embedding, chunks
   produced, and rows written to `weather_embeddings`.

4. **Search** — a query interesting enough to actually exercise the
   `alert` source type if any are active:

   ```bash
   curl -X POST http://localhost:8000/weather/search \
     -H "Content-Type: application/json" \
     -d '{"query": "flash flood risk this weekend", "top_k": 3}'
   ```

   *Illustrative example response* (hand-built from `app.py`'s actual
   `_run_search` field names — **not** an actual captured response; no live
   run against a populated database has happened in this build):

   ```json
   {
     "query": "flash flood risk this weekend",
     "top_k": 3,
     "source_type": null,
     "results": [
       {
         "document_id": "urn:oid:2.49.0.1.840.0.abcdef123456",
         "location": "Austin, TX",
         "source_type": "alert",
         "event": "Flash Flood Watch",
         "headline": "Flash Flood Watch issued for central Texas",
         "severity": "Moderate",
         "issued_at": "2026-08-05T18:10:00+00:00",
         "chunk_index": 0,
         "chunk_text": "Heavy rainfall totals of 2 to 4 inches, with isolated higher amounts, are possible through the weekend, leading to a risk of flash flooding in low-lying and poor-drainage areas.",
         "similarity": 0.8123
       },
       {
         "document_id": "discussion:9f3a1c2b-...-...",
         "location": "Austin, TX",
         "source_type": "discussion",
         "event": null,
         "headline": "Area Forecast Discussion",
         "severity": null,
         "issued_at": "2026-08-05T17:32:00+00:00",
         "chunk_index": 4,
         "chunk_text": "...an unsettled pattern persists into the weekend, with PWATs approaching 1.8 inches supporting locally heavy rainfall and an isolated flash flood threat, especially south of I-10...",
         "similarity": 0.7654
       },
       {
         "document_id": "b7e6c1e4f9a2...",
         "location": "Austin, TX",
         "source_type": "forecast",
         "event": null,
         "headline": "Saturday",
         "severity": null,
         "issued_at": "2026-08-05T09:05:00+00:00",
         "chunk_index": 0,
         "chunk_text": "Saturday: Showers and thunderstorms likely, with a high near 88. Chance of precipitation is 70%.",
         "similarity": 0.6890
       }
     ],
     "count": 3
   }
   ```

5. **Search with a summary** — the `GET` variant, with `summarize=true` to
   also ask a Foundation Model endpoint to answer using only the retrieved
   chunks:

   ```bash
   curl "http://localhost:8000/weather/search?query=flash+flood+risk+this+weekend&summarize=true"
   ```

   Same result shape as above, plus (illustrative) `"summary": "There is an
   active Flash Flood Watch for central Texas through the weekend, with 2-4
   inches of rain possible and localized flash flooding in low-lying
   areas...", "summary_error": null`. If the Foundation Model endpoint named
   by `FOUNDATION_MODEL_ENDPOINT` isn't reachable or provisioned, this
   degrades to `"summary": null` with a `summary_error` string instead of
   failing the whole request.

6. **Stretch goal — scheduled resync job.** Deploy the Databricks Asset
   Bundle that re-syncs `alert` data every 15 minutes against the
   already-deployed app:

   ```bash
   databricks bundle deploy -t dev
   databricks bundle run weather-alert-resync -t dev
   ```

   `dev` mode deploys the schedule paused by design (a DAB behavior); use
   `databricks bundle deploy -t prod` for it to actually fire on its own.
   `WEATHER_APP_URL` has no default and must be set (see
   `docs/scheduled_job.md`) before either target is useful.

## Known limitations and what I would improve given more time

- **Query-time embedding depends on `torch` + `sentence-transformers`
  actually installing inside the Databricks App container.** That's a real
  deployment risk: `torch`'s CPU wheel is still ~200MB and the model weights
  another ~90MB, against a typical Databricks App's modest memory budget.
  `embedder.py` was designed with an escape hatch in mind — swapping
  `get_model()`/`embed_texts()`/`embed_query()` for an ONNX Runtime session
  over the same `all-MiniLM-L6-v2` weights (same 384-dim output, ~50MB, no
  `torch` dependency) — but that fallback has **not actually been built or
  tested**, only planned for. If the app container can't carry `torch` in
  practice, this is the first thing to build.
- **The location resolver is a small, fixed lookup table** — ten named
  `"City, ST"` strings in `weather_client.LOCATIONS`, plus a raw `"lat,lon"`
  pair typed directly by the caller. It is not a geocoding service. Any
  other city name, however commonly recognized, fails with a `ValueError`
  naming the accepted forms rather than being resolved. This is a real
  coverage limit, stated plainly rather than implied to be broader.
  The UI's Sync panel used to expose this as a free-text field, comma-parsed
  client-side into `"City, ST"` pairs — real production bug found from a
  screenshot: an odd number of comma tokens left the last one dangling as
  its own bogus location (a lone `"IL"` sent as if it were a whole city,
  rejected with a confusing per-location error). Replaced with a
  click-to-select grid built from `GET /api/locations`, so a click can only
  ever produce an exact string the server already knows how to resolve —
  the parsing step, and the bug class that came with it, no longer exists.
  `scripts/check_api.py` asserts the removed parser function stays removed.
- **No live run against a real Lakebase instance has happened yet.** All
  three verification scripts exist and the two offline ones are green
  (`scripts/check_api.py`: 85 checks; `scripts/check_sql.py`: 14 checks,
  covering both `.sql` files, the exact SQL constants read out of `app.py`
  and the ingestion notebook by AST rather than retyped, and the
  `execute_values` template after simulating its row substitution) — but
  every one of those checks ran against mocks and fixtures, not a live
  database. `scripts/check_connection.py` is written and its logic has been
  read and reasoned through carefully, but it has not actually been *run*
  against a real instance in this build session. Standing up a real Lakebase
  instance and running `python scripts/check_connection.py --write` is the
  single highest-value next step, and the first thing to do before trusting
  any of this end to end.
- **Other gaps noticed while writing this, not previously flagged:**
  - `lakebase.LakebaseUnavailable` is a `RuntimeError` subclass, but
    `_sync_location`'s point-resolution call and `_sync_forecast` /
    `_sync_discussion` in `app.py` only catch `requests.RequestException`
    around their NWS calls. An unconfigured/unreachable Lakebase raised from
    inside `_upsert_weather_document` (or from `resolve_point`'s indirect
    dependence on a working sync) surfaces as a bare `500 internal_error`
    for `forecast`/`discussion` sync, instead of the clean per-location
    `error` field alert-sync gets when there happen to be zero features to
    upsert. This is exactly the failure mode a fresh, not-yet-configured
    deployment would hit first.
  - Nothing filters, in search or anywhere else, on `expires_at`. An alert
    that expired days ago remains fully retrievable and fully rankable by
    `/weather/search` forever — there is no cleanup job and no `WHERE`
    clause excluding expired rows. Given more time this would be either a
    scheduled prune (delete expired alerts, letting `ON DELETE CASCADE`
    take their embeddings with them) or a `WHERE expires_at IS NULL OR
    expires_at > now()` filter at query time.
  - `normalize_forecast`'s document `id` is a hash of `(location, updated)`
    and `normalize_discussion`'s `id` embeds the NWS product's own UUID —
    both change on every genuinely new forecast run or newly issued
    discussion, so re-syncing produces a growing set of historical rows in
    `weather_documents`/`weather_embeddings` rather than updating one row in
    place. Nothing currently prunes the older ones, so a location synced
    repeatedly over weeks would accumulate many near-duplicate forecast/
    discussion documents, all still searchable.
  - `POST /weather/sync`'s `limit` parameter only caps the number of alert
    *features* processed per location (`features[:limit]`) — it has no
    effect on forecast or discussion, each of which is always exactly one
    document (or zero) per location regardless of `limit`. That's not
    documented anywhere in the route itself.
  - `normalize_alert` sets `latitude`, `longitude`, `office`, `grid_x`, and
    `grid_y` all to `None`, even though `_sync_location` already has all
    five values in hand before calling `_sync_alerts`. Alert rows are the
    one source type in `weather_documents` missing geographic metadata that
    forecast/discussion rows keep — worth passing through if a future
    feature wants to query or map `weather_documents` by geography across
    all three source types uniformly.
  - `sql/README.md` still describes the schema bootstrap as
    "`app.py`'s `ensure_weather_tables()`" — the current, actual function is
    `lakebase.py`'s `ensure_weather_schema()`. Stale from an earlier
    iteration of the design; worth fixing whenever that file is next
    touched.
  - A dimension inconsistency in `scripts/check_connection.py`: its schema-
    bootstrap step called `ensure_weather_schema(embedding_dim=384)` with a
    hardcoded literal rather than `embedder.EMBEDDING_DIM`, while the column-
    width check later in the same script correctly used the latter. An
    `EMBEDDING_DIM` override via the environment would have bootstrapped one
    width while every other consumer expected another. Fixed to read
    `embedder.EMBEDDING_DIM` in both places.
