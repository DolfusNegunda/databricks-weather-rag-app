# Lakebase Weather RAG

A Retrieval-Augmented Generation pipeline over the National Weather Service
API (`api.weather.gov`): harvest active alerts, gridpoint forecasts, and area
forecast discussions for a handful of US locations, chunk and embed their
free text with `sentence-transformers/all-MiniLM-L6-v2`, store the vectors in
a `pgvector` column on Lakebase (Databricks-managed Postgres), and retrieve
the most relevant chunks for a natural-language query by cosine similarity
(`<=>`), optionally summarized by a Databricks Foundation Model endpoint. It
mirrors the structure of the bootcamp's reference stock-ticker-news app
(`github.com/EcZachly/databricks-lakebase-app-day-2`) with a different,
keyless data source. See `README_WEATHER.md` for the homework-specific
writeup (data source justification, schema decisions, limitations).

## Files

| Path | What it is |
| --- | --- |
| `app.py` | Flask entrypoint. Boots the schema, serves the UI, and exposes `/healthz`, `/weather/sync`, `/weather/search`. |
| `lakebase.py` | The only module that opens a database connection — pooled `psycopg2` connections, a SQLAlchemy engine, `run_query`/`run_write`, and `ensure_weather_schema()`. Every connection is pointed at `LAKEBASE_SCHEMA` (default `weather`, created automatically) via `SET search_path`, so this project's tables can share a Lakebase instance with another app with zero collision risk. |
| `weather_client.py` | NWS API client: fetches alerts/forecasts/discussions and normalizes each into a `weather_documents` row. |
| `embedder.py` | Chunking (800 chars / 100 overlap) and embedding (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim), plus the pgvector literal formatter. |
| `sql/` | `01_weather_documents.sql`, `02_weather_embeddings.sql` — idempotent DDL, applied automatically by `ensure_weather_schema()` and also runnable by hand. |
| `notebooks/ingest_weather_embeddings.py` | Embeds every un-embedded `weather_documents` row into `weather_embeddings`. Runs standalone (`python notebooks/...py`) or as a Databricks notebook. |
| `notebooks/weather_sync_job.py` | Calls the deployed app's `POST /weather/sync` for the scheduled resync job below. |
| `databricks.yml`, `resources/weather_sync_job.yml` | Databricks Asset Bundle for the stretch-goal scheduled job (`weather-alert-resync`, every 15 minutes) that re-syncs alerts against the already-deployed app. |
| `setup_secrets.py` | Run once, as a Databricks notebook, to store your Lakebase connection string in a secret scope before deploying. |
| `templates/`, `static/` | The "Weather Intelligence" single-page UI — a click-to-select city grid (backed by `GET /api/locations`, no free-text location parsing), search form, results pane. |
| `docs/scheduled_job.md` | How the scheduled resync job works and how to deploy/run it. |
| `docs/crash-course/` | A runnable, six-part crash course on the concepts this pipeline uses — embeddings, chunking, pgvector, retrieval/RAG, and an HNSW benchmark. See `docs/crash-course/00-overview.md`. |
| `scripts/` | Offline (`check_api.py`, `check_sql.py`) and live (`check_connection.py`) verification — see below. |
| `app.yaml` | Databricks App runtime config: start command and environment variables. |
| `env.example` | Template for local `.env`. Copy, don't edit in place. |
| `requirements.txt` | Python dependencies, including a CPU-only `torch` wheel index. |

## Local setup

```bash
python -m venv .venv
```

Activate it, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the environment template and fill in your Lakebase connection string:

```bash
cp env.example .env
```

Edit `.env` and set `LAKEBASE_URL` (or the secret-scope / OAuth alternatives
documented inline in `env.example` and in `lakebase.py`'s module docstring).
`app.py` calls `load_dotenv()` before importing anything that reads the
environment, so a `.env` file next to `app.py` is picked up automatically —
just run:

```bash
python app.py
```

Once the app is running, `http://localhost:8000/healthz` should
report `"status": "ok"` if `LAKEBASE_URL` is reachable, or `"degraded"` with
a `bootstrap_error` explaining why if not — the app is designed to come up
either way rather than crash at startup.

### Sharing a Lakebase instance with another app

This project's tables live in their own schema (`LAKEBASE_SCHEMA`, default
`weather`), not `public` — created automatically on first boot via `SET
search_path` on every connection (see `lakebase.py`). That means the *same*
Lakebase instance, and the *same* `database`/`lakebase-url` secret if you
already have one from another project, can be reused here with zero
table-name collision risk. Point `setup_secrets.py` at the connection string
you already have; storing an identical value is a no-op. Torn out later with
`DROP SCHEMA weather CASCADE`, leaving anything else on that instance intact.

## Verification scripts

```bash
python scripts/check_api.py          # offline: weather_client + embedder + app.py routes
python scripts/check_sql.py          # offline: every SQL statement, via pglast
python scripts/check_connection.py   # live, read-only: real Lakebase required
python scripts/check_connection.py --write   # + a self-cleaning write/read round trip
```

The two offline scripts need no database, no network, and no
`sentence-transformers`/`torch` install — everything that would need those is
monkeypatched with small fakes. They must both pass before any commit.

`check_connection.py` is the one that can see what the other two structurally
cannot: it connects to a real instance and proves the *arithmetic* — a known
vector compared against itself returns cosine similarity 1.0, and Postgres's
`1 - (a <=> b)` matches an independently-computed Python cosine to within
floating-point precision. A pglast grammar check would happily accept SQL
that computes the wrong number; this is what actually catches that.

## Databricks deployment

1. Push this repository to a Git provider your Databricks workspace can
   reach (GitHub, etc.).
2. In the workspace: **Workspace → Create → Git folder**, and point it at
   this repo's URL to clone it in.
3. Open `setup_secrets.py` in that Git folder and run it as a notebook:
   run Cell 1, paste your Lakebase connection string into the widget box it
   creates (never into a code cell), run Cell 2 to store it in the
   `database/lakebase-url` secret scope/key, and optionally run Cell 3 if
   the deployed app later reports it cannot read the secret. Delete this
   notebook when done — widget values persist in revision history.
4. **Compute → Apps → Create app**, pointing it at this Git folder.
   `app.yaml` supplies the start command and environment variables. The
   default auth path (`LAKEBASE_SECRET_SCOPE`/`LAKEBASE_SECRET_KEY`, already
   set in `app.yaml`) needs no extra resource binding; only the alternative
   `valueFrom: lakebase-url` path (commented out in `app.yaml`) requires
   adding a Secret resource to the app.
5. **Deploy.** The app creates its own tables on first boot
   (`ensure_weather_schema()`); there is nothing else to run before calling
   `POST /weather/sync`.
6. Confirm at `https://<your-app-url>/healthz`.

**Every time you push a change:** a Databricks Git folder does **not**
auto-pull. The sequence is always:

1. `git push` to your remote.
2. In the workspace, open the Git folder and click **Pull**.
3. Go to the app and click **Deploy** again.
4. Hard-refresh your browser (Ctrl+Shift+R / Cmd+Shift+R) — the UI's static
   JS/CSS can otherwise serve a stale cached copy even after a successful
   deploy.

Skipping step 2 is the most common way to deploy and see no change at all.

The scheduled alert-resync job (stretch goal) is a separate deployment via
Databricks Asset Bundles — see `docs/scheduled_job.md` for
`databricks bundle deploy` / `databricks bundle run` instructions.

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Renders the Weather Intelligence single-page UI. |
| `GET` | `/healthz` | Always returns 200; reports schema-bootstrap status and where Lakebase is pointed. |
| `GET` | `/api/locations` | The known city list and the default set, for the UI's location picker. |
| `POST` | `/weather/sync` | Harvests NWS alerts/forecasts/discussions for one or more locations and upserts them into `weather_documents`. |
| `POST` | `/weather/search` | Embeds a query (JSON body) and returns the top-k most similar chunks by pgvector cosine similarity, optionally summarized. |
| `GET` | `/weather/search` | The same search, via query string, for quick browser/`curl` use. |
