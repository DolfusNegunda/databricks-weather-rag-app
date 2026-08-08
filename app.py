"""
Flask app for the weather RAG pipeline -- harvest via weather_client, persist
via lakebase, retrieve later by pgvector cosine similarity.

Route surface (this file):
  GET  /healthz         -- always 200, reports schema-bootstrap health and
                            where we're pointed, so it stays inspectable even
                            against an unreachable database.
  GET  /                 -- renders templates/index.html.
  POST /weather/sync     -- harvests NWS data for one or more locations and
                            upserts it into weather_documents.
  POST /weather/search   -- embeds a query and returns the top-k most
  GET  /weather/search      similar chunks by pgvector cosine similarity,
                            optionally summarized by a Foundation Model
                            endpoint. Same underlying search, two ways in
                            (JSON body vs. query string) for convenience.
"""

from __future__ import annotations

import hashlib
import logging
import os

from dotenv import load_dotenv

# Must run before importing embedder/lakebase/weather_client below: all three
# read os.environ.get(...) at import time (EMBEDDING_DIM, LAKEBASE_URL, the
# NWS base URL/User-Agent), so a .env file loaded any later would arrive too
# late for those defaults to pick it up. load_dotenv() is a no-op if there is
# no .env file, which is the normal case in a deployed Databricks App where
# app.yaml supplies the environment directly.
load_dotenv()

import requests
from flask import Flask, jsonify, render_template, request
from psycopg2.extras import Json

import embedder
import lakebase
import weather_client

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")


# --------------------------------------------------------------------------
# startup: bring the schema up to date once, never let it take the app down
# --------------------------------------------------------------------------

_bootstrap_result: dict = {"ok": False, "error": "bootstrap has not run yet"}


def _bootstrap() -> None:
    """Apply the weather schema at import time.

    ensure_weather_schema() already captures its own failures into the
    returned dict rather than raising, but this still wraps the call --
    /healthz must be able to explain *why* the app is degraded, and the one
    thing that can never happen is an exception here taking the Flask app
    object down with it.
    """
    global _bootstrap_result
    try:
        embedding_dim = int(os.environ.get("EMBEDDING_DIM", "384"))
        result = lakebase.ensure_weather_schema(embedding_dim=embedding_dim)
    except Exception as exc:  # noqa: BLE001 -- must never escape import time
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    _bootstrap_result = result
    if result.get("ok"):
        logger.info("Weather schema bootstrap ok")
    else:
        logger.warning("Weather schema bootstrap degraded: %s", result.get("error"))


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------


@app.errorhandler(Exception)
def handle_exception(err):
    logger.exception("Unhandled exception")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": {"code": "internal_error", "message": str(err)}}), status_code


# --------------------------------------------------------------------------
# pages / health
# --------------------------------------------------------------------------


@app.route("/healthz", methods=["GET"])
def healthz():
    return (
        jsonify(
            {
                "status": "ok" if _bootstrap_result.get("ok") else "degraded",
                "bootstrap_error": _bootstrap_result.get("error"),
                "lakebase": lakebase.target_summary(),
            }
        ),
        200,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/locations", methods=["GET"])
def list_locations():
    """The known city list, for the UI's location picker.

    Exists so the frontend never hardcodes a second copy of
    weather_client.LOCATIONS that could drift out of sync with the one
    resolve_location() actually checks against -- there is exactly one
    source of truth for "what location strings actually work," and this
    just exposes it as JSON.
    """
    return jsonify(
        {
            "locations": list(weather_client.LOCATIONS.keys()),
            "default_locations": [
                loc.strip()
                for loc in os.environ.get(
                    "WEATHER_DEFAULT_LOCATIONS", "Chicago, IL|Austin, TX|Miami, FL"
                ).split("|")
                if loc.strip()
            ],
        }
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

_UPSERT_WEATHER_DOCUMENT_SQL = """
WITH previous AS (
    SELECT content_hash FROM weather_documents WHERE id = %(id)s
)
INSERT INTO weather_documents (
    id, location, latitude, longitude, office, grid_x, grid_y,
    source_type, event, headline, severity, narrative_text,
    issued_at, effective_at, expires_at, payload, synced_at, content_hash
) VALUES (
    %(id)s, %(location)s, %(latitude)s, %(longitude)s, %(office)s, %(grid_x)s, %(grid_y)s,
    %(source_type)s, %(event)s, %(headline)s, %(severity)s, %(narrative_text)s,
    %(issued_at)s, %(effective_at)s, %(expires_at)s, %(payload)s, now(), %(content_hash)s
)
ON CONFLICT (id) DO UPDATE SET
    location = EXCLUDED.location,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    office = EXCLUDED.office,
    grid_x = EXCLUDED.grid_x,
    grid_y = EXCLUDED.grid_y,
    event = EXCLUDED.event,
    headline = EXCLUDED.headline,
    severity = EXCLUDED.severity,
    narrative_text = EXCLUDED.narrative_text,
    issued_at = EXCLUDED.issued_at,
    effective_at = EXCLUDED.effective_at,
    expires_at = EXCLUDED.expires_at,
    payload = EXCLUDED.payload,
    synced_at = now(),
    content_hash = EXCLUDED.content_hash,
    embedded_at = CASE
        WHEN weather_documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
        THEN NULL
        ELSE weather_documents.embedded_at
    END
RETURNING
    (SELECT content_hash FROM previous) AS previous_content_hash,
    content_hash AS new_content_hash
"""


def _upsert_weather_document(doc: dict) -> bool:
    """Insert or update one weather_documents row.

    Returns True if this was a first-time insert or the narrative text
    actually changed since the last sync (either way embedded_at was just
    cleared, so the row is due for (re-)embedding); False if this was a
    re-sync of unchanged text.

    The `previous` CTE reads content_hash from before this statement's
    write, which is the one piece of information a plain RETURNING on an
    INSERT ... ON CONFLICT DO UPDATE cannot give you -- by the time
    RETURNING evaluates, the row already reflects the new values.
    """
    params = dict(doc)
    params["payload"] = Json(doc["payload"])

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_UPSERT_WEATHER_DOCUMENT_SQL, params)
            row = cur.fetchone()
        conn.commit()

    previous_hash = row["previous_content_hash"]
    return previous_hash is None or previous_hash != row["new_content_hash"]


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


def _requested_source_types(body: dict) -> tuple[list[str], list[str]]:
    requested = body.get("source_types") or list(weather_client.DEFAULT_SOURCE_TYPES)
    kept = [s for s in requested if s in weather_client.DEFAULT_SOURCE_TYPES]
    dropped = [s for s in requested if s not in weather_client.DEFAULT_SOURCE_TYPES]
    return kept, dropped


def _sync_location(location: str, source_types: list[str], limit: int, by_source_type: dict) -> dict:
    """Sync every requested source_type for one location.

    Returns this location's own result dict. Errors from resolving the
    location or its grid point stop processing for *this* location only --
    they're recorded on its result, never raised, so one bad location can't
    take the rest of the request down with it.
    """
    result = {"considered": 0, "synced": 0}

    try:
        lat, lon = weather_client.resolve_location(location)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    try:
        point = weather_client.resolve_point(lat, lon)
    except requests.RequestException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    office = point.get("office")
    grid_x = point.get("grid_x")
    grid_y = point.get("grid_y")

    if "alert" in source_types:
        _sync_alerts(location, lat, lon, limit, result, by_source_type)
    if "forecast" in source_types:
        _sync_forecast(location, office, grid_x, grid_y, result, by_source_type)
    if "discussion" in source_types:
        _sync_discussion(location, office, result, by_source_type)

    return result


def _record_error(result: dict, source_type: str, exc: Exception) -> None:
    result.setdefault("errors", {})[source_type] = f"{type(exc).__name__}: {exc}"


def _sync_alerts(location: str, lat: float, lon: float, limit: int, result: dict, by_source_type: dict) -> None:
    try:
        features = weather_client.get_alerts(lat, lon)
    except requests.RequestException as exc:
        _record_error(result, "alert", exc)
        return

    for properties in features[:limit]:
        result["considered"] += 1
        by_source_type["alert"]["considered"] += 1
        try:
            doc = weather_client.normalize_alert(properties, location)
        except ValueError:
            continue
        if _upsert_weather_document(doc):
            result["synced"] += 1
            by_source_type["alert"]["synced"] += 1


def _sync_forecast(location: str, office: str, grid_x: int, grid_y: int, result: dict, by_source_type: dict) -> None:
    try:
        properties = weather_client.get_forecast(office, grid_x, grid_y)
    except requests.RequestException as exc:
        _record_error(result, "forecast", exc)
        return

    result["considered"] += 1
    by_source_type["forecast"]["considered"] += 1
    try:
        doc = weather_client.normalize_forecast(properties, location, office, grid_x, grid_y)
    except ValueError:
        return
    if _upsert_weather_document(doc):
        result["synced"] += 1
        by_source_type["forecast"]["synced"] += 1


def _sync_discussion(location: str, office: str, result: dict, by_source_type: dict) -> None:
    try:
        product = weather_client.get_latest_discussion(office)
    except requests.RequestException as exc:
        _record_error(result, "discussion", exc)
        return

    if product is None:
        return

    result["considered"] += 1
    by_source_type["discussion"]["considered"] += 1
    try:
        doc = weather_client.normalize_discussion(product, location, office)
    except ValueError:
        return
    if _upsert_weather_document(doc):
        result["synced"] += 1
        by_source_type["discussion"]["synced"] += 1


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    body = request.get_json(silent=True) or {}

    locations = body.get("locations") or os.environ.get(
        "WEATHER_DEFAULT_LOCATIONS", "Chicago, IL|Austin, TX|Miami, FL"
    ).split("|")
    if isinstance(locations, str):
        locations = locations.split("|")
    try:
        limit = int(body.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    source_types, dropped_source_types = _requested_source_types(body)

    by_source_type = {
        source_type: {"considered": 0, "synced": 0} for source_type in weather_client.DEFAULT_SOURCE_TYPES
    }
    by_location = {}

    for raw_location in locations:
        location = raw_location.strip()
        if not location:
            continue
        by_location[location] = _sync_location(location, source_types, limit, by_source_type)

    response = {
        "synced": sum(entry["synced"] for entry in by_location.values()),
        "considered": sum(entry["considered"] for entry in by_location.values()),
        "by_location": by_location,
        "by_source_type": by_source_type,
    }
    if dropped_source_types:
        response["dropped_source_types"] = dropped_source_types

    return jsonify(response)


# --------------------------------------------------------------------------
# search / RAG
# --------------------------------------------------------------------------

_WEATHER_SEARCH_SQL = """
SELECT
    d.id AS document_id,
    d.location,
    d.source_type,
    d.event,
    d.headline,
    d.severity,
    d.issued_at,
    e.chunk_index,
    e.chunk_text,
    1 - (e.embedding <=> %s::vector) AS similarity
FROM weather_embeddings e
JOIN weather_documents d ON d.id = e.document_id
WHERE (%s::text IS NULL OR d.source_type = %s)
ORDER BY e.embedding <=> %s::vector
LIMIT %s
"""


def _run_search(query: str, top_k: int, source_type: str | None) -> dict:
    query = str(query or "").strip()
    if not query:
        raise ValueError("query is required")

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(20, top_k))

    source_type = str(source_type or "").strip() or None
    if source_type and source_type not in weather_client.DEFAULT_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {list(weather_client.DEFAULT_SOURCE_TYPES)}, "
            f"got {source_type!r}"
        )

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM weather_embeddings LIMIT 1)")
            has_embeddings = cur.fetchone()["exists"]

    if not has_embeddings:
        return {
            "query": query,
            "top_k": top_k,
            "source_type": source_type,
            "results": [],
            "count": 0,
            "reason": (
                "No weather documents have been embedded yet. Run "
                "notebooks/ingest_weather_embeddings.py after syncing some "
                "documents with POST /weather/sync."
            ),
        }

    vector_literal = embedder.to_vector_literal(embedder.embed_query(query))

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                _WEATHER_SEARCH_SQL,
                (vector_literal, source_type, source_type, vector_literal, top_k),
            )
            rows = cur.fetchall()

    results = [
        {
            "document_id": row["document_id"],
            "location": row["location"],
            "source_type": row["source_type"],
            "event": row["event"],
            "headline": row["headline"],
            "severity": row["severity"],
            "issued_at": row["issued_at"].isoformat() if row["issued_at"] else None,
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
            "similarity": round(float(row["similarity"]), 4),
        }
        for row in rows
    ]

    return {
        "query": query,
        "top_k": top_k,
        "source_type": source_type,
        "results": results,
        "count": len(results),
    }


def _summarize_results(query: str, results: list[dict]) -> dict:
    """RAG-answer the query using only the retrieved chunks.

    Wraps the whole Foundation Model call in one broad except: whether the
    endpoint is provisioned in this workspace, and the exact response shape
    a given databricks-sdk version returns, can't be verified from here, and
    a wrong guess must degrade to summary_error rather than 500 the request
    -- the same "optional feature, missing is fine" pattern embedder.py uses
    for a missing sentence-transformers install.
    """
    if not results:
        return {"summary": None, "summary_error": "No results to summarize."}

    instruction = (
        "You are a weather assistant. Answer the user's question using only "
        "the weather documents provided below. Do not invent facts that are "
        "not present in them. If the documents don't contain enough "
        "information to answer, say so."
    )
    lines = [
        f"[{row['location']} - {row['source_type']} - {row['headline'] or row['event'] or 'weather update'}]: "
        f"{row['chunk_text']}"
        for row in results
    ]
    user_content = query + "\n\n" + "\n".join(lines)

    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        endpoint = os.environ.get(
            "FOUNDATION_MODEL_ENDPOINT", "databricks-meta-llama-3-1-70b-instruct"
        )
        response = WorkspaceClient().serving_endpoints.query(
            name=endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=instruction),
                ChatMessage(role=ChatMessageRole.USER, content=user_content),
            ],
        )
        text = response.choices[0].message.content
        return {"summary": text.strip(), "summary_error": None}
    except Exception as exc:  # noqa: BLE001 -- see docstring
        return {"summary": None, "summary_error": f"{type(exc).__name__}: {exc}"}


@app.route("/weather/search", methods=["POST"])
def weather_search_post():
    body = request.get_json(silent=True) or {}

    try:
        result = _run_search(body.get("query", ""), body.get("top_k", 5), body.get("source_type"))
    except ValueError as exc:
        return jsonify({"error": {"code": "validation_error", "message": str(exc)}}), 400
    except lakebase.LakebaseUnavailable as exc:
        return jsonify({"error": {"code": "database_unavailable", "message": str(exc)}}), 503
    except RuntimeError as exc:
        return jsonify({"error": {"code": "embedding_unavailable", "message": str(exc)}}), 503

    if body.get("summarize"):
        result.update(_summarize_results(result["query"], result["results"]))

    return jsonify(result), 200


@app.route("/weather/search", methods=["GET"])
def weather_search_get():
    summarize = request.args.get("summarize", "").strip().lower() in ("true", "1", "yes")

    try:
        result = _run_search(
            request.args.get("query", ""),
            request.args.get("top_k", 5),
            request.args.get("source_type"),
        )
    except ValueError as exc:
        return jsonify({"error": {"code": "validation_error", "message": str(exc)}}), 400
    except lakebase.LakebaseUnavailable as exc:
        return jsonify({"error": {"code": "database_unavailable", "message": str(exc)}}), 503
    except RuntimeError as exc:
        return jsonify({"error": {"code": "embedding_unavailable", "message": str(exc)}}), 503

    if summarize:
        result.update(_summarize_results(result["query"], result["results"]))

    return jsonify(result), 200


_bootstrap()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.environ.get("FLASK_RUN_PORT", "8000")),
    )
