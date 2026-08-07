"""
Offline checks for the weather RAG pipeline's Python layer: weather_client's
normalize_* functions, embedder's chunk_text/to_vector_literal, and app.py's
route behavior.

Fully offline -- no network call, no real database, and no
sentence_transformers/torch import. lakebase.get_connection and
embedder.embed_query are monkeypatched with small fakes before app.py is
imported (app.py bootstraps the schema at import time), and
weather_client's network functions (resolve_point/get_alerts/get_forecast/
get_latest_discussion) are monkeypatched only for the /weather/sync check.

Runnable as `python scripts/check_api.py` or pasted whole into a Databricks
notebook cell -- the project-root resolution below does not assume __file__
is defined.
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _candidate = Path.cwd()
    while not (_candidate / "app.py").exists() and _candidate.parent != _candidate:
        _candidate = _candidate.parent
    _PROJECT_ROOT = _candidate

sys.path.insert(0, str(_PROJECT_ROOT))

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    global PASSED
    if condition:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name} :: {detail}")
    return condition


def section(title: str) -> None:
    print(f"\n== {title} ==")


def _no_raise(fn, *args, **kwargs) -> tuple[bool, object]:
    """Call fn and report whether it raised, without letting an unexpected
    exception type crash the check script itself."""
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- reported, not re-raised
        return False, exc


def _expect_value_error(fn, *args, **kwargs) -> tuple[bool, str]:
    """True (with empty detail) iff fn(*args, **kwargs) raises ValueError --
    any other outcome (no exception, or the wrong exception type) is a
    failure, so a guard clause that silently does nothing can't pass this."""
    try:
        fn(*args, **kwargs)
        return False, "expected ValueError, no exception was raised"
    except ValueError:
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"expected ValueError, got {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# weather_documents column ground truth, read from the real .sql file --
# never hardcoded, so a column rename in the schema shows up here too.
# ----------------------------------------------------------------------------


def _weather_documents_columns() -> set[str]:
    sql_text = (_PROJECT_ROOT / "sql" / "01_weather_documents.sql").read_text(encoding="utf-8")
    sql_text = "\n".join(line.split("--", 1)[0] for line in sql_text.splitlines())

    marker = re.search(r"CREATE TABLE IF NOT EXISTS weather_documents\s*\(", sql_text, re.I)
    if not marker:
        raise ValueError("could not find CREATE TABLE weather_documents in the .sql file")

    start = marker.end()
    depth = 1
    i = start
    while depth > 0:
        if sql_text[i] == "(":
            depth += 1
        elif sql_text[i] == ")":
            depth -= 1
        i += 1
    body = sql_text[start : i - 1]

    parts = []
    depth = 0
    current = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))

    columns = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_word = part.split(None, 1)[0].strip('"').upper()
        if first_word in ("CONSTRAINT", "CHECK", "PRIMARY", "UNIQUE", "FOREIGN"):
            continue
        columns.add(part.split(None, 1)[0].strip('"'))
    return columns


import weather_client  # noqa: E402
import embedder  # noqa: E402

WEATHER_DOCUMENTS_COLUMNS = _weather_documents_columns()
check(
    "weather_documents column ground truth has 19 columns",
    len(WEATHER_DOCUMENTS_COLUMNS) == 19,
    f"got {len(WEATHER_DOCUMENTS_COLUMNS)}: {sorted(WEATHER_DOCUMENTS_COLUMNS)}",
)


def _check_doc_shape(prefix: str, doc: dict) -> None:
    keys = set(doc.keys())
    check(
        f"{prefix} keys are a subset of weather_documents columns",
        keys <= WEATHER_DOCUMENTS_COLUMNS,
        f"unexpected keys: {sorted(keys - WEATHER_DOCUMENTS_COLUMNS)}",
    )
    check(
        f"{prefix} omits exactly synced_at and embedded_at",
        WEATHER_DOCUMENTS_COLUMNS - keys == {"synced_at", "embedded_at"},
        f"missing (beyond synced_at/embedded_at): "
        f"{WEATHER_DOCUMENTS_COLUMNS - keys - {'synced_at', 'embedded_at'}}",
    )
    content_hash = doc.get("content_hash", "")
    check(
        f"{prefix} content_hash is a 40-char hex string",
        bool(re.fullmatch(r"[0-9a-f]{40}", content_hash or "")),
        f"content_hash={content_hash!r}",
    )
    check(
        f"{prefix} narrative_text is non-blank",
        bool((doc.get("narrative_text") or "").strip()),
        f"narrative_text={doc.get('narrative_text')!r}",
    )


# ----------------------------------------------------------------------------
# weather_client.normalize_alert
# ----------------------------------------------------------------------------

section("weather_client.normalize_alert")

_FAKE_ALERT_PROPERTIES = {
    "id": "urn:oid:2.49.0.1.840.0.5f6a1234abcd5678ef90.001.1",
    "event": "Flash Flood Warning",
    "headline": "Flash Flood Warning issued for Travis County",
    "severity": "Moderate",
    "description": "FAKE_DESCRIPTION_TOKEN heavy rain has fallen and additional rainfall is expected.",
    "instruction": "FAKE_INSTRUCTION_TOKEN move to higher ground immediately, do not drive through flood waters.",
    "effective": "2026-08-05T10:00:00-05:00",
    "onset": "2026-08-05T10:00:00-05:00",
    "expires": "2026-08-05T16:00:00-05:00",
    "areaDesc": "Travis, TX",
}

_alert_doc = weather_client.normalize_alert(_FAKE_ALERT_PROPERTIES, "Austin, TX")
_check_doc_shape("normalize_alert happy path", _alert_doc)
check(
    "normalize_alert joins description AND instruction (neither dropped)",
    "FAKE_DESCRIPTION_TOKEN" in _alert_doc["narrative_text"]
    and "FAKE_INSTRUCTION_TOKEN" in _alert_doc["narrative_text"],
    f"narrative_text={_alert_doc['narrative_text']!r}",
)
check(
    "normalize_alert issued_at parses well-formed ISO timestamp",
    _alert_doc["issued_at"] is not None,
    f"issued_at={_alert_doc['issued_at']!r}",
)
check(
    "normalize_alert expires_at parses well-formed ISO timestamp",
    _alert_doc["expires_at"] is not None,
    f"expires_at={_alert_doc['expires_at']!r}",
)

_malformed_alert = dict(_FAKE_ALERT_PROPERTIES)
_malformed_alert["effective"] = "not-a-timestamp"
_malformed_alert["expires"] = "not-a-timestamp"
_ok, _malformed_alert_doc = _no_raise(weather_client.normalize_alert, _malformed_alert, "Austin, TX")
check(
    "normalize_alert does not raise on malformed timestamps",
    _ok,
    f"{_malformed_alert_doc!r}" if not _ok else "",
)
if _ok:
    check(
        "normalize_alert malformed effective -> effective_at is None",
        _malformed_alert_doc["effective_at"] is None,
        f"effective_at={_malformed_alert_doc['effective_at']!r}",
    )
    check(
        "normalize_alert malformed expires -> expires_at is None",
        _malformed_alert_doc["expires_at"] is None,
        f"expires_at={_malformed_alert_doc['expires_at']!r}",
    )

_headline_only_alert = {
    "id": "urn:oid:2.49.0.1.840.0.headline-only.001.1",
    "event": None,
    "headline": "FAKE_HEADLINE_ONLY_TOKEN",
    "severity": None,
    "description": None,
    "instruction": None,
    "effective": None,
    "expires": None,
}
_headline_doc = weather_client.normalize_alert(_headline_only_alert, "Chicago, IL")
check(
    "normalize_alert falls back to headline when description/instruction absent",
    _headline_doc["narrative_text"] == "FAKE_HEADLINE_ONLY_TOKEN",
    f"narrative_text={_headline_doc['narrative_text']!r}",
)

_blank_alert = {
    "id": "urn:oid:2.49.0.1.840.0.blank.001.1",
    "event": None,
    "headline": None,
    "severity": None,
    "description": "   ",
    "instruction": None,
    "effective": None,
    "expires": None,
}
_ok, _detail = _expect_value_error(weather_client.normalize_alert, _blank_alert, "Chicago, IL")
check("normalize_alert raises ValueError with no usable text at all", _ok, _detail)


# ----------------------------------------------------------------------------
# weather_client.normalize_forecast
# ----------------------------------------------------------------------------

section("weather_client.normalize_forecast")

_FAKE_FORECAST_PROPERTIES = {
    "updated": "2026-08-05T09:00:00+00:00",
    "periods": [
        {
            "number": 1,
            "name": "Today",
            "startTime": "2026-08-05T06:00:00-05:00",
            "endTime": "2026-08-05T18:00:00-05:00",
            "isDaytime": True,
            "temperature": 84,
            "temperatureUnit": "F",
            "windSpeed": "5 to 10 mph",
            "windDirection": "S",
            "shortForecast": "Mostly Sunny",
            "detailedForecast": "FAKE_TODAY_TOKEN mostly sunny, with a high near 84. South wind 5 to 10 mph.",
        },
        {
            "number": 2,
            "name": "Tonight",
            "startTime": "2026-08-05T18:00:00-05:00",
            "endTime": "2026-08-06T06:00:00-05:00",
            "isDaytime": False,
            "temperature": 68,
            "temperatureUnit": "F",
            "windSpeed": "5 mph",
            "windDirection": "SE",
            "shortForecast": "Partly Cloudy",
            "detailedForecast": "FAKE_TONIGHT_TOKEN partly cloudy, with a low around 68.",
        },
        {
            # Deliberately missing detailedForecast, matching a real gap the
            # normalizer must skip rather than crash on.
            "number": 3,
            "name": "Tuesday",
            "shortForecast": "Sunny",
        },
    ],
}

_forecast_doc = weather_client.normalize_forecast(_FAKE_FORECAST_PROPERTIES, "Austin, TX", "EWX", 100, 80)
_check_doc_shape("normalize_forecast happy path", _forecast_doc)
check(
    "normalize_forecast joins periods with detailedForecast",
    "FAKE_TODAY_TOKEN" in _forecast_doc["narrative_text"]
    and "FAKE_TONIGHT_TOKEN" in _forecast_doc["narrative_text"],
    f"narrative_text={_forecast_doc['narrative_text']!r}",
)
check(
    "normalize_forecast skips a period missing detailedForecast rather than crashing",
    "Tuesday" not in _forecast_doc["narrative_text"],
    f"narrative_text={_forecast_doc['narrative_text']!r}",
)
check(
    "normalize_forecast issued_at parses well-formed ISO timestamp",
    _forecast_doc["issued_at"] is not None,
    f"issued_at={_forecast_doc['issued_at']!r}",
)

_malformed_forecast = dict(_FAKE_FORECAST_PROPERTIES)
_malformed_forecast["updated"] = "not-a-timestamp"
_ok, _malformed_forecast_doc = _no_raise(
    weather_client.normalize_forecast, _malformed_forecast, "Austin, TX", "EWX", 100, 80
)
check("normalize_forecast does not raise on malformed 'updated' timestamp", _ok, f"{_malformed_forecast_doc!r}" if not _ok else "")
if _ok:
    check(
        "normalize_forecast malformed 'updated' -> issued_at is None",
        _malformed_forecast_doc["issued_at"] is None,
        f"issued_at={_malformed_forecast_doc['issued_at']!r}",
    )

_ok, _detail = _expect_value_error(
    weather_client.normalize_forecast,
    {"updated": "2026-08-05T09:00:00+00:00", "periods": []},
    "Austin, TX",
    "EWX",
    100,
    80,
)
check("normalize_forecast raises ValueError on zero usable periods", _ok, _detail)

_ok, _detail = _expect_value_error(
    weather_client.normalize_forecast,
    {"updated": "2026-08-05T09:00:00+00:00", "periods": [{"number": 1, "name": "Today"}]},
    "Austin, TX",
    "EWX",
    100,
    80,
)
check("normalize_forecast raises ValueError when no period has detailedForecast", _ok, _detail)


# ----------------------------------------------------------------------------
# weather_client.normalize_discussion
# ----------------------------------------------------------------------------

section("weather_client.normalize_discussion")

_FAKE_DISCUSSION_PRODUCT = {
    "id": "3f9a2b10-4c3e-4a3f-9c2e-9a1b2c3d4e5f",
    "issuingOffice": "KLOT",
    "issuanceTime": "2026-08-05T12:00:00+00:00",
    "productName": "Area Forecast Discussion",
    "productText": "FAKE_DISCUSSION_TOKEN " + ("Weather discussion narrative text. " * 20),
}

_discussion_doc = weather_client.normalize_discussion(_FAKE_DISCUSSION_PRODUCT, "Chicago, IL", "LOT")
_check_doc_shape("normalize_discussion happy path", _discussion_doc)
check(
    "normalize_discussion id is prefixed discussion:",
    _discussion_doc["id"] == f"discussion:{_FAKE_DISCUSSION_PRODUCT['id']}",
    f"id={_discussion_doc['id']!r}",
)
check(
    "normalize_discussion narrative_text carries productText",
    "FAKE_DISCUSSION_TOKEN" in _discussion_doc["narrative_text"],
    f"narrative_text[:60]={_discussion_doc['narrative_text'][:60]!r}",
)
check(
    "normalize_discussion issued_at parses well-formed ISO timestamp",
    _discussion_doc["issued_at"] is not None,
    f"issued_at={_discussion_doc['issued_at']!r}",
)

_malformed_discussion = dict(_FAKE_DISCUSSION_PRODUCT)
_malformed_discussion["issuanceTime"] = "not-a-timestamp"
_ok, _malformed_discussion_doc = _no_raise(
    weather_client.normalize_discussion, _malformed_discussion, "Chicago, IL", "LOT"
)
check(
    "normalize_discussion does not raise on malformed issuanceTime",
    _ok,
    f"{_malformed_discussion_doc!r}" if not _ok else "",
)
if _ok:
    check(
        "normalize_discussion malformed issuanceTime -> issued_at is None",
        _malformed_discussion_doc["issued_at"] is None,
        f"issued_at={_malformed_discussion_doc['issued_at']!r}",
    )

_blank_discussion = dict(_FAKE_DISCUSSION_PRODUCT)
_blank_discussion["productText"] = "   "
_ok, _detail = _expect_value_error(weather_client.normalize_discussion, _blank_discussion, "Chicago, IL", "LOT")
check("normalize_discussion raises ValueError on blank productText", _ok, _detail)


# ----------------------------------------------------------------------------
# embedder.chunk_text
# ----------------------------------------------------------------------------

section("embedder.chunk_text")

CS = embedder.CHUNK_SIZE
CO = embedder.CHUNK_OVERLAP

check("chunk_text('') -> []", embedder.chunk_text("") == [], f"got {embedder.chunk_text('')!r}")
_whitespace_only = "   \n\t  "
check(
    "chunk_text(whitespace-only) -> []",
    embedder.chunk_text(_whitespace_only) == [],
    f"got {embedder.chunk_text(_whitespace_only)!r}",
)

_short_text = "  a short piece of text, well under chunk size  "
_short_chunks = embedder.chunk_text(_short_text)
check(
    "chunk_text(shorter than chunk_size) -> exactly one chunk",
    len(_short_chunks) == 1,
    f"got {len(_short_chunks)} chunks: {_short_chunks!r}",
)
if _short_chunks:
    check(
        "chunk_text(shorter than chunk_size) chunk equals stripped input",
        _short_chunks[0] == _short_text.strip(),
        f"got {_short_chunks[0]!r}",
    )

_exact_text = "".join(chr(ord("a") + (i % 26)) for i in range(CS))
_exact_chunks = embedder.chunk_text(_exact_text)
check(
    "chunk_text(exactly chunk_size chars) -> exactly one chunk",
    len(_exact_chunks) == 1,
    f"got {len(_exact_chunks)} chunks",
)
if _exact_chunks:
    check(
        "chunk_text(exactly chunk_size chars) chunk equals input",
        _exact_chunks[0] == _exact_text,
        "chunk did not equal the exact-length input",
    )

import random  # noqa: E402

_rng = random.Random(20260805)
_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_long_len = CS + (CO // 2) + 5
_long_text = "".join(_rng.choice(_alphabet) for _ in range(_long_len))
_long_chunks = embedder.chunk_text(_long_text)

check(
    "chunk_text(longer than chunk_size) -> more than one chunk",
    len(_long_chunks) > 1,
    f"got {len(_long_chunks)} chunk(s)",
)
if len(_long_chunks) >= 2:
    check(
        "chunk_text first chunk matches exact source slice [0:chunk_size)",
        _long_chunks[0] == _long_text[0:CS],
        "first chunk did not match the exact source slice",
    )
    check(
        "chunk_text second chunk matches exact source slice [chunk_size-overlap:end)",
        _long_chunks[1] == _long_text[CS - CO : _long_len],
        "second chunk did not match the exact source slice",
    )
    check(
        "chunk_text consecutive chunks overlap by ~chunk_overlap characters",
        _long_chunks[0][-CO:] == _long_chunks[1][:CO],
        "tail of chunk 0 did not match head of chunk 1",
    )

_ok, _detail = _expect_value_error(embedder.chunk_text, "some text", chunk_size=100, chunk_overlap=100)
check("chunk_text raises ValueError when chunk_overlap == chunk_size", _ok, _detail)

_ok, _detail = _expect_value_error(embedder.chunk_text, "some text", chunk_size=100, chunk_overlap=150)
check("chunk_text raises ValueError when chunk_overlap > chunk_size", _ok, _detail)

_ok, _detail = _expect_value_error(embedder.chunk_text, "some text", chunk_size=0, chunk_overlap=0)
check("chunk_text raises ValueError when chunk_size <= 0", _ok, _detail)


# ----------------------------------------------------------------------------
# embedder.to_vector_literal
# ----------------------------------------------------------------------------

section("embedder.to_vector_literal")

_vec = [0.1, -0.2, 0.30000001]
_literal = embedder.to_vector_literal(_vec)
check("to_vector_literal starts with '['", _literal.startswith("["), f"got {_literal!r}")
check("to_vector_literal ends with ']'", _literal.endswith("]"), f"got {_literal!r}")

_roundtrip_ok = False
_roundtrip_detail = ""
try:
    _parts = _literal.strip("[]").split(",")
    _roundtrip = [float(p) for p in _parts]
    _roundtrip_ok = len(_roundtrip) == len(_vec) and all(
        abs(a - b) < 1e-6 for a, b in zip(_roundtrip, _vec)
    )
    _roundtrip_detail = f"parsed {_roundtrip} vs original {_vec}"
except Exception as exc:  # noqa: BLE001
    _roundtrip_detail = f"{type(exc).__name__}: {exc}"
check("to_vector_literal round-trips through float() parsing", _roundtrip_ok, _roundtrip_detail)


# ----------------------------------------------------------------------------
# app.py route behavior -- lakebase and embedder faked out
# ----------------------------------------------------------------------------

section("app.py route behavior (setup)")

import lakebase  # noqa: E402

EXECUTED: list[tuple[str, object]] = []
STATE = {
    "has_embeddings": True,
    "search_rows": [],
    "upsert_row": {"previous_content_hash": None, "new_content_hash": "0" * 40},
}


class _FakeCursor:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        EXECUTED.append((sql, params))
        text = sql if isinstance(sql, str) else str(sql)
        if "SELECT EXISTS" in text:
            self._rows = [{"exists": bool(STATE["has_embeddings"])}]
        elif "WITH previous AS" in text:
            self._rows = [dict(STATE["upsert_row"])]
        elif "weather_embeddings" in text and "ORDER BY" in text:
            self._rows = [dict(row) for row in STATE["search_rows"]]
        else:
            self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConnection:
    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def _fake_get_connection():
    yield _FakeConnection()


lakebase.get_connection = _fake_get_connection
lakebase.target_summary = lambda: {
    "host": None,
    "database": None,
    "user": None,
    "auth": "unconfigured",
}
lakebase.ensure_weather_schema = lambda embedding_dim=384: {"ok": True, "error": None}

EMBED_QUERY_CALLS = {"count": 0}
_FIXED_VECTOR = [0.01] * embedder.EMBEDDING_DIM


def _fake_embed_query(text: str) -> list[float]:
    EMBED_QUERY_CALLS["count"] += 1
    return list(_FIXED_VECTOR)


embedder.embed_query = _fake_embed_query

import app as weather_app_module  # noqa: E402

check(
    "app module imports cleanly with lakebase/embedder faked out",
    hasattr(weather_app_module, "app"),
    "app module has no 'app' Flask instance",
)

client = weather_app_module.app.test_client()


section("app.py: /weather/search validation")

EXECUTED.clear()
resp = client.post("/weather/search", json={"query": "   "})
body = resp.get_json()
check("blank query -> 400", resp.status_code == 400, f"status={resp.status_code} body={body}")
check(
    "blank query -> error.code == validation_error",
    (body or {}).get("error", {}).get("code") == "validation_error",
    f"body={body}",
)

resp = client.post("/weather/search", json={"query": "flood warnings", "source_type": "not_a_real_type"})
body = resp.get_json()
check("invalid source_type -> 400", resp.status_code == 400, f"status={resp.status_code} body={body}")
check(
    "invalid source_type -> error.code == validation_error",
    (body or {}).get("error", {}).get("code") == "validation_error",
    f"body={body}",
)


section("app.py: /weather/search top_k clamping")

STATE["has_embeddings"] = True
STATE["search_rows"] = [
    {
        "document_id": "doc-1",
        "location": "Austin, TX",
        "source_type": "alert",
        "event": "Flash Flood Warning",
        "headline": "Flash Flood Warning issued",
        "severity": "Moderate",
        "issued_at": None,
        "chunk_index": 0,
        "chunk_text": "Heavy rain expected across central Texas.",
        "similarity": 0.8765,
    }
]

for _requested_top_k in (0, 500):
    EXECUTED.clear()
    resp = client.post("/weather/search", json={"query": "flood warnings", "top_k": _requested_top_k})
    body = resp.get_json()
    check(
        f"top_k={_requested_top_k} -> 200, not rejected",
        resp.status_code == 200,
        f"status={resp.status_code} body={body}",
    )
    reported_top_k = (body or {}).get("top_k")
    check(
        f"top_k={_requested_top_k} -> response top_k clamped into [1, 20]",
        isinstance(reported_top_k, int) and 1 <= reported_top_k <= 20,
        f"body={body}",
    )
    check(
        f"top_k={_requested_top_k} -> a search query ran",
        bool(EXECUTED),
        "no SQL was executed for this request",
    )
    if EXECUTED:
        _last_sql, _last_params = EXECUTED[-1]
        bound_limit = _last_params[-1] if isinstance(_last_params, (tuple, list)) else None
        check(
            f"top_k={_requested_top_k} -> bound LIMIT param is in [1, 20]",
            isinstance(bound_limit, int) and 1 <= bound_limit <= 20,
            f"bound params={_last_params}",
        )


section("app.py: /weather/search on an empty weather_embeddings table")

STATE["has_embeddings"] = False
EMBED_QUERY_CALLS["count"] = 0
resp = client.post("/weather/search", json={"query": "flood warnings near Austin"})
body = resp.get_json()
check("empty embeddings table -> 200", resp.status_code == 200, f"status={resp.status_code} body={body}")
check("empty embeddings table -> count == 0", (body or {}).get("count") == 0, f"body={body}")
check("empty embeddings table -> results == []", (body or {}).get("results") == [], f"body={body}")
check("empty embeddings table -> reason is present", bool((body or {}).get("reason")), f"body={body}")
check(
    "empty embeddings table -> embed_query was never called",
    EMBED_QUERY_CALLS["count"] == 0,
    f"embed_query was called {EMBED_QUERY_CALLS['count']} time(s)",
)
STATE["has_embeddings"] = True


section("app.py: /healthz always 200")

weather_app_module._bootstrap_result = {"ok": True, "error": None}
resp = client.get("/healthz")
body = resp.get_json()
check("healthz (ok bootstrap state) -> 200", resp.status_code == 200, f"status={resp.status_code} body={body}")
check("healthz (ok bootstrap state) -> status == ok", (body or {}).get("status") == "ok", f"body={body}")

weather_app_module._bootstrap_result = {"ok": False, "error": "RuntimeError: simulated bootstrap failure"}
resp = client.get("/healthz")
body = resp.get_json()
check(
    "healthz (degraded bootstrap state) -> 200",
    resp.status_code == 200,
    f"status={resp.status_code} body={body}",
)
check(
    "healthz (degraded bootstrap state) -> status == degraded",
    (body or {}).get("status") == "degraded",
    f"body={body}",
)


section("app.py: /weather/sync with one bad location and one good location")

import weather_client as weather_client_module  # noqa: E402


def _fake_resolve_point(lat: float, lon: float) -> dict:
    return {"office": "EWX", "grid_x": 100, "grid_y": 80, "city": "Austin", "state": "TX"}


def _fake_get_alerts(lat: float, lon: float) -> list:
    return []


def _fake_get_forecast(office: str, grid_x: int, grid_y: int) -> dict:
    return {
        "updated": "2026-08-05T09:00:00+00:00",
        "periods": [
            {
                "number": 1,
                "name": "Today",
                "detailedForecast": "Sunny, with a high near 90.",
            }
        ],
    }


def _fake_get_latest_discussion(office: str):
    return None


weather_client_module.resolve_point = _fake_resolve_point
weather_client_module.get_alerts = _fake_get_alerts
weather_client_module.get_forecast = _fake_get_forecast
weather_client_module.get_latest_discussion = _fake_get_latest_discussion

STATE["upsert_row"] = {"previous_content_hash": None, "new_content_hash": "1" * 40}

resp = client.post("/weather/sync", json={"locations": ["Austin, TX", "Nowhereville"]})
body = resp.get_json()
check("sync with a bad+good location -> 200, not a 500", resp.status_code == 200, f"status={resp.status_code} body={body}")

by_location = (body or {}).get("by_location", {})
good = by_location.get("Austin, TX", {})
bad = by_location.get("Nowhereville", {})
check("sync: bogus location recorded as a per-location error", "error" in bad, f"by_location={by_location}")
check(
    "sync: valid location alongside the bad one has no error",
    "error" not in good,
    f"good_location_result={good}",
)
check(
    "sync: valid location alongside the bad one actually synced something",
    good.get("synced", 0) >= 1,
    f"good_location_result={good}",
)


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------

section("Summary")
print(f"{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed checks:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
