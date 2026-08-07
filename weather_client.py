"""
National Weather Service (api.weather.gov) client.

Two layers, kept separate on purpose:
  - network functions (resolve_point, get_alerts, get_forecast,
    get_latest_discussion) -- thin wrappers over requests.Session that return
    raw API JSON (or the most useful sub-dict of it) and raise
    requests.HTTPError on a non-2xx response. They never catch that error --
    the caller (the /weather/sync route) decides how to handle one location
    or one office failing without aborting the sync for the others.
  - normalize functions (normalize_alert, normalize_forecast,
    normalize_discussion) -- pure, no network call, take already-fetched
    JSON and return a dict shaped exactly like a weather_documents row so
    they're trivial to unit test.

Contract every normalize_* function honors, for whoever builds the INSERT:
  - The returned dict's keys are a subset of weather_documents' columns,
    omitting exactly `synced_at` and `embedded_at` -- both have DB-side
    defaults (`now()` and NULL respectively), and `embedded_at` staying NULL
    is the sentinel idx_weather_documents_unembedded and the ingestion
    notebook key on to find rows that still need embedding. Do not set it
    here.
  - `payload` is a live Python dict, not a JSON string. weather_documents.payload
    is JSONB and psycopg2 will not adapt a bare dict -- the caller must
    `json.dumps(doc["payload"])` or wrap it in `psycopg2.extras.Json(...)`
    before binding it as a query parameter.
  - `issued_at` / `effective_at` / `expires_at` are `datetime` objects (or
    None), not ISO strings -- psycopg2 adapts these natively for a
    TIMESTAMPTZ column.
  - `narrative_text` is guaranteed non-blank when a dict is returned at all.
    weather_documents has `CHECK (length(btrim(narrative_text)) > 0)`; rather
    than let that surface as an opaque CheckViolation at INSERT time, every
    normalize_* function raises ValueError itself when it would otherwise
    produce a blank narrative. Callers should wrap each normalize_* call in
    `except ValueError` and skip that one document.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime

import requests

NWS_API_BASE_URL = (os.environ.get("NWS_API_BASE_URL") or "https://api.weather.gov").rstrip("/")
NWS_USER_AGENT = os.environ.get("NWS_USER_AGENT") or (
    "lakebase-weather-rag (bootcamp homework; contact via github.com/DolfusNegunda)"
)

_TIMEOUT_SECONDS = 15

DEFAULT_SOURCE_TYPES = ("alert", "forecast", "discussion")

LOCATIONS: dict[str, tuple[float, float]] = {
    "Chicago, IL": (41.8781, -87.6298),
    "Austin, TX": (30.2672, -97.7431),
    "Miami, FL": (25.7617, -80.1918),
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Seattle, WA": (47.6062, -122.3321),
    "Denver, CO": (39.7392, -104.9903),
    "Atlanta, GA": (33.7490, -84.3880),
    "Boston, MA": (42.3601, -71.0589),
    "Phoenix, AZ": (33.4484, -112.0740),
}

_LOCATIONS_BY_KEY = {name.strip().lower(): coords for name, coords in LOCATIONS.items()}

_session = requests.Session()
_session.headers.update({"User-Agent": NWS_USER_AGENT})

# Grid assignment for a point never changes, so this cache never needs to
# expire or be bounded -- keyed by (lat, lon) already rounded to 4 decimals.
_point_cache: dict[tuple[float, float], dict] = {}


def resolve_location(location: str) -> tuple[float, float]:
    """Parse "lat,lon" directly, or look up a "City, ST" in LOCATIONS.

    Case-insensitive on the "City, ST" form after stripping surrounding
    whitespace. Raises ValueError if `location` matches neither shape --
    this is a small, honest lookup table, not a geocoding service.
    """
    raw = location.strip()

    lat_str, sep, lon_str = raw.partition(",")
    if sep:
        try:
            return (float(lat_str.strip()), float(lon_str.strip()))
        except ValueError:
            pass

    coords = _LOCATIONS_BY_KEY.get(raw.lower())
    if coords is not None:
        return coords

    raise ValueError(
        f"Unknown location {location!r}: must be \"City, ST\" (one of "
        f"{', '.join(LOCATIONS)}) or a raw \"lat,lon\" pair."
    )


def resolve_point(lat: float, lon: float) -> dict:
    """GET /points/{lat},{lon} -> office/grid/city/state for that point.

    lat/lon are rounded to 4 decimal places before hitting the URL or the
    cache -- NWS grid assignment does not change at that precision.
    """
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    key = (lat_r, lon_r)
    if key in _point_cache:
        return _point_cache[key]

    url = f"{NWS_API_BASE_URL}/points/{lat_r},{lon_r}"
    response = _session.get(url, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    properties = response.json()["properties"]
    relative_location = (properties.get("relativeLocation") or {}).get("properties") or {}

    result = {
        "office": properties.get("gridId"),
        "grid_x": properties.get("gridX"),
        "grid_y": properties.get("gridY"),
        "city": relative_location.get("city"),
        "state": relative_location.get("state"),
    }
    _point_cache[key] = result
    return result


def get_alerts(lat: float, lon: float) -> list[dict]:
    """GET /alerts/active?point={lat},{lon} -> list of feature `properties` dicts.

    An empty list is a normal, successful result -- an entire state can have
    zero active alerts at a given moment.
    """
    url = f"{NWS_API_BASE_URL}/alerts/active"
    response = _session.get(
        url, params={"point": f"{lat},{lon}"}, timeout=_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    return [feature["properties"] for feature in features]


def get_forecast(office: str, grid_x: int, grid_y: int) -> dict:
    """GET /gridpoints/{office}/{grid_x},{grid_y}/forecast -> the `properties` dict."""
    url = f"{NWS_API_BASE_URL}/gridpoints/{office}/{grid_x},{grid_y}/forecast"
    response = _session.get(url, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["properties"]


def get_latest_discussion(office: str) -> dict | None:
    """Fetch the newest Area Forecast Discussion product for `office`.

    Returns None if the office has no AFD products -- not an error.
    """
    graph_url = f"{NWS_API_BASE_URL}/products/types/AFD/locations/{office}"
    graph_response = _session.get(graph_url, timeout=_TIMEOUT_SECONDS)
    graph_response.raise_for_status()
    graph = graph_response.json().get("@graph", [])
    if not graph:
        return None

    product_id = graph[0]["id"]
    product_response = _session.get(
        f"{NWS_API_BASE_URL}/products/{product_id}", timeout=_TIMEOUT_SECONDS
    )
    product_response.raise_for_status()
    return product_response.json()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_alert(properties: dict, location: str) -> dict:
    """Turn one /alerts/active feature `properties` dict into a weather_documents row.

    Raises ValueError if no narrative text at all is available (description,
    instruction, headline, AND event all missing/blank) -- the caller should
    catch this and skip the alert rather than insert a blank row.
    """
    description = (properties.get("description") or "").strip()
    instruction = (properties.get("instruction") or "").strip()
    parts = [part for part in (description, instruction) if part]
    if parts:
        narrative_text = "\n\n".join(parts)
    else:
        narrative_text = (properties.get("headline") or properties.get("event") or "").strip()

    if not narrative_text:
        raise ValueError(f"No narrative text available for alert {properties.get('id')!r}")

    effective = _parse_timestamp(properties.get("effective"))

    return {
        "id": properties["id"],
        "location": location,
        "latitude": None,
        "longitude": None,
        "office": None,
        "grid_x": None,
        "grid_y": None,
        "source_type": "alert",
        "event": properties.get("event"),
        "headline": properties.get("headline"),
        "severity": properties.get("severity"),
        "narrative_text": narrative_text,
        "issued_at": effective,
        "effective_at": effective,
        "expires_at": _parse_timestamp(properties.get("expires")),
        "payload": properties,
        "content_hash": hashlib.sha1(narrative_text.encode("utf-8")).hexdigest(),
    }


def normalize_forecast(
    properties: dict, location: str, office: str, grid_x: int, grid_y: int
) -> dict:
    """Turn one /gridpoints/.../forecast `properties` dict into one weather_documents row.

    One document per forecast fetch, not one per period. Raises ValueError if
    there are no periods with a detailedForecast to join into a narrative --
    the caller should catch this and skip rather than insert a blank row.
    """
    periods = properties.get("periods") or []
    narrative_text = "\n".join(
        f"{period['name']}: {period['detailedForecast']}"
        for period in periods
        if period.get("detailedForecast")
    )

    if not narrative_text:
        raise ValueError(f"No forecast periods with text for {location!r}")

    updated = properties.get("updated")
    issued = _parse_timestamp(updated)
    doc_id = hashlib.sha1(f"forecast:{location}:{updated}".encode("utf-8")).hexdigest()

    return {
        "id": doc_id,
        "location": location,
        "latitude": None,
        "longitude": None,
        "office": office,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "source_type": "forecast",
        "event": None,
        "headline": periods[0]["name"] if periods else None,
        "severity": None,
        "narrative_text": narrative_text,
        "issued_at": issued,
        "effective_at": issued,
        "expires_at": None,
        "payload": properties,
        "content_hash": hashlib.sha1(narrative_text.encode("utf-8")).hexdigest(),
    }


def normalize_discussion(product: dict, location: str, office: str) -> dict:
    """Turn one /products/{id} discussion dict into a weather_documents row.

    Raises ValueError when productText is blank/missing -- the caller should
    catch this and skip creating a document for that office rather than
    insert a blank row.
    """
    narrative_text = (product.get("productText") or "").strip()
    if not narrative_text:
        raise ValueError(f"Blank productText for discussion product {product.get('id')!r}")

    issued = _parse_timestamp(product.get("issuanceTime"))

    return {
        "id": f"discussion:{product['id']}",
        "location": location,
        "latitude": None,
        "longitude": None,
        "office": office,
        "grid_x": None,
        "grid_y": None,
        "source_type": "discussion",
        "event": None,
        "headline": product.get("productName"),
        "severity": None,
        "narrative_text": narrative_text,
        "issued_at": issued,
        "effective_at": issued,
        "expires_at": None,
        "payload": product,
        "content_hash": hashlib.sha1(narrative_text.encode("utf-8")).hexdigest(),
    }
