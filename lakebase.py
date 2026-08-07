"""
Lakebase (Databricks-managed Postgres) connection helper.

Same public surface as the reference app's lakebase.py --
get_connection() / get_engine() / run_query() / run_write() -- so app.py and
the ingestion notebook read identically to the pattern this project is built
from. What changed is what happens behind that surface.

The reference opened a brand-new psycopg2 connection on every call, and each
one first re-fetched the connection string from the Databricks secrets API.
Every read cost two network round trips before a query reached Postgres, and
get_engine() was defined and never used elsewhere in that codebase. Here the
credential is resolved once and a small connection pool is kept warm, so the
cost of a query is the query.

Three authentication paths, in precedence order:
  1. LAKEBASE_URL           -- a full DSN, set directly or via app.yaml's
                                `valueFrom` binding to a secret resource.
  2. A Databricks secret     -- scope/key given by LAKEBASE_SECRET_SCOPE /
     (the default)             LAKEBASE_SECRET_KEY, read once via the SDK.
  3. No static credential    -- mint a short-lived OAuth Postgres token from
                                the app's own service principal. Needs PGHOST
                                (and ideally LAKEBASE_INSTANCE_NAME) set.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

log = logging.getLogger("weather.lakebase")

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

_POOL_MIN = int(os.environ.get("LAKEBASE_POOL_MIN", "1"))
_POOL_MAX = int(os.environ.get("LAKEBASE_POOL_MAX", "5"))
_CONNECT_TIMEOUT = int(os.environ.get("LAKEBASE_CONNECT_TIMEOUT", "30"))
# Kept below the ~1 hour lifetime of a minted OAuth credential (see
# _TokenProvider below) so a pooled socket can never outlive the credential
# that opened it. Also bounds how long a stale password can linger in the
# pool after a rotation.
_POOL_MAX_AGE_SECONDS = int(os.environ.get("LAKEBASE_POOL_MAX_AGE_SECONDS", "1500"))

PGDATABASE = os.environ.get("PGDATABASE", "databricks_postgres")
PGSSLMODE = os.environ.get("PGSSLMODE", "require")


class LakebaseUnavailable(RuntimeError):
    """Lakebase could not be reached or is not configured. Always caught and
    reported, never allowed to crash the process at import time."""


# --------------------------------------------------------------------------
# credential resolution
# --------------------------------------------------------------------------


class _SecretResolver:
    """Reads the DSN out of a Databricks secret scope, once, and caches it."""

    def __init__(self, scope: str, key: str):
        self.scope = scope
        self.key = key
        self._value: str | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def resolve(self) -> str | None:
        if self._value is not None:
            return self._value
        with self._lock:
            if self._value is not None:
                return self._value
            if self._error is not None:
                return None
            try:
                from databricks.sdk import WorkspaceClient

                secret = WorkspaceClient().secrets.get_secret(
                    scope=self.scope, key=self.key
                )
                self._value = base64.b64decode(secret.value or "").decode("utf-8").strip()
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                self._error = f"{type(exc).__name__}: {exc}"
                log.info(
                    "No DSN from secret %s/%s (%s)", self.scope, self.key, self._error
                )
                return None
            return self._value

    @property
    def error(self) -> str | None:
        return self._error


_secret_resolver = _SecretResolver(_SCOPE, _KEY)


def _parse_dsn(dsn: str) -> dict:
    parsed = urlparse(dsn)
    if parsed.scheme not in ("postgres", "postgresql", "postgresql+psycopg2"):
        raise LakebaseUnavailable(
            "The Lakebase connection string must start with postgresql://"
        )
    if not parsed.hostname:
        raise LakebaseUnavailable("The Lakebase connection string has no host.")
    database = (parsed.path or "/").lstrip("/") or PGDATABASE
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "user": unquote(parsed.username) if parsed.username else "",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": database,
    }


def _dsn() -> str | None:
    url = (os.environ.get("LAKEBASE_URL") or "").strip()
    if url:
        return url
    return _secret_resolver.resolve()


class _TokenProvider:
    """Mints short-lived Postgres credentials from the app's OAuth identity.

    Used only when there is no static password anywhere. Tokens are cached
    until shortly before they expire.
    """

    _SKEW_SECONDS = 120

    def __init__(self, instance_name: str):
        self.instance_name = instance_name
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def token(self) -> str:
        now = time.time()
        if self._token and now < self._expires_at - self._SKEW_SECONDS:
            return self._token
        with self._lock:
            now = time.time()
            if self._token and now < self._expires_at - self._SKEW_SECONDS:
                return self._token
            self._token, self._expires_at = self._mint()
            return self._token

    def _mint(self) -> tuple[str, float]:
        try:
            from databricks.sdk import WorkspaceClient

            credential = WorkspaceClient().database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[self.instance_name]
            )
        except Exception as exc:  # noqa: BLE001
            raise LakebaseUnavailable(
                f"Could not mint a Lakebase credential for '{self.instance_name}' "
                f"({type(exc).__name__})."
            ) from exc

        token = getattr(credential, "token", None)
        if not token:
            raise LakebaseUnavailable("Databricks returned an empty Lakebase credential.")

        expiry = getattr(credential, "expiration_time", None)
        expires_at = time.time() + 3000
        if isinstance(expiry, datetime):
            when = expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)
            expires_at = when.timestamp()
        return token, expires_at


def _instance_name(host: str) -> str:
    configured = (os.environ.get("LAKEBASE_INSTANCE_NAME") or "").strip()
    if configured:
        return configured
    return host.split(".", 1)[0]


# --------------------------------------------------------------------------
# connection pool
# --------------------------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()
_pool_password: str | None = None
_pool_created_at: float = 0.0
_token_provider: _TokenProvider | None = None


def _connect_kwargs() -> dict:
    dsn = _dsn()
    if dsn:
        parts = _parse_dsn(dsn)
        host, port, user, database = (
            parts["host"],
            parts["port"],
            parts["user"],
            parts["database"],
        )
        password = parts["password"] or None
    else:
        host = (os.environ.get("PGHOST") or "").strip()
        if not host:
            raise LakebaseUnavailable(
                "Lakebase is not configured. Set LAKEBASE_URL, or store the "
                f"connection string in the secret {_SCOPE}/{_KEY}, or set PGHOST."
            )
        port = int(os.environ.get("PGPORT", "5432"))
        user = (os.environ.get("PGUSER") or "").strip()
        database = PGDATABASE
        password = None

    if not user:
        raise LakebaseUnavailable("The Lakebase connection has no username.")

    if password is None:
        global _token_provider
        if _token_provider is None or _token_provider.instance_name != _instance_name(host):
            _token_provider = _TokenProvider(_instance_name(host))
        password = _token_provider.token()

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "dbname": database,
        "sslmode": PGSSLMODE,
        "connect_timeout": _CONNECT_TIMEOUT,
        "cursor_factory": RealDictCursor,
        "application_name": "lakebase-weather-rag",
    }


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool, _pool_created_at
    with _pool_lock:
        age = time.time() - _pool_created_at
        if _pool is not None and age < _POOL_MAX_AGE_SECONDS:
            return _pool
        if _pool is not None:
            # Past its age limit -- likely an OAuth-token pool. Drop it so the
            # next connection mints (or re-fetches) a fresh credential rather
            # than handing out a socket opened with one that has expired.
            try:
                _pool.closeall()
            except Exception:  # noqa: BLE001
                pass
        kwargs = _connect_kwargs()
        _pool = psycopg2.pool.ThreadedConnectionPool(_POOL_MIN, _POOL_MAX, **kwargs)
        _pool_created_at = time.time()
        return _pool


def dispose_pool() -> None:
    """Drop the pool so the next connection re-resolves credentials."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:  # noqa: BLE001
                pass
        _pool = None


@contextmanager
def get_connection():
    """Yield a pooled psycopg2 connection with a RealDictCursor factory.

    On any exception the connection is discarded rather than returned to the
    pool -- a connection that failed mid-transaction may be in an unknown
    state, and handing it to the next caller is how you get "connection
    already closed" errors that have nothing to do with what they asked for.
    """
    pool = _get_pool()
    conn = pool.getconn()
    broken = False
    try:
        yield conn
    except Exception:
        broken = True
        raise
    finally:
        try:
            if broken:
                conn.close()
            pool.putconn(conn, close=broken)
        except Exception:  # noqa: BLE001
            pass


def get_engine():
    """A SQLAlchemy engine, for code that prefers it. Not used for pooling
    here -- get_connection()'s pool is the one every helper below shares."""
    dsn = _dsn()
    if not dsn:
        raise LakebaseUnavailable("Lakebase is not configured.")
    return create_engine(dsn)


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def target_summary() -> dict:
    """Where we are pointed and how we authenticate. Never a secret value."""
    dsn = _dsn()
    summary = {"host": None, "database": None, "user": None, "auth": "unconfigured"}
    if dsn:
        try:
            parts = _parse_dsn(dsn)
        except LakebaseUnavailable as exc:
            summary["auth"] = f"invalid connection string: {exc}"
            return summary
        summary["host"] = parts["host"]
        summary["database"] = parts["database"]
        summary["user"] = parts["user"] or None
        summary["auth"] = "connection string password" if parts["password"] else "oauth token"
    elif os.environ.get("PGHOST"):
        summary["host"] = os.environ.get("PGHOST")
        summary["database"] = PGDATABASE
        summary["user"] = os.environ.get("PGUSER") or None
        summary["auth"] = "oauth token"
    return summary
