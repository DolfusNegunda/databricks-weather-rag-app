"""
Live diagnostics against a real Lakebase Postgres instance.

Counterpart to this repo's offline checks (grammar/shape checks over the SQL
and API code that never touch a database) -- this script is the one that
actually connects. Run it once Lakebase is configured:

    python scripts/check_connection.py            # read-only checks (1-6)
    python scripts/check_connection.py --write     # + a write/read round
                                                    # trip (7-8), self-cleaning

Follows a uniform report convention throughout: check(name, condition, detail)
prints "PASS name" or "FAIL name :: detail" and tallies a running total;
section(title) prints a banner between groups; main() exits 1 if anything
failed (including "no target configured" -- see section 1).
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psycopg2.extras import Json

import embedder
import lakebase

_passed = 0
_failed = 0


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check(name: str, condition: bool, detail: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} :: {detail}")
    return condition


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine similarity, computed independently of pgvector's
    <=> operator so section 7 has two separate implementations to disagree,
    not one formula checked against itself."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------
# 1. target
# --------------------------------------------------------------------------


def _check_target() -> bool:
    section("1. Target")
    summary = lakebase.target_summary()
    print(f"  host    : {summary['host']}")
    print(f"  database: {summary['database']}")
    print(f"  user    : {summary['user']}")
    print(f"  auth    : {summary['auth']}")
    return check(
        "lakebase_target_configured",
        bool(summary["host"]),
        "no host in LAKEBASE_URL / a configured secret / PGHOST",
    )


# --------------------------------------------------------------------------
# 2. connect
# --------------------------------------------------------------------------


def _check_connect() -> None:
    section("2. Connect")
    try:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version(), current_user")
                row = cur.fetchone()
        if check("connect_and_select_version", bool(row and row.get("version")), f"row={row}"):
            print(f"  version      : {row['version']}")
            print(f"  current_user : {row['current_user']}")
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        check("connect_and_select_version", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 3. schema bootstrap
# --------------------------------------------------------------------------


def _check_schema_bootstrap() -> None:
    section("3. Schema bootstrap")
    # embedder.EMBEDDING_DIM, not a hardcoded 384: if EMBEDDING_DIM is
    # overridden via the environment, this must bootstrap the same width
    # section 5 later checks for, or a real override would bootstrap one
    # dimension while every other consumer expects another.
    result = lakebase.ensure_weather_schema(embedding_dim=embedder.EMBEDDING_DIM)
    check("ensure_weather_schema_ok", result.get("ok") is True, f"result={result}")


# --------------------------------------------------------------------------
# 4. pgvector extension
# --------------------------------------------------------------------------


def _check_pgvector_extension() -> None:
    section("4. pgvector extension")
    try:
        rows = lakebase.run_query("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        check("pgvector_extension_installed", len(rows) == 1, f"rows={rows}")
    except Exception as exc:  # noqa: BLE001
        check("pgvector_extension_installed", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 5. table columns
# --------------------------------------------------------------------------

_EXPECTED_COLUMNS = {
    "weather_documents": {
        "id", "location", "latitude", "longitude", "office", "grid_x", "grid_y",
        "source_type", "event", "headline", "severity", "narrative_text",
        "issued_at", "effective_at", "expires_at", "payload", "synced_at",
        "content_hash", "embedded_at",
    },
    "weather_embeddings": {
        "id", "document_id", "chunk_index", "chunk_text", "embedding",
        "model_name", "created_at",
    },
}


def _check_columns() -> None:
    section("5. Table columns")
    for table, expected in _EXPECTED_COLUMNS.items():
        try:
            rows = lakebase.run_query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s",
                (table,),
            )
            found = {r["column_name"] for r in rows}
            missing = expected - found
            check(f"{table}_has_expected_columns", not missing, f"missing={sorted(missing)}")
        except Exception as exc:  # noqa: BLE001
            check(f"{table}_has_expected_columns", False, f"{type(exc).__name__}: {exc}")

    # information_schema reports pgvector columns as data_type "USER-DEFINED"
    # with udt_name "vector" -- not a specific type name -- so that is what
    # this checks for rather than assuming.
    try:
        rows = lakebase.run_query(
            "SELECT data_type, udt_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'weather_embeddings' "
            "AND column_name = 'embedding'"
        )
        col = rows[0] if rows else None
        check(
            "weather_embeddings_embedding_is_vector_type",
            bool(col) and col["data_type"] == "USER-DEFINED" and col["udt_name"] == "vector",
            f"got {col}",
        )
    except Exception as exc:  # noqa: BLE001
        check("weather_embeddings_embedding_is_vector_type", False, f"{type(exc).__name__}: {exc}")

    # information_schema's udt_name is just "vector" regardless of dimension --
    # a column created at the wrong width (e.g. a stale 768-dim table from a
    # model swap) would pass the check above cleanly. pg_attribute's
    # format_type() is what actually prints the "(384)" part.
    try:
        rows = lakebase.run_query(
            "SELECT format_type(a.atttypid, a.atttypmod) AS coltype "
            "FROM pg_attribute a "
            "WHERE a.attrelid = 'weather_embeddings'::regclass "
            "AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped"
        )
        coltype = rows[0]["coltype"] if rows else None
        expected_dim = embedder.EMBEDDING_DIM
        print(f"  weather_embeddings.embedding type: {coltype} (expected vector({expected_dim}))")
        check(
            "weather_embeddings_embedding_dimension_matches",
            coltype == f"vector({expected_dim})",
            f"got {coltype}, expected vector({expected_dim}) from EMBEDDING_DIM",
        )
    except Exception as exc:  # noqa: BLE001
        check("weather_embeddings_embedding_dimension_matches", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 6. HNSW index + query plan
# --------------------------------------------------------------------------


def _check_hnsw_and_plan() -> None:
    section("6. HNSW index and query plan")
    try:
        rows = lakebase.run_query(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'weather_embeddings'"
        )
        hnsw_indexes = [r["indexname"] for r in rows if "hnsw" in r["indexdef"].lower()]
        check("hnsw_index_exists", bool(hnsw_indexes), f"indexes_found={[r['indexname'] for r in rows]}")
    except Exception as exc:  # noqa: BLE001
        check("hnsw_index_exists", False, f"{type(exc).__name__}: {exc}")
        return

    try:
        row_count = lakebase.run_query("SELECT count(*) AS n FROM weather_embeddings")[0]["n"]
    except Exception as exc:  # noqa: BLE001
        check("query_plan_probe_ran", False, f"{type(exc).__name__}: {exc}")
        return

    dim = embedder.EMBEDDING_DIM
    probe_literal = embedder.to_vector_literal([1.0] + [0.0] * (dim - 1))

    try:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "EXPLAIN SELECT id FROM weather_embeddings "
                    "ORDER BY embedding <=> %s::vector LIMIT 5",
                    (probe_literal,),
                )
                plan_rows = cur.fetchall()
        plan_text = "\n".join(str(list(row.values())[0]) for row in plan_rows)
        check("query_plan_probe_ran", True)

        uses_index_scan = "index scan" in plan_text.lower()
        print(f"  rows in weather_embeddings: {row_count}")
        print(f"  planner chose index scan : {uses_index_scan}")
        if not uses_index_scan:
            print(
                f"  -> sequential scan chosen, not a failure by itself: with only "
                f"{row_count} row(s) Postgres's own cost estimator can reasonably "
                "prefer a full scan over the index. Re-check once the table holds "
                "a realistic number of embedded chunks."
            )
        print("  plan:")
        for line in plan_text.splitlines():
            print(f"    {line}")
    except Exception as exc:  # noqa: BLE001
        check("query_plan_probe_ran", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 7/8. write round trip: semantic check, then cleanup
# --------------------------------------------------------------------------


def _run_write_round_trip() -> None:
    dim = embedder.EMBEDDING_DIM
    doc_id = f"check_connection_test:{uuid.uuid4()}"
    embedding_id = f"{doc_id}:0"

    section("7. Semantic round trip (cosine correctness)")
    try:
        lakebase.run_write(
            "INSERT INTO weather_documents (id, location, source_type, narrative_text, payload) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                doc_id,
                "check_connection.py test entry",
                "forecast",
                "Throwaway narrative text inserted by scripts/check_connection.py.",
                Json({"note": "throwaway row; safe to delete", "source": "check_connection.py"}),
            ),
        )
        check("insert_throwaway_document", True)

        # Hand-built, deterministic, model-free -- this proves the database's
        # vector arithmetic, not the embedding model, so it does not need
        # sentence-transformers installed or a real embedding computed.
        vector_a = [math.sin(i * 0.1) for i in range(dim)]
        vector_b = [math.cos(i * 0.37 + 1.0) for i in range(dim)]
        literal_a = embedder.to_vector_literal(vector_a)
        literal_b = embedder.to_vector_literal(vector_b)

        lakebase.run_write(
            "INSERT INTO weather_embeddings "
            "(id, document_id, chunk_index, chunk_text, embedding, model_name) "
            "VALUES (%s, %s, %s, %s, %s::vector, %s)",
            (embedding_id, doc_id, 0, "throwaway chunk text", literal_a, "check_connection.py-synthetic"),
        )
        check("insert_throwaway_embedding", True)

        self_rows = lakebase.run_query(
            "SELECT 1 - (embedding <=> %s::vector) AS similarity FROM weather_embeddings WHERE id = %s",
            (literal_a, embedding_id),
        )
        self_similarity = float(self_rows[0]["similarity"]) if self_rows else None
        check(
            "self_similarity_is_one",
            self_similarity is not None and abs(self_similarity - 1.0) < 1e-6,
            f"a vector compared against itself should give cosine similarity 1.0, got {self_similarity}",
        )

        cross_rows = lakebase.run_query(
            "SELECT 1 - (embedding <=> %s::vector) AS similarity FROM weather_embeddings WHERE id = %s",
            (literal_b, embedding_id),
        )
        pg_similarity = float(cross_rows[0]["similarity"]) if cross_rows else None
        python_similarity = cosine_similarity(vector_a, vector_b)
        check(
            "postgres_matches_python_cosine",
            pg_similarity is not None and abs(pg_similarity - python_similarity) < 1e-4,
            f"postgres={pg_similarity} python={python_similarity}",
        )
    except Exception as exc:  # noqa: BLE001
        check("write_round_trip_completed", False, f"{type(exc).__name__}: {exc}")
    finally:
        _cleanup(doc_id, embedding_id)


def _cleanup(doc_id: str, embedding_id: str) -> None:
    section("8. Cleanup")
    try:
        lakebase.run_write("DELETE FROM weather_documents WHERE id = %s", (doc_id,))
        check("cleanup_delete_executed", True)
    except Exception as exc:  # noqa: BLE001
        check("cleanup_delete_executed", False, f"{type(exc).__name__}: {exc}")
        return

    try:
        remaining_docs = lakebase.run_query(
            "SELECT id FROM weather_documents WHERE id = %s", (doc_id,)
        )
        remaining_embeddings = lakebase.run_query(
            "SELECT id FROM weather_embeddings WHERE id = %s", (embedding_id,)
        )
        check(
            "cleanup_verified_gone",
            not remaining_docs and not remaining_embeddings,
            f"remaining_docs={remaining_docs} remaining_embeddings={remaining_embeddings}",
        )
    except Exception as exc:  # noqa: BLE001
        check("cleanup_verified_gone", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Also run the write/read round trip (sections 7-8): insert a "
        "throwaway document + embedding, verify cosine similarity, clean up.",
    )
    args = parser.parse_args()

    if not _check_target():
        print(
            "\nNo Lakebase target configured -- every check below would just "
            "fail with a connection error instead of saying anything useful. "
            "Set LAKEBASE_URL (see env.example), or PGHOST/PGUSER for the "
            "OAuth-token path, then re-run this script."
        )
        return _finish()

    _check_connect()
    _check_schema_bootstrap()
    _check_pgvector_extension()
    _check_columns()
    _check_hnsw_and_plan()

    if args.write:
        _run_write_round_trip()
    else:
        print("\n(sections 7-8, the write/read round trip, skipped -- pass --write to run them)")

    return _finish()


def _finish() -> int:
    section("Summary")
    print(f"{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
