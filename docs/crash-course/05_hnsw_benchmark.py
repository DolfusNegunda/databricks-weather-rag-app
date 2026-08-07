# Databricks notebook source
# MAGIC %md
# MAGIC # Crash course 05: HNSW benchmark
# MAGIC
# MAGIC `03_pgvector.py` explained HNSW as a latency/accuracy trade-off in the
# MAGIC abstract. This notebook measures it, on this project's own
# MAGIC `weather_embeddings` table -- both **latency** (how long a query takes,
# MAGIC with the index vs. a forced sequential scan, and at a couple of
# MAGIC `hnsw.ef_search` settings) and **recall** (whether the approximate search
# MAGIC actually finds the same rows an exact search would).
# MAGIC
# MAGIC This also satisfies the project's stretch goal: "add a
# MAGIC `CREATE INDEX ... USING hnsw` benchmark comparing query latency with vs.
# MAGIC without the index."
# MAGIC
# MAGIC **Needs a real Lakebase connection with embedded rows to produce real
# MAGIC numbers.** With no connection, this notebook explains the methodology and
# MAGIC prints the exact code that would run, clearly labeled as unexecuted.

# COMMAND ----------

import statistics
import sys
import time
from pathlib import Path

if "__file__" in globals():
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import embedder

REPEATS = 20
WARMUP_RUNS = 1
EF_SEARCH_VALUES = (40, 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Methodology
# MAGIC
# MAGIC For a fixed probe vector and `LIMIT 20`:
# MAGIC
# MAGIC 1. **Latency, with the HNSW index** (the table's normal state): run the
# MAGIC    query `REPEATS` times, discard the first (`WARMUP_RUNS`) as cache
# MAGIC    warm-up, report min/median/max of the rest.
# MAGIC 2. **Latency, forced sequential scan**: same query, same probe vector,
# MAGIC    inside a transaction with `SET LOCAL enable_indexscan = off` and
# MAGIC    `enable_bitmapscan = off`, always rolled back afterward so the setting
# MAGIC    never leaks into any other session.
# MAGIC 3. **Latency at different `hnsw.ef_search` values** -- this parameter
# MAGIC    controls how many candidate nodes the graph search keeps at each
# MAGIC    layer; higher means slower but more likely to find the true nearest
# MAGIC    neighbors. Tried at two values inside the same rolled-back-transaction
# MAGIC    pattern via `SET LOCAL hnsw.ef_search = ...`.
# MAGIC 4. **Recall@20**: run the *exact* top-20 (forced sequential scan) and the
# MAGIC    *approximate* top-20 (HNSW) for the same probe vector, and count how
# MAGIC    many of the exact top-20 IDs also appear in the approximate top-20.
# MAGIC    This is the number that actually tells you what the index costs you
# MAGIC    in accuracy -- latency alone never shows that; a fast approximate
# MAGIC    search that returns the wrong rows is still fast.

# COMMAND ----------


def _timed_query(conn_factory, sql, params, repeats=REPEATS, warmup=WARMUP_RUNS):
    """Run `sql` `repeats` times through a fresh connection each time (so pool
    connection-acquisition cost is part of every measurement equally, rather
    than only the first), discard the first `warmup` runs, return the timings
    in seconds for the rest."""
    timings = []
    for i in range(repeats):
        start = time.perf_counter()
        with conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cur.fetchall()
            conn.rollback()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            timings.append(elapsed)
    return timings


def _report(label, timings):
    if not timings:
        print(f"{label}: no timings collected")
        return
    print(
        f"{label}: min={min(timings)*1000:.2f}ms  "
        f"median={statistics.median(timings)*1000:.2f}ms  "
        f"max={max(timings)*1000:.2f}ms  (n={len(timings)})"
    )


# COMMAND ----------

try:
    import lakebase

    with lakebase.get_connection() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute("SELECT count(*) AS n FROM weather_embeddings")
            ROW_COUNT = _cur.fetchone()["n"]
        _conn.rollback()

    if ROW_COUNT == 0:
        raise RuntimeError("weather_embeddings is empty -- sync and embed some documents first")

    print(f"weather_embeddings has {ROW_COUNT} row(s).")
    if ROW_COUNT < 200:
        print(
            f"NOTE: {ROW_COUNT} rows is small. HNSW's advantage over a sequential "
            "scan grows with table size -- at this scale the two may be close, "
            "or the planner may even prefer the sequential scan on its own. The "
            "numbers below are still real measurements, just not yet at a scale "
            "where the index's benefit is dramatic."
        )

    PROBE_VECTOR = embedder.to_vector_literal([0.05] * embedder.EMBEDDING_DIM)
    TOP_K_SQL = "SELECT id FROM weather_embeddings ORDER BY embedding <=> %s::vector LIMIT 20"

    # -- 1. latency, with the index (the table's normal state) --------------
    print("\n--- Latency: HNSW index available (normal state) ---")
    _with_index_timings = _timed_query(lakebase.get_connection, TOP_K_SQL, (PROBE_VECTOR,))
    _report("with HNSW index", _with_index_timings)

    # -- 2. latency, forced sequential scan ----------------------------------
    print("\n--- Latency: forced sequential scan (enable_indexscan = off) ---")

    def _seq_scan_connection():
        from contextlib import contextmanager

        @contextmanager
        def _wrapped():
            with lakebase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute("SET LOCAL enable_indexscan = off")
                    cur.execute("SET LOCAL enable_bitmapscan = off")
                yield conn
                with conn.cursor() as cur:
                    cur.execute("ROLLBACK")

        return _wrapped()

    _seq_scan_timings = _timed_query(_seq_scan_connection, TOP_K_SQL, (PROBE_VECTOR,))
    _report("forced sequential scan", _seq_scan_timings)

    if _with_index_timings and _seq_scan_timings:
        _speedup = statistics.median(_seq_scan_timings) / statistics.median(_with_index_timings)
        print(f"\nHNSW median speedup over sequential scan: {_speedup:.2f}x")

    # -- 3. latency at different ef_search values ----------------------------
    print("\n--- Latency at different hnsw.ef_search values ---")

    def _ef_search_connection(ef_value):
        from contextlib import contextmanager

        @contextmanager
        def _wrapped():
            with lakebase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_value,))
                yield conn
                with conn.cursor() as cur:
                    cur.execute("ROLLBACK")

        return _wrapped()

    for ef_value in EF_SEARCH_VALUES:
        _timings = _timed_query(lambda ef=ef_value: _ef_search_connection(ef), TOP_K_SQL, (PROBE_VECTOR,))
        _report(f"ef_search={ef_value}", _timings)

    print(
        "\nhnsw.ef_search trades recall for speed: a higher value makes the "
        "graph search keep more candidates at each layer, which costs time but "
        "makes it less likely to miss a true nearest neighbor. This benchmark "
        "measures the time side of that trade; recall is measured directly next."
    )

    # -- 4. recall@20 ---------------------------------------------------------
    print("\n--- Recall@20: does HNSW find the same rows an exact search would? ---")

    with lakebase.get_connection() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(TOP_K_SQL, (PROBE_VECTOR,))
            _approx_ids = [row["id"] for row in _cur.fetchall()]
        _conn.rollback()

    with lakebase.get_connection() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute("BEGIN")
            _cur.execute("SET LOCAL enable_indexscan = off")
            _cur.execute("SET LOCAL enable_bitmapscan = off")
            _cur.execute(TOP_K_SQL, (PROBE_VECTOR,))
            _exact_ids = [row["id"] for row in _cur.fetchall()]
            _cur.execute("ROLLBACK")

    _overlap = len(set(_approx_ids) & set(_exact_ids))
    _recall = _overlap / len(_exact_ids) if _exact_ids else None

    print(f"Exact top-20 IDs:       {len(_exact_ids)} row(s)")
    print(f"Approximate top-20 IDs: {len(_approx_ids)} row(s)")
    print(f"Overlap:                {_overlap} row(s)")
    if _recall is not None:
        print(f"Recall@20:              {_recall:.0%}")
        if _recall == 1.0:
            print("HNSW found every one of the exact nearest neighbors for this probe vector.")
        else:
            print(
                f"HNSW missed {len(_exact_ids) - _overlap} of the exact nearest "
                "neighbor(s) for this probe vector -- this is the real cost the "
                "latency numbers above don't show on their own."
            )

except Exception as exc:  # noqa: BLE001 -- must degrade, not crash, with no live DB
    print(f"Skipping the live benchmark -- {type(exc).__name__}: {exc}")
    print(
        "\nWith a real connection and embedded rows, this notebook would: time "
        "20 repeated queries with the HNSW index available, time 20 more with "
        "enable_indexscan/enable_bitmapscan forced off inside a rolled-back "
        "transaction, report min/median/max latency for both plus the median "
        "speedup, repeat the timing at hnsw.ef_search values of 40 and 200, and "
        "finally compare the exact top-20 (sequential scan) against the "
        "approximate top-20 (HNSW) for the same probe vector to report "
        "Recall@20 -- the fraction of the true nearest neighbors the "
        "approximate index search actually found."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - "The index is faster" is a latency claim; "the index is worth it" also
# MAGIC needs a recall number, because an approximate search can be fast and
# MAGIC still miss the actual nearest neighbors -- this notebook measures both.
# MAGIC - `hnsw.ef_search` is the dial: higher costs latency, buys recall. What
# MAGIC counts as an acceptable trade is a product decision (would a user notice
# MAGIC or care if result #18 out of 20 were slightly wrong?), not something the
# MAGIC database can decide for you.
# MAGIC - At small table sizes the sequential scan can be competitive or even
# MAGIC faster -- the index's advantage is asymptotic, and this project's own
# MAGIC table may not be past that crossover point yet. Re-run this notebook
# MAGIC after syncing and embedding more locations to see the gap change.
# MAGIC
# MAGIC That's the crash course. `00-overview.md` has the full map back to every
# MAGIC file in this repository if you want to trace any of these five notebooks'
# MAGIC claims back to the actual running application.
