# Databricks notebook source
# MAGIC %md
# MAGIC # Crash course 03: pgvector
# MAGIC
# MAGIC `01_embeddings.py` made vectors; `02_chunking.py` made more of them, from
# MAGIC longer text. This notebook is about the database side: `sql/
# MAGIC 02_weather_embeddings.sql`'s `VECTOR(384)` column, the three distance
# MAGIC operators pgvector adds to Postgres, and why `weather_embeddings` has an
# MAGIC HNSW index at all.
# MAGIC
# MAGIC **The live sections below (operator behavior against real rows, `EXPLAIN
# MAGIC ANALYZE` with/without the index) need a real Lakebase connection with at
# MAGIC least a few embedded rows.** Everything conceptual runs regardless.

# COMMAND ----------

import sys
from pathlib import Path

if "__file__" in globals():
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import embedder

# COMMAND ----------

# MAGIC %md
# MAGIC ## The `vector` type
# MAGIC
# MAGIC `sql/02_weather_embeddings.sql` declares:
# MAGIC
# MAGIC ```sql
# MAGIC embedding VECTOR(384) NOT NULL
# MAGIC ```
# MAGIC
# MAGIC `pgvector`'s `vector` type stores a fixed-length array of single-precision
# MAGIC floats directly as a Postgres column -- not JSON, not a `float[]` array
# MAGIC with no length enforcement. The `(384)` is enforced: inserting a vector of
# MAGIC any other length is a hard error, which is exactly what would catch a
# MAGIC model swap that changed `EMBEDDING_DIM` without updating this column (see
# MAGIC `scripts/check_connection.py`'s dimension check, which exists specifically
# MAGIC to catch that mismatch before it becomes a confusing insert failure).
# MAGIC
# MAGIC A value going into that column, from Python, is the bracketed string
# MAGIC `embedder.to_vector_literal()` builds -- `"[0.1,-0.2,...]"` -- bound as an
# MAGIC ordinary `%s` string parameter with an explicit `::vector` cast in the SQL
# MAGIC text. Nothing here relies on psycopg2 having a special adapter for Python
# MAGIC lists; the cast is what does the real work, on the Postgres side.

# COMMAND ----------

_sample = embedder.to_vector_literal([0.1, -0.2, 0.3])
print(f"to_vector_literal([0.1, -0.2, 0.3]) = {_sample!r}")
print(f"Bound in SQL as:  embedding = {_sample}::vector")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Three distance operators, and why this project uses one of them
# MAGIC
# MAGIC | Operator | Name | Formula | When it's the right choice |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `<->` | Euclidean (L2) distance | `sqrt(sum((a_i - b_i)^2))` | Vectors where absolute position in space matters, not just direction -- image embeddings are a common example. |
# MAGIC | `<#>` | Negative inner product | `-sum(a_i * b_i)` | Cheapest to compute (no square roots, no division); correct ranking specifically when every vector is already unit-length. |
# MAGIC | `<=>` | Cosine distance | `1 - (dot(a,b) / (||a|| * ||b||))` | Text embeddings in general -- ranks by the *angle* between vectors, ignoring magnitude differences that don't reflect meaning. |
# MAGIC
# MAGIC This project uses `<=>` everywhere (`app.py`'s `_WEATHER_SEARCH_SQL`,
# MAGIC `scripts/check_connection.py`'s semantic round trip) because it's the
# MAGIC conventional choice for sentence-transformer output, and it's what
# MAGIC `01_embeddings.py` already proved equals `1 - cosine_similarity`.
# MAGIC
# MAGIC **A fact worth knowing, not a contradiction:** `all-MiniLM-L6-v2`'s output
# MAGIC is *already* very close to unit length in practice. For genuinely
# MAGIC unit-length vectors, `<=>` and `<->` produce the *same ranking* (cosine
# MAGIC distance becomes a monotonic function of Euclidean distance when both
# MAGIC vectors have length 1) -- so switching this project to `<->` would likely
# MAGIC return search results in the same order. That's a property of this
# MAGIC specific model's output, not a property of cosine distance in general;
# MAGIC `<=>` stays the conventionally correct choice because it's correct
# MAGIC regardless of whether a given model happens to normalize its output, and
# MAGIC it's what the rest of this bootcamp's material standardizes on.

# COMMAND ----------

import math


def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 1.0 - dot / (norm_a * norm_b)


def negative_inner_product(a, b):
    return -sum(x * y for x, y in zip(a, b))


# Two unit vectors, 45 degrees apart -- and one NOT unit-length, same direction
# as the first, to show <=> ignoring magnitude while <-> does not.
UNIT_A = [1.0, 0.0]
UNIT_B = [0.7071067811865476, 0.7071067811865476]  # 45 degrees from UNIT_A
SCALED_A = [5.0, 0.0]  # same direction as UNIT_A, 5x the length

print(f"l2_distance(UNIT_A, UNIT_B)       = {l2_distance(UNIT_A, UNIT_B):.4f}")
print(f"cosine_distance(UNIT_A, UNIT_B)   = {cosine_distance(UNIT_A, UNIT_B):.4f}")
print()
print(f"l2_distance(UNIT_A, SCALED_A)     = {l2_distance(UNIT_A, SCALED_A):.4f}   <- large: different magnitude")
print(f"cosine_distance(UNIT_A, SCALED_A) = {cosine_distance(UNIT_A, SCALED_A):.4f}   <- zero: identical direction")
print()
print(
    "SCALED_A points in exactly the same direction as UNIT_A, just 5x longer. "
    "Cosine distance calls them identical (0.0); Euclidean distance does not. "
    "For text embeddings, direction is what encodes meaning -- that's the "
    "argument for <=> in one number."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## HNSW vs IVFFlat vs a sequential scan
# MAGIC
# MAGIC With no index, `ORDER BY embedding <=> query LIMIT k` is a **sequential
# MAGIC scan**: compute the distance to *every* row, sort, take the top k. Exact,
# MAGIC and fine up to some tens of thousands of rows -- past that it's the first
# MAGIC thing that gets slow as this project's corpus grows.
# MAGIC
# MAGIC - **IVFFlat** ("inverted file, flat") clusters vectors into `lists` groups
# MAGIC   at index-build time; a query only searches the nearest few clusters
# MAGIC   instead of every row. Fast to build, needs the table to already have
# MAGIC   representative data before building (clustering an empty table
# MAGIC   produces useless clusters), and needs periodic rebuilding as data grows.
# MAGIC - **HNSW** ("hierarchical navigable small world") builds a multi-layer
# MAGIC   graph where each node connects to its approximate nearest neighbors;
# MAGIC   a query walks the graph from a coarse top layer down to a fine bottom
# MAGIC   layer. Slower to build than IVFFlat, but doesn't need pre-existing data
# MAGIC   to be useful and generally gives better recall for the same query speed.
# MAGIC   This is what `sql/02_weather_embeddings.sql` uses:
# MAGIC
# MAGIC   ```sql
# MAGIC   CREATE INDEX idx_weather_embeddings_hnsw
# MAGIC       ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
# MAGIC   ```
# MAGIC
# MAGIC Both are **approximate** -- neither is guaranteed to return the exact
# MAGIC top-k nearest rows, trading a small, usually-negligible chance of missing
# MAGIC the true nearest neighbor for a large speedup. `05_hnsw_benchmark.py`
# MAGIC measures that trade-off directly, on this project's own data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Live: does the query actually use the index?
# MAGIC
# MAGIC Needs a real Lakebase connection with at least a few rows in
# MAGIC `weather_embeddings`. Runs `EXPLAIN` (not `EXPLAIN ANALYZE` here --
# MAGIC that's `05_hnsw_benchmark.py`'s job) with the HNSW index available, then
# MAGIC again with `SET LOCAL enable_indexscan = off` forcing a sequential scan,
# MAGIC inside a transaction that's always rolled back so the setting never
# MAGIC leaks past this cell.

# COMMAND ----------

try:
    import lakebase

    _probe_vector = embedder.to_vector_literal([0.05] * embedder.EMBEDDING_DIM)

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM weather_embeddings")
            _row_count = cur.fetchone()["n"]
        conn.rollback()

    if _row_count == 0:
        print("weather_embeddings is empty -- sync and embed some documents first. Skipping the live EXPLAIN.")
    else:
        print(f"weather_embeddings has {_row_count} row(s).")
        if _row_count < 100:
            print(
                f"NOTE: with only {_row_count} row(s), Postgres's own cost estimator can "
                "reasonably prefer a sequential scan over the index either way -- the "
                "comparison below is illustrative, not a real performance signal yet."
            )

        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "EXPLAIN SELECT id FROM weather_embeddings ORDER BY embedding <=> %s::vector LIMIT 5",
                    (_probe_vector,),
                )
                _with_index_plan = "\n".join(str(list(r.values())[0]) for r in cur.fetchall())
            conn.rollback()

        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
                cur.execute(
                    "EXPLAIN SELECT id FROM weather_embeddings ORDER BY embedding <=> %s::vector LIMIT 5",
                    (_probe_vector,),
                )
                _forced_seq_plan = "\n".join(str(list(r.values())[0]) for r in cur.fetchall())
                cur.execute("ROLLBACK")

        print("\n--- plan, index available ---")
        print(_with_index_plan)
        print(f"\n  -> uses an index scan: {'index scan' in _with_index_plan.lower()}")

        print("\n--- plan, index scans disabled (forced sequential scan) ---")
        print(_forced_seq_plan)
        print(f"\n  -> uses a sequential scan: {'seq scan' in _forced_seq_plan.lower()}")

except Exception as exc:  # noqa: BLE001 -- optional live section, must degrade not crash
    print(f"Skipping the live EXPLAIN comparison -- no usable Lakebase connection ({type(exc).__name__}: {exc}).")
    print("The code above is exactly what would run; see 05_hnsw_benchmark.py for the timed version.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - `VECTOR(384)` is a real, length-enforced Postgres column type, not a
# MAGIC   convention layered on top of `float[]` or JSON.
# MAGIC - `<=>` (cosine), `<->` (Euclidean), `<#>` (negative inner product) rank
# MAGIC   rows differently in general; this project standardizes on `<=>`.
# MAGIC - HNSW and IVFFlat both trade a small, usually-negligible accuracy loss
# MAGIC   for a large speedup over a sequential scan; HNSW is what this project's
# MAGIC   schema actually builds.
# MAGIC - `EXPLAIN` shows whether the planner is actually using that index, which
# MAGIC   is the only way to know for certain -- reasoning about it in the
# MAGIC   abstract is not the same as checking.
# MAGIC
# MAGIC Next: `04_retrieval_rag.py` -- the full query-to-answer path, including
# MAGIC what happens when retrieval hands a language model an incomplete chunk.
