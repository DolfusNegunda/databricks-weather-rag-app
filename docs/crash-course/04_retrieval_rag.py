# Databricks notebook source
# MAGIC %md
# MAGIC # Crash course 04: Retrieval and RAG
# MAGIC
# MAGIC Everything before this notebook built the pieces. This one runs the
# MAGIC actual query path -- the same SQL constant and the same summarize
# MAGIC function `app.py` uses, imported from `app.py` itself, not retyped -- and
# MAGIC then constructs a worked example of the failure mode retrieval can still
# MAGIC have even when every earlier stage worked correctly: a chunk boundary
# MAGIC that splits a hazard description from the instruction for it.
# MAGIC
# MAGIC **The live section (an actual search against real embedded data) needs a
# MAGIC Lakebase connection with synced and embedded documents.** The bad-chunking
# MAGIC demonstration at the end is self-contained and always runs.

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
# MAGIC ## Live: the actual retrieval query, imported from `app.py`
# MAGIC
# MAGIC `import app` runs `app.py`'s own startup (`_bootstrap()`, which calls
# MAGIC `ensure_weather_schema()`) -- safe with no database configured, because
# MAGIC that function is designed to capture its own failures rather than raise
# MAGIC (see `docs/crash-course/00-overview.md`'s map: this is the same discipline
# MAGIC that lets `/healthz` stay inspectable instead of crash-looping).

# COMMAND ----------

LIVE_RESULTS = None

try:
    import app as weather_app

    print(f"Imported app.py. Bootstrap state: {weather_app._bootstrap_result}")

    _query = "flash flood risk this weekend"
    _result = weather_app._run_search(_query, top_k=3, source_type=None)

    if _result.get("reason"):
        print(f"\n{_result['reason']}")
    else:
        LIVE_RESULTS = _result["results"]
        print(f"\nQuery: {_query!r} -- {_result['count']} result(s)")
        for row in LIVE_RESULTS:
            print(
                f"  [{row['source_type']:<10}] similarity={row['similarity']:.4f}  "
                f"{row['location']} -- {row['headline'] or row['event'] or '(no headline)'}"
            )
            print(f"    {row['chunk_text'][:140]}...")

except Exception as exc:  # noqa: BLE001 -- live section, must degrade not crash
    print(f"Skipping the live search -- {type(exc).__name__}: {exc}")
    print("The code above is the exact call app.py's /weather/search route makes internally.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Live: the RAG step, same function `app.py` uses
# MAGIC
# MAGIC `_summarize_results` builds one prompt from the retrieved chunks and asks
# MAGIC a Databricks Foundation Model endpoint to answer using **only** that
# MAGIC text. If no endpoint is configured/reachable in this workspace, it
# MAGIC degrades to `summary_error` -- by design, the same pattern used
# MAGIC throughout this project for an optional dependency that might not be
# MAGIC provisioned everywhere.

# COMMAND ----------

if LIVE_RESULTS:
    _summary = weather_app._summarize_results("flash flood risk this weekend", LIVE_RESULTS)
    if _summary.get("summary"):
        print("Model answer:")
        print(_summary["summary"])
    else:
        print(f"No summary: {_summary.get('summary_error')}")
        print("(Expected if FOUNDATION_MODEL_ENDPOINT isn't provisioned in this workspace.)")
else:
    print("Skipped -- no live results from the previous cell to summarize.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The failure mode retrieval alone can't fix: a split hazard/instruction
# MAGIC
# MAGIC This is why `weather_client.normalize_alert` joins `description` and
# MAGIC `instruction` into **one** `narrative_text` before chunking, instead of
# MAGIC treating them as two separate documents -- and why that choice reduces,
# MAGIC but does not eliminate, this exact problem.
# MAGIC
# MAGIC Constructed below the same deterministic way as `02_chunking.py`: the
# MAGIC hazard sentence sits at the very start, and the instruction is padded to
# MAGIC start at a **computed** offset -- the start of the third chunk window --
# MAGIC so it's guaranteed to land in a chunk that does not overlap the one
# MAGIC holding the hazard text, not just "probably" long enough to. Deliberately
# MAGIC built this way for clarity, not a real captured alert -- the real alert
# MAGIC measured during planning was 620 combined characters, comfortably under
# MAGIC one chunk -- but a long, detailed one can absolutely split like this.

# COMMAND ----------

CHUNK_SIZE = embedder.CHUNK_SIZE
CHUNK_OVERLAP = embedder.CHUNK_OVERLAP
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP

HAZARD_TEXT = (
    "A Flash Flood Warning is in effect. Radar indicates a slow-moving band of "
    "thunderstorms producing torrential rainfall rates of 2 to 3 inches per hour "
    "across the warned area, and this is a life-threatening situation for anyone "
    "caught in a low-lying area when the water rises. "
)
INSTRUCTION_TEXT = (
    "Turn around, don't drown -- do not attempt to cross flooded roadways on "
    "foot or in a vehicle, even if the water looks shallow or the road is "
    "familiar. Move immediately to higher ground if you are in a low-lying "
    "area, and stay away from storm drains, culverts, and normally-dry creek "
    "beds. If your vehicle becomes surrounded by rising water, abandon it and "
    "move to higher ground on foot if you can do so safely."
)
FILLER_SENTENCE = "Additional showers and thunderstorms are possible through the evening. "

# The start of the *third* chunk window (chunk 0 is [0, CHUNK_SIZE), chunk 1 is
# [STRIDE, STRIDE + CHUNK_SIZE)) -- padding the instruction to start exactly
# here guarantees it falls after chunk 1 ends, so no single chunk can contain
# both the hazard marker (position ~0) and the instruction marker.
_instruction_start = 2 * STRIDE
_padding_needed = _instruction_start - len(HAZARD_TEXT)
_filler_padding = (FILLER_SENTENCE * (_padding_needed // len(FILLER_SENTENCE) + 1))[:_padding_needed]

ALERT_NARRATIVE = HAZARD_TEXT + _filler_padding + INSTRUCTION_TEXT
assert ALERT_NARRATIVE[_instruction_start : _instruction_start + len(INSTRUCTION_TEXT)] == INSTRUCTION_TEXT

print(f"HAZARD_TEXT:      {len(HAZARD_TEXT)} characters, at offset 0")
print(f"Filler padding:   {len(_filler_padding)} characters")
print(f"INSTRUCTION_TEXT: {len(INSTRUCTION_TEXT)} characters, at offset {_instruction_start}")
print(f"Combined narrative_text: {len(ALERT_NARRATIVE)} characters (chunk_size={CHUNK_SIZE}, stride={STRIDE})")

# COMMAND ----------

alert_chunks = embedder.chunk_text(ALERT_NARRATIVE)
print(f"\n{len(alert_chunks)} chunk(s) produced from the combined narrative:")
for i, chunk in enumerate(alert_chunks):
    which = []
    if any(word in chunk for word in ("Flash Flood Warning", "torrential", "runoff")):
        which.append("hazard")
    if any(word in chunk for word in ("Turn around", "don't drown", "abandon it")):
        which.append("instruction")
    print(f"  chunk {i}: {len(chunk)} chars, contains: {', '.join(which) or 'neither marker'}")

hazard_chunks = {i for i, c in enumerate(alert_chunks) if "Flash Flood Warning" in c}
instruction_chunks = {i for i, c in enumerate(alert_chunks) if "Turn around" in c}
split_across_chunks = bool(hazard_chunks) and bool(instruction_chunks) and hazard_chunks.isdisjoint(instruction_chunks)

print(f"\nHazard text lands in chunk(s): {sorted(hazard_chunks)}")
print(f"Instruction text lands in chunk(s): {sorted(instruction_chunks)}")
print(f"Split into non-overlapping chunks: {split_across_chunks}")

assert split_across_chunks, (
    "expected the hazard and instruction markers in disjoint chunks by "
    "construction -- if this fails, the offset math above no longer "
    "guarantees the split (e.g. CHUNK_SIZE/CHUNK_OVERLAP changed)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why that split matters for the query "what should I do"
# MAGIC
# MAGIC If the hazard chunk and the instruction chunk are separate rows in
# MAGIC `weather_embeddings`, a query worded like "what should I do about flash
# MAGIC flooding" embeds closer to whichever chunk's *wording* it resembles more.
# MAGIC A query about *action* ("what should I do", "how do I stay safe") should
# MAGIC embed closer to the instruction chunk's "turn around", "move to higher
# MAGIC ground" language than to the hazard chunk's radar/rainfall-rate language --
# MAGIC meaning a low `top_k` (this project defaults to 5, same as the reference
# MAGIC pipeline) could plausibly return the hazard chunk explaining *what* is
# MAGIC happening without ever surfacing the instruction chunk explaining *what to
# MAGIC do about it*. A RAG summary built from only that one chunk would then
# MAGIC correctly describe the hazard while omitting the actual safety
# MAGIC instruction -- a factually-grounded-but-incomplete answer, which is a
# MAGIC harder failure to notice than an obviously wrong one.

# COMMAND ----------

try:
    _mock_query = "what should I do about flash flooding"
    _texts = [_mock_query] + alert_chunks
    _vectors = embedder.embed_texts(_texts)
    _query_vec, _chunk_vecs = _vectors[0], _vectors[1:]

    def _cosine(a, b):
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    print(f"Query: {_mock_query!r}\n")
    for i, vec in enumerate(_chunk_vecs):
        sim = _cosine(_query_vec, vec)
        tag = "hazard" if i in hazard_chunks else ("instruction" if i in instruction_chunks else "?")
        print(f"  chunk {i} ({tag}): cosine similarity to query = {sim:.4f}")

    if len(alert_chunks) > 1:
        print(
            "\nWith top_k=1 here, only the highest-scoring chunk above would be "
            "retrieved -- if that happens to be the hazard chunk, the instruction "
            "chunk (and everything in it) is invisible to this query, not because "
            "retrieval is broken, but because it can only return what got asked for."
        )
except Exception as exc:  # noqa: BLE001 -- optional, needs a real model
    print(f"Skipping the real-embedding similarity comparison -- {type(exc).__name__}: {exc}")
    print(
        "The structural point still holds without a model: the hazard and "
        "instruction text are in different chunk_text rows above, so any "
        "top_k that returns fewer than all of this alert's chunks can return "
        "one without the other."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - Retrieval quality is not just "did the model produce good vectors" --
# MAGIC   it's also "did chunking keep related facts together." Both matter.
# MAGIC - Joining `description` + `instruction` into one `narrative_text` before
# MAGIC   chunking (see `weather_client.normalize_alert`) is a real mitigation,
# MAGIC   not a complete fix -- it removes the guaranteed split for *short*
# MAGIC   alerts (they fit in one chunk together) without removing the
# MAGIC   *possibility* for long ones.
# MAGIC - A grounded RAG answer can still be misleadingly incomplete if
# MAGIC   retrieval didn't surface every chunk that mattered -- "the model didn't
# MAGIC   invent anything" and "the answer is complete" are different claims.
# MAGIC
# MAGIC Next: `05_hnsw_benchmark.py` -- what the HNSW index actually costs and
# MAGIC saves, measured, not assumed.
