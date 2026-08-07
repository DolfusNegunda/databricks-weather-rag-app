# Databricks notebook source
# MAGIC %md
# MAGIC # Crash course 02: Chunking
# MAGIC
# MAGIC `01_embeddings.py` proved a model turns text into a point in space. This
# MAGIC notebook answers the question that raises immediately: **what if the text
# MAGIC is 9,275 characters long, like a real Area Forecast Discussion?**
# MAGIC `all-MiniLM-L6-v2` still returns exactly one 384-number vector no matter
# MAGIC how much text you hand it -- that vector becomes a blurred average of
# MAGIC every sub-topic in the input. Chunking is how this project avoids asking
# MAGIC one vector to represent a synoptic overview, a flash-flood-risk paragraph,
# MAGIC and an overnight-low-temperature paragraph all at once.
# MAGIC
# MAGIC What this notebook does, in order:
# MAGIC 1. Tries to fetch one real discussion's `narrative_text` from Lakebase, as
# MAGIC    context -- falls back to a clearly-labeled realistic sample if there's
# MAGIC    no live connection or nothing synced yet.
# MAGIC 2. Runs the project's real `embedder.chunk_text` on a **deterministically
# MAGIC    constructed** ~2000-character passage and shows the actual chunk
# MAGIC    boundaries and lengths.
# MAGIC 3. Prints the literal overlapping substring shared by two consecutive
# MAGIC    chunks, so "100 characters of overlap" stops being an abstract number.
# MAGIC 4. Proves, by construction and by assertion (not by eyeballing), that a
# MAGIC    sentence positioned right at the chunk boundary survives **whole** in
# MAGIC    one chunk when `chunk_overlap=100`, but is split across **both**
# MAGIC    chunks -- whole in neither -- when `chunk_overlap=0`.
# MAGIC
# MAGIC **This file runs with no live database.** Step 1's live fetch is
# MAGIC optional context; the deterministic demonstration in steps 2-4 uses
# MAGIC constructed text specifically so the result doesn't depend on what (if
# MAGIC anything) happens to be synced when you run this.

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
# MAGIC ## Step 1 (optional context): what does a real discussion look like?
# MAGIC
# MAGIC Not required for the rest of this notebook -- just grounding. A real one
# MAGIC fetched during planning ran 9,275 characters; here's whatever is actually
# MAGIC in your database right now, if anything.

# COMMAND ----------

LIVE_SAMPLE = None
LIVE_SAMPLE_ERROR = None

try:
    import lakebase

    _rows = lakebase.run_query(
        "SELECT location, length(narrative_text) AS len, narrative_text "
        "FROM weather_documents WHERE source_type = 'discussion' "
        "ORDER BY synced_at DESC LIMIT 1"
    )
    if _rows:
        LIVE_SAMPLE = _rows[0]
except Exception as exc:  # noqa: BLE001 -- optional context, must not block the rest
    LIVE_SAMPLE_ERROR = f"{type(exc).__name__}: {exc}"

if LIVE_SAMPLE:
    print(f"Found a real discussion for {LIVE_SAMPLE['location']}: {LIVE_SAMPLE['len']} characters.")
    print("First 300 characters:")
    print(LIVE_SAMPLE["narrative_text"][:300])
else:
    reason = LIVE_SAMPLE_ERROR or "no discussion documents have been synced yet"
    print(f"No live discussion available ({reason}) -- skipping, using constructed text below instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: a deterministically constructed passage
# MAGIC
# MAGIC Rather than hope a real document happens to put an interesting sentence
# MAGIC exactly at a chunk boundary, this builds one: filler sentences, then one
# MAGIC **target sentence** whose starting character offset is computed to land
# MAGIC 20 characters before `CHUNK_SIZE` -- guaranteed to straddle the boundary
# MAGIC no matter what `CHUNK_SIZE` is currently set to.

# COMMAND ----------

CHUNK_SIZE = embedder.CHUNK_SIZE
CHUNK_OVERLAP = embedder.CHUNK_OVERLAP

FILLER_SENTENCE = "Skies will remain partly cloudy through the afternoon with light winds. "
TARGET_SENTENCE = (
    "Heavy rainfall streaming north of the stalled front will raise the risk "
    "of flash flooding across low-lying and poor-drainage areas overnight."
)

_target_start = CHUNK_SIZE - 20
_filler_before = (FILLER_SENTENCE * (_target_start // len(FILLER_SENTENCE) + 1))[:_target_start]
_filler_after = FILLER_SENTENCE * 6

TEXT = _filler_before + TARGET_SENTENCE + " " + _filler_after
_target_end = _target_start + len(TARGET_SENTENCE)

print(f"CHUNK_SIZE = {CHUNK_SIZE}, CHUNK_OVERLAP = {CHUNK_OVERLAP}")
print(f"Constructed passage length: {len(TEXT)} characters")
print(f"TARGET_SENTENCE spans characters [{_target_start}, {_target_end}) -- straddling position {CHUNK_SIZE}")
assert TEXT[_target_start:_target_end] == TARGET_SENTENCE, "construction bug: offsets don't line up"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: chunk it with this project's real defaults

# COMMAND ----------

chunks_with_overlap = embedder.chunk_text(TEXT, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(f"{len(chunks_with_overlap)} chunk(s) produced:")
for i, chunk in enumerate(chunks_with_overlap):
    print(f"  chunk {i}: {len(chunk)} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: the literal overlap between chunk 0 and chunk 1
# MAGIC
# MAGIC `CHUNK_OVERLAP` characters of chunk 0's tail should equal chunk 1's head,
# MAGIC exactly -- not "similar text", the identical substring.

# COMMAND ----------

if len(chunks_with_overlap) >= 2:
    tail_of_chunk_0 = chunks_with_overlap[0][-CHUNK_OVERLAP:]
    head_of_chunk_1 = chunks_with_overlap[1][:CHUNK_OVERLAP]
    print(f"Last {CHUNK_OVERLAP} characters of chunk 0:\n  {tail_of_chunk_0!r}")
    print(f"First {CHUNK_OVERLAP} characters of chunk 1:\n  {head_of_chunk_1!r}")
    assert tail_of_chunk_0 == head_of_chunk_1, "overlap should be an exact substring match"
    print("\nMATCH CONFIRMED -- these are the same characters, appearing in both chunks.")
else:
    print("Only one chunk produced -- not enough text to show an overlap boundary here.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: the point of this whole notebook
# MAGIC
# MAGIC Does `TARGET_SENTENCE` -- the one deliberately placed at the boundary --
# MAGIC survive **whole** inside some chunk? With `chunk_overlap=100` (the
# MAGIC project default), it should. Re-chunking the *same text* with
# MAGIC `chunk_overlap=0` should show the opposite: the sentence cut in half,
# MAGIC whole in **no** chunk.

# COMMAND ----------

chunks_no_overlap = embedder.chunk_text(TEXT, chunk_size=CHUNK_SIZE, chunk_overlap=0)

whole_with_overlap = any(TARGET_SENTENCE in chunk for chunk in chunks_with_overlap)
whole_without_overlap = any(TARGET_SENTENCE in chunk for chunk in chunks_no_overlap)

print(f"TARGET_SENTENCE appears whole in some chunk, chunk_overlap={CHUNK_OVERLAP}: {whole_with_overlap}")
print(f"TARGET_SENTENCE appears whole in some chunk, chunk_overlap=0:            {whole_without_overlap}")

assert whole_with_overlap, (
    "expected the overlap to preserve the target sentence whole -- if this "
    "fails, the offsets in Step 2 no longer straddle the boundary correctly"
)
assert not whole_without_overlap, (
    "expected zero overlap to split the target sentence -- if this fails, "
    "either chunk_text changed behavior or the offsets no longer straddle"
)

print(
    "\nCONFIRMED: the exact same sentence, in the exact same document, is "
    "recoverable whole with overlap and is NOT recoverable whole without it. "
    "That is the entire reason CHUNK_OVERLAP exists -- a fact split across a "
    "chunk boundary should still be findable from whichever side of the "
    "boundary a search query's wording happens to resemble."
)

# Show precisely how it split, computed directly from the known offsets
# rather than guessed at with a substring heuristic.
_split_at = CHUNK_SIZE - _target_start  # characters of TARGET_SENTENCE before the cut
print(
    f"\nWith chunk_overlap=0, the cut lands {_split_at} characters into "
    f"TARGET_SENTENCE:\n"
    f"  chunk 0 ends with:   ...{TARGET_SENTENCE[:_split_at]!r}\n"
    f"  chunk 1 begins with: {TARGET_SENTENCE[_split_at:_split_at + 40]!r}..."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - `chunk_text` slides a `CHUNK_SIZE`-character window forward by
# MAGIC   `CHUNK_SIZE - CHUNK_OVERLAP` characters each step -- the overlap is
# MAGIC   literally re-including the last `CHUNK_OVERLAP` characters of the
# MAGIC   previous chunk at the start of the next one.
# MAGIC - A short alert (measured ~620 characters combined description +
# MAGIC   instruction) almost always stays exactly one chunk at `CHUNK_SIZE=800`
# MAGIC   -- chunking is doing nothing there, and that's fine. A 9,275-character
# MAGIC   discussion is where it earns its keep.
# MAGIC - Overlap is not a nice-to-have: this notebook just proved, with an
# MAGIC   assertion rather than a claim, that a sentence sitting on a chunk
# MAGIC   boundary can be **provably unrecoverable as a whole unit** without it.
# MAGIC
# MAGIC Next: `03_pgvector.py` -- what actually happens once these chunks are
# MAGIC embedded and stored, and how Postgres finds the nearest ones fast.
