# Databricks notebook source
# MAGIC %md
# MAGIC # Crash course 01: Embeddings
# MAGIC
# MAGIC This notebook answers one question with real numbers instead of a
# MAGIC diagram: **what actually comes out of `embedder.embed_texts()`, and why
# MAGIC does it make "search by meaning" possible?**
# MAGIC
# MAGIC What this notebook does, in order:
# MAGIC 1. Loads `sentence-transformers/all-MiniLM-L6-v2` via `embedder.get_model()`
# MAGIC    and encodes 7 hand-picked weather-domain sentences.
# MAGIC 2. Prints the shape and the first 8 numbers of one real vector, so "a
# MAGIC    384-float vector" stops being an abstraction.
# MAGIC 3. Writes a tiny `cosine(a, b)` helper by hand (`dot(a, b) / (norm(a) *
# MAGIC    norm(b))`) and uses it -- not a library function -- to build the full
# MAGIC    pairwise similarity matrix across all 7 sentences.
# MAGIC 4. Shows that two very differently worded flood-warning sentences score
# MAGIC    far more similar to each other than either scores to an unrelated
# MAGIC    sentence -- the entire intuition behind semantic search, in one table.
# MAGIC 5. Proves, independently, that `1 - cosine_distance(a, b) == cosine(a, b)`
# MAGIC    -- the identity the real SQL in this project (`1 - (e.embedding <=>
# MAGIC    %s::vector) AS similarity` in `app.py`) depends on.
# MAGIC
# MAGIC **This file runs with no live database and does not require
# MAGIC `sentence-transformers`/`torch` to be installed.** If either is missing,
# MAGIC every step below still runs, using a clearly labeled, hand-built
# MAGIC substitute vector instead of a real model output -- see the "FALLBACK"
# MAGIC banners if that happens on your machine.

# COMMAND ----------

import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np

# `python docs/crash-course/01_embeddings.py` puts docs/crash-course/ on
# sys.path, not the repo root where embedder.py lives -- same fix used by
# notebooks/ingest_weather_embeddings.py. `__file__` isn't defined when
# Databricks runs this as real notebook cells, where the repo root is
# already importable anyway, hence the guard.
if "__file__" in globals():
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import embedder

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: seven sentences, chosen to make one point each
# MAGIC
# MAGIC Two of these are paraphrases of the *same* flood warning, deliberately
# MAGIC worded with almost no words in common -- that pair is the whole
# MAGIC experiment. The rest give the similarity matrix below some texture: a
# MAGIC calm forecast, a heat advisory, a beach hazard, and two sentences that
# MAGIC have nothing to do with weather at all.

# COMMAND ----------

SENTENCES = {
    "flood_a": "Flash flood warning issued for Travis County; seek higher ground immediately and avoid low-lying roads.",
    "flood_b": "Water levels along area creeks are rising fast tonight; move to elevated terrain right now and stay off roadways that dip below grade.",
    "sunny": "Expect mostly sunny skies today with a high near 84 degrees and a light south wind.",
    "heat": "A heat advisory is in effect this afternoon; limit time outdoors and drink plenty of water.",
    "rip_current": "Rip current risk remains high along area beaches through Sunday evening.",
    "unrelated_it": "Please open a ticket with IT to have your work computer's password reset.",
    "unrelated_wifi": "The building's WiFi network will be offline Tuesday night for scheduled maintenance.",
}

LABELS = list(SENTENCES.keys())
TEXTS = list(SENTENCES.values())

print(f"{len(TEXTS)} sentences loaded:")
for label, text in SENTENCES.items():
    print(f"  {label:>15}: {text}")

print()
print("Notice flood_a and flood_b share almost no literal words -- 'and' is")
print("about the only one -- despite describing the same hazard. That's on")
print("purpose: literal word overlap is exactly what semantic embeddings do")
print("NOT need in order to recognize these as related.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: encode them
# MAGIC
# MAGIC `embedder.get_model()` lazily loads the real `sentence-transformers`
# MAGIC model. If `sentence-transformers`/`torch` aren't installed, it raises
# MAGIC `RuntimeError` (that's its documented contract) -- caught here and
# MAGIC replaced with a hand-built fallback so this cell never crashes.

# COMMAND ----------

USING_REAL_MODEL = True
FALLBACK_REASON = None

try:
    _model = embedder.get_model()
    vectors = np.asarray(_model.encode(TEXTS))
except Exception as exc:  # noqa: BLE001 -- teaching notebook, must degrade not crash
    USING_REAL_MODEL = False
    FALLBACK_REASON = f"{type(exc).__name__}: {exc}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### The fallback, if you're seeing it
# MAGIC
# MAGIC This is **not** a neural embedding. It's a hand-built stand-in so this
# MAGIC notebook still runs without `torch` installed:
# MAGIC 1. Lower-case the sentence, split into words, drop a short stopword list.
# MAGIC 2. Hash each remaining word into one of 384 dimensions (a classic
# MAGIC    technique called *feature hashing* -- the same idea behind
# MAGIC    scikit-learn's `HashingVectorizer`), with a hash-derived `+1`/`-1`
# MAGIC    sign so unrelated words don't all pile up positively in the same
# MAGIC    place.
# MAGIC 3. Also check the sentence against a **small, hand-written list of
# MAGIC    topic phrases** (`_CONCEPT_GROUPS` below) -- e.g. both "flash flood"
# MAGIC    and "water levels" are manually mapped to the same
# MAGIC    `CONCEPT_FLOOD_HAZARD` token, so a sentence containing either one
# MAGIC    gets a matching hashed dimension bumped.
# MAGIC 4. L2-normalize the result to length 1 (matching a real fact about the
# MAGIC    actual model, covered in `03_pgvector.py`: `all-MiniLM-L6-v2`
# MAGIC    outputs are close to unit length already).
# MAGIC
# MAGIC Step 3 is doing all of the work, and it is a **hand-curated cheat**,
# MAGIC not a general mechanism -- it only "understands" that flood language and
# MAGIC safety language are related because *we* wrote that down for these
# MAGIC exact seven sentences. A real trained model learned tens of thousands of
# MAGIC equivalences like that from data, for wording nobody hand-curated. The
# MAGIC side-by-side printed further down makes this gap concrete: with the
# MAGIC concept list turned off, this same method's opinion of the flood_a /
# MAGIC flood_b pair drops substantially, because on raw words alone they barely
# MAGIC overlap.

# COMMAND ----------

_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "for", "in", "is", "it",
    "of", "on", "that", "the", "this", "to", "will", "with",
}

# Hand-written topic phrases. Each key is a synthetic token that multiple
# real phrases get mapped onto -- this is the "cheat" described above,
# curated by hand for these seven sentences only.
_CONCEPT_GROUPS = {
    "CONCEPT_FLOOD_HAZARD": ["flash flood", "flood", "water levels", "rising fast", "creek"],
    "CONCEPT_SEEK_SAFETY": ["seek higher ground", "higher ground", "move to elevated", "elevated terrain"],
    "CONCEPT_AVOID_ROADWAY": ["avoid low-lying", "roadways", "below grade"],
    "CONCEPT_HOT_WEATHER": ["heat advisory", "drink plenty of water"],
    "CONCEPT_MILD_SUNNY": ["sunny skies", "light south wind"],
    "CONCEPT_BEACH_HAZARD": ["rip current", "beaches"],
    "CONCEPT_IT_SUPPORT": ["ticket with it", "password reset", "wifi network", "scheduled maintenance"],
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _hash_token_to_dim(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    value = int(digest, 16)
    index = value % dim
    sign = 1.0 if (value // dim) % 2 == 0 else -1.0
    return index, sign


def fallback_embed_one(text: str, dim: int = 384, use_concepts: bool = True) -> np.ndarray:
    """Hand-built substitute for a real embedding. See the markdown above --
    this rewards shared literal words (always) and shared hand-curated topic
    phrases (only when use_concepts=True); it does not "understand" anything.
    """
    tokens = _tokenize(text)
    if use_concepts:
        lowered = text.lower()
        for canonical, phrases in _CONCEPT_GROUPS.items():
            for phrase in phrases:
                if phrase in lowered:
                    tokens.append(canonical)

    vector = np.zeros(dim)
    for token in tokens:
        index, sign = _hash_token_to_dim(token, dim)
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


if not USING_REAL_MODEL:
    print("=" * 72)
    print("FALLBACK IN USE -- these are NOT real sentence-transformers vectors.")
    print(f"Reason get_model() failed: {FALLBACK_REASON}")
    print("Install torch + sentence-transformers and re-run this file to see")
    print("genuine model output instead of the hand-built stand-in below.")
    print("=" * 72)
    vectors = np.array([fallback_embed_one(t) for t in TEXTS])
else:
    print("Real sentence-transformers/all-MiniLM-L6-v2 vectors loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: a 384-float vector is not a metaphor
# MAGIC
# MAGIC Whichever path just ran, `vectors` is now a real, printable NumPy array.
# MAGIC Here's its actual shape and the first 8 numbers of `flood_a`'s vector.

# COMMAND ----------

print(f"vectors.shape = {vectors.shape}")
assert vectors.shape[1] == embedder.EMBEDDING_DIM, "vector width must match embedder.EMBEDDING_DIM"

flood_a_vector = vectors[LABELS.index("flood_a")]
print(f"\nFirst 8 numbers of flood_a's vector ({'REAL MODEL' if USING_REAL_MODEL else 'FALLBACK'}):")
print([round(float(x), 6) for x in flood_a_vector[:8]])
print(
    f"\n...and {len(flood_a_vector) - 8} more numbers after that, all real "
    "floats, none of them individually meaningful to a human -- the meaning "
    "lives in the vector's *position relative to other vectors*, which is "
    "exactly what Step 4 goes and measures."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: cosine similarity, written by hand
# MAGIC
# MAGIC No library function here -- this is the entire operation pgvector's
# MAGIC `<=>` operator performs on every row, spelled out in five lines of
# MAGIC NumPy:

# COMMAND ----------


def cosine(a, b) -> float:
    """dot(a, b) / (norm(a) * norm(b)) -- the textbook definition, and the
    same arithmetic every later file in this crash course trusts SQL to do.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot_product = float(np.dot(a, b))
    length_a = float(np.linalg.norm(a))
    length_b = float(np.linalg.norm(b))
    if length_a == 0.0 or length_b == 0.0:
        return 0.0
    return dot_product / (length_a * length_b)


n = len(LABELS)
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = cosine(vectors[i], vectors[j])

# COMMAND ----------

# MAGIC %md
# MAGIC ### The full pairwise similarity table

# COMMAND ----------

col_width = max(len(label) for label in LABELS) + 2
header = " " * col_width + "".join(f"{label[:10]:>11}" for label in LABELS)
print(header)
for i, label in enumerate(LABELS):
    row = f"{label:<{col_width}}" + "".join(f"{similarity_matrix[i, j]:>11.3f}" for j in range(n))
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: the flood pair, called out explicitly
# MAGIC
# MAGIC This is the one comparison this whole notebook exists to make.

# COMMAND ----------

i_flood_a = LABELS.index("flood_a")
i_flood_b = LABELS.index("flood_b")
i_unrelated_it = LABELS.index("unrelated_it")
i_unrelated_wifi = LABELS.index("unrelated_wifi")

sim_flood_pair = similarity_matrix[i_flood_a, i_flood_b]
sim_flood_a_vs_it = similarity_matrix[i_flood_a, i_unrelated_it]
sim_flood_a_vs_wifi = similarity_matrix[i_flood_a, i_unrelated_wifi]
worst_unrelated = max(sim_flood_a_vs_it, sim_flood_a_vs_wifi)

print(f"cosine(flood_a, flood_b)        = {sim_flood_pair:.4f}   <- two paraphrases of the SAME flood warning")
print(f"cosine(flood_a, unrelated_it)   = {sim_flood_a_vs_it:.4f}   <- a request to reset a work computer's password")
print(f"cosine(flood_a, unrelated_wifi) = {sim_flood_a_vs_wifi:.4f}   <- building WiFi maintenance notice")

if sim_flood_pair > worst_unrelated:
    print(
        f"\n=> flood_a/flood_b score {sim_flood_pair - worst_unrelated:+.4f} higher than flood_a's best "
        "match among the unrelated sentences, despite sharing almost no words. "
        "That gap IS semantic search: it's the entire reason pgvector's <=> "
        "search in app.py finds the right chunk for a differently-worded query."
    )
else:
    print(
        "\n=> Not observed this run -- with the fallback vectors above this can "
        "happen if the hand-curated concept list undershoots; see the markdown "
        "on the fallback above for why it's a curated stand-in, not a general fix. "
        "Install torch + sentence-transformers and re-run to see the real model's result."
    )

if not USING_REAL_MODEL:
    raw_pair = cosine(
        fallback_embed_one(SENTENCES["flood_a"], use_concepts=False),
        fallback_embed_one(SENTENCES["flood_b"], use_concepts=False),
    )
    print(
        f"\nFor comparison, with the hand-curated concept list turned OFF "
        f"(pure shared-literal-word scoring): cosine(flood_a, flood_b) = {raw_pair:.4f} "
        f"vs {sim_flood_pair:.4f} with it on. The gap between those two numbers is "
        "entirely the hand-curated cheat described above -- a real model needs no "
        "such list to see these two sentences as related."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: the bridge to SQL -- proving `1 - distance == similarity`
# MAGIC
# MAGIC `app.py`'s actual retrieval query computes
# MAGIC `1 - (e.embedding <=> %s::vector) AS similarity`. Postgres/pgvector
# MAGIC defines `<=>` (cosine distance) as `1 - cosine_similarity`. Before any
# MAGIC of that is trusted from SQL, here it is proven in plain Python, with two
# MAGIC **independently written** functions -- `pgvector_cosine_distance` below
# MAGIC does not call `cosine()` above, it repeats the same dot/norm arithmetic
# MAGIC on its own, so the two numbers agreeing below is a real check, not `x == x`.

# COMMAND ----------


def pgvector_cosine_distance(a, b) -> float:
    """Mirrors pgvector's <=> operator definition: 1 - cosine similarity.
    Written independently of cosine() above (own dot/norm calls) on purpose.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot_product = float(np.dot(a, b))
    length_a = float(np.linalg.norm(a))
    length_b = float(np.linalg.norm(b))
    similarity = dot_product / (length_a * length_b)
    return 1.0 - similarity


pair_a, pair_b = vectors[i_flood_a], vectors[i_flood_b]

similarity_via_cosine = cosine(pair_a, pair_b)
distance_via_pgvector_formula = pgvector_cosine_distance(pair_a, pair_b)
similarity_via_one_minus_distance = 1.0 - distance_via_pgvector_formula

print(f"dot(flood_a, flood_b)                    = {float(np.dot(pair_a, pair_b)):.6f}")
print(f"norm(flood_a)                            = {float(np.linalg.norm(pair_a)):.6f}")
print(f"norm(flood_b)                            = {float(np.linalg.norm(pair_b)):.6f}")
print()
print(f"cosine(flood_a, flood_b)                 = {similarity_via_cosine:.8f}   (Step 4's helper)")
print(f"pgvector_cosine_distance(flood_a, flood_b) = {distance_via_pgvector_formula:.8f}   (independent formula)")
print(f"1 - pgvector_cosine_distance(...)        = {similarity_via_one_minus_distance:.8f}   <- must equal the line above cosine()")

assert math.isclose(similarity_via_cosine, similarity_via_one_minus_distance, rel_tol=1e-9, abs_tol=1e-9), (
    "1 - cosine_distance must exactly equal cosine_similarity -- if this ever "
    "fails, something is wrong with the arithmetic, not with floating point."
)
print(
    "\nMATCH CONFIRMED: cosine(a, b) == 1 - pgvector_cosine_distance(a, b), to "
    "within floating-point precision. This is why app.py's SQL can compute "
    "`1 - (e.embedding <=> %s::vector) AS similarity` and call the result a "
    "similarity score -- it's the exact same operation just verified here in "
    "plain Python, not a different, untrusted thing happening inside Postgres."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - A "384-dim embedding" is a NumPy array with 384 floats in it -- you
# MAGIC   just printed one.
# MAGIC - Cosine similarity is `dot(a, b) / (norm(a) * norm(b))` -- five lines of
# MAGIC   NumPy, no library call needed.
# MAGIC - Two sentences meaning the same thing can score high on that measure
# MAGIC   while sharing almost no words -- that's what makes search "semantic"
# MAGIC   instead of keyword matching.
# MAGIC - `1 - cosine_distance == cosine_similarity` always, which is exactly
# MAGIC   what lets `app.py` read pgvector's `<=>` result as a similarity score.
# MAGIC
# MAGIC Next: `02_chunking.py` -- what happens to a 9,275-character discussion
# MAGIC before any of these numbers ever get computed on it.
