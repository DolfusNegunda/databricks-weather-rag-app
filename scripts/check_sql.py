"""
Offline checks that every SQL statement this project can emit is valid
PostgreSQL grammar, via pglast (libpg_query -- Postgres's own parser).

Fully offline: no network, no database, no import of app.py/the notebook as
live modules (both run side-effecting code -- a schema bootstrap, a live
`lakebase.run_query` -- at module-import time, which would need a real
database connection). Instead this reads their *source text* and pulls out
the SQL string literals by name via `ast`, so what gets checked is always
the actual current SQL those files execute, never a hand-retyped copy that
could quietly drift from it.

A pass here proves the grammar parses. It does NOT prove the SQL computes
the right answer -- pglast would happily accept a query that joins the wrong
columns or divides by the wrong value. scripts/check_connection.py is the
counterpart that checks against a live database and proves the *arithmetic*
(a vector compared against itself really does return cosine similarity 1.0).

Runnable as `python scripts/check_sql.py` or pasted into a notebook cell.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    from pglast import parse_sql
except ImportError as exc:  # pragma: no cover
    sys.exit(f"pglast is required: pip install pglast ({exc})")

try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _candidate = Path.cwd()
    while not (_candidate / "app.py").exists() and _candidate.parent != _candidate:
        _candidate = _candidate.parent
    _PROJECT_ROOT = _candidate

PASSED = 0
FAILED: list[str] = []


def check(name: str, sql: str) -> None:
    global PASSED
    try:
        statements = parse_sql(sql)
        if not statements:
            raise ValueError("parsed to zero statements")
        PASSED += 1
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 -- reported, not re-raised
        FAILED.append(name)
        print(f"FAIL {name} :: {exc}\n      sql={sql!r}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ----------------------------------------------------------------------------
# extracting real SQL text from app.py / the notebook, without importing
# either as a live module (both run database-touching code at import time)
# ----------------------------------------------------------------------------

_NAMED_PARAM = re.compile(r"%\(\w+\)s")
_POSITIONAL_PARAM = re.compile(r"%s")


def _string_constants(source: str) -> dict[str, str]:
    """Every module-level `NAME = "..."` / `NAME = '''...'''` string literal
    in `source`, keyed by NAME. Triple-quoted strings are plain ast.Constant
    nodes like any other string literal -- no special-casing needed."""
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out[target.id] = node.value.value
    return out


def _placeholder_string(sql: str) -> str:
    """Named (%(x)s) and positional (%s) psycopg2 placeholders are not valid
    Postgres syntax on their own -- swap each for a literal NULL so the
    surrounding SQL parses as real Postgres would eventually receive it after
    binding. NULL is valid wherever these placeholders appear in this
    project's queries, including inside a %s::vector / %s::text cast and in
    a LIMIT clause."""
    sql = _NAMED_PARAM.sub("NULL", sql)
    sql = _POSITIONAL_PARAM.sub("NULL", sql)
    return sql


# ----------------------------------------------------------------------------
# sql/*.sql -- the DDL, exactly as ensure_weather_schema() renders it
# ----------------------------------------------------------------------------

section("sql/*.sql (DDL, {{EMBEDDING_DIM}} substituted)")

for filename in ("01_weather_documents.sql", "02_weather_embeddings.sql"):
    text = (_PROJECT_ROOT / "sql" / filename).read_text(encoding="utf-8")
    text = text.replace("{{EMBEDDING_DIM}}", "384")
    check(filename, text)

# ----------------------------------------------------------------------------
# app.py -- the two hand-built queries, pulled from the real source
# ----------------------------------------------------------------------------

section("app.py SQL constants")

_app_source = (_PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
_app_sql = _string_constants(_app_source)

for name in ("_UPSERT_WEATHER_DOCUMENT_SQL", "_WEATHER_SEARCH_SQL"):
    if name not in _app_sql:
        FAILED.append(f"app.py defines {name}")
        print(f"FAIL app.py defines {name} :: not found as a module-level string constant")
        continue
    PASSED += 1
    print(f"PASS app.py defines {name}")
    check(f"app.py {name} (placeholders -> NULL)", _placeholder_string(_app_sql[name]))

# The search query is bound with 5 positional params in this exact order --
# found via ast (not a hand-rolled regex over source text, which is fragile
# against incidental parens/commas elsewhere in the file) so a future edit
# that adds/removes a placeholder without updating the bind tuple, or vice
# versa, fails this check immediately rather than only at request time.
def _find_execute_call(tree: ast.AST, sql_name: str) -> ast.Call | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Name) and first_arg.id == sql_name:
            return node
    return None


_app_tree = ast.parse(_app_source)
_execute_call = _find_execute_call(_app_tree, "_WEATHER_SEARCH_SQL")
_placeholder_count = len(_POSITIONAL_PARAM.findall(_app_sql.get("_WEATHER_SEARCH_SQL", "")))
check_name = "_WEATHER_SEARCH_SQL: bound param count matches %s placeholder count"

if _execute_call is None or len(_execute_call.args) < 2:
    FAILED.append(check_name)
    print(f"FAIL {check_name} :: could not locate a cur.execute(_WEATHER_SEARCH_SQL, (...)) call")
else:
    _params_arg = _execute_call.args[1]
    _bound_count = len(_params_arg.elts) if isinstance(_params_arg, (ast.Tuple, ast.List)) else None
    if _bound_count == _placeholder_count:
        PASSED += 1
        print(f"PASS {check_name} ({_bound_count} == {_placeholder_count})")
    else:
        FAILED.append(check_name)
        print(f"FAIL {check_name} :: {_bound_count} bound value(s) vs {_placeholder_count} placeholder(s)")

# ----------------------------------------------------------------------------
# the ingestion notebook -- the execute_values upsert, after simulating the
# substitution execute_values performs (a template repeated once per row)
# ----------------------------------------------------------------------------

section("notebooks/ingest_weather_embeddings.py SQL constants")

_notebook_source = (_PROJECT_ROOT / "notebooks" / "ingest_weather_embeddings.py").read_text(
    encoding="utf-8"
)
_notebook_sql = _string_constants(_notebook_source)

if "_UPSERT_EMBEDDINGS_SQL" not in _notebook_sql:
    FAILED.append("notebook defines _UPSERT_EMBEDDINGS_SQL")
    print("FAIL notebook defines _UPSERT_EMBEDDINGS_SQL :: not found as a module-level string constant")
else:
    PASSED += 1
    print("PASS notebook defines _UPSERT_EMBEDDINGS_SQL")

    _template_match = re.search(
        r'execute_values\(\s*cur,\s*_UPSERT_EMBEDDINGS_SQL,\s*insert_rows,\s*template=("(?:[^"\\]|\\.)*")',
        _notebook_source,
    )
    if not _template_match:
        FAILED.append("notebook: could not locate execute_values(...) template= to check")
        print("FAIL notebook: could not locate execute_values(...) template= to check")
    else:
        template = ast.literal_eval(_template_match.group(1))
        print(f"      found template: {template!r}")

        # This is exactly what execute_values does under the hood: bind one
        # occurrence of `template` per row, comma-join them, and substitute
        # that in place of the single %s in the outer "VALUES %s".
        one_row = template % (
            "'alert:0'", "'alert'", "0", "'chunk text'", "'[0.1,0.2,0.3]'", "'sentence-transformers/all-MiniLM-L6-v2'",
        )
        rendered = _notebook_sql["_UPSERT_EMBEDDINGS_SQL"].replace("VALUES %s", f"VALUES {one_row}")
        check("notebook _UPSERT_EMBEDDINGS_SQL (execute_values, one row substituted)", rendered)

        two_rows = ", ".join(
            [
                template % ("'alert:0'", "'alert'", "0", "'chunk text A'", "'[0.1,0.2,0.3]'", "'m'"),
                template % ("'alert:1'", "'alert'", "1", "'chunk text B'", "'[0.4,0.5,0.6]'", "'m'"),
            ]
        )
        rendered_two = _notebook_sql["_UPSERT_EMBEDDINGS_SQL"].replace("VALUES %s", f"VALUES {two_rows}")
        check("notebook _UPSERT_EMBEDDINGS_SQL (execute_values, two rows substituted)", rendered_two)

# ----------------------------------------------------------------------------
# negative controls -- pglast must actually reject these, proving the checks
# above are discriminating rather than accepting anything handed to them
# ----------------------------------------------------------------------------

section("negative controls (the checker must reject these)")

_BAD = {
    "unbalanced parenthesis": "SELECT count(* FROM weather_documents;",
    "typo'd keyword": "SELCT 1 FROM weather_documents;",
    "dangling ORDER BY": "SELECT 1 FROM weather_documents ORDER BY;",
    "unterminated string literal": "SELECT 'unterminated FROM weather_documents;",
}
for label, bad_sql in _BAD.items():
    try:
        parse_sql(bad_sql)
    except Exception:  # noqa: BLE001 -- expected; this IS the pass condition
        PASSED += 1
        print(f"PASS rejected: {label}")
    else:
        FAILED.append(f"negative control not rejected: {label}")
        print(f"FAIL accepted invalid SQL: {label}")

# ----------------------------------------------------------------------------
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed checks:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
print("SQL ALL GREEN")
