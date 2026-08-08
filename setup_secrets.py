# Databricks notebook source
# MAGIC %md
# MAGIC # Lakebase Weather RAG — store your connection string
# MAGIC
# MAGIC Run this **once**, before deploying the app. It puts your Lakebase
# MAGIC connection string into a Databricks secret scope so the app can read it
# MAGIC at runtime without it ever appearing in Git.
# MAGIC
# MAGIC No API key is needed for the data source (the National Weather Service
# MAGIC API is free and keyless), so this notebook only handles one secret.
# MAGIC
# MAGIC **Already have a Lakebase instance from another project?** Reuse its
# MAGIC connection string here -- this app keeps its tables in their own schema
# MAGIC (`weather`, not `public`), created automatically, so it cannot collide
# MAGIC with another app's tables on the same instance. Pasting the same
# MAGIC connection string you already have is safe; it just overwrites the
# MAGIC secret with an identical value.
# MAGIC
# MAGIC ### Before you start
# MAGIC
# MAGIC Have your Lakebase connection string ready. It looks like:
# MAGIC
# MAGIC ```
# MAGIC postgresql://<role>:<password>@<instance>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
# MAGIC ```
# MAGIC
# MAGIC Percent-encode special characters in the password: `@` → `%40`, `:` →
# MAGIC `%3A`, `/` → `%2F`, `?` → `%3F`, `#` → `%23`, `%` → `%25`.
# MAGIC
# MAGIC ### How to use this notebook
# MAGIC
# MAGIC 1. Run **Cell 1** — creates the scope and an input box at the top.
# MAGIC 2. **Paste your connection string into that box.** Do not type it into a
# MAGIC    code cell; code cells are saved in the notebook's revision history.
# MAGIC 3. Run **Cell 2** — stores the secret and clears the box.
# MAGIC 4. Attach the secret to your app (instructions in the last cell).
# MAGIC 5. **Delete this notebook** when you are done.
# MAGIC
# MAGIC Nothing here ever prints your password.
# MAGIC
# MAGIC > Widget creation and widget reading are deliberately in **separate
# MAGIC > cells**. Doing both in one cell reads the box before you have had a
# MAGIC > chance to type into it, and silently stores an empty secret.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — create the scope and the input box

# COMMAND ----------

from databricks.sdk import WorkspaceClient

SCOPE = "database"
KEY = "lakebase-url"

w = WorkspaceClient()

try:
    w.secrets.create_scope(scope=SCOPE)
    print(f"Created secret scope '{SCOPE}'.")
except Exception as exc:
    print(f"Scope '{SCOPE}' already exists (that is fine): {type(exc).__name__}")

dbutils.widgets.text("dsn", "", "Lakebase connection string")

print("\nNow paste your connection string into the 'Lakebase connection string'")
print("box at the TOP of this notebook, then run Cell 2.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — store it
# MAGIC
# MAGIC Only run this **after** pasting the value into the box above. If the box
# MAGIC is still empty this cell stops with an error rather than storing a blank
# MAGIC secret, which would fail later in a confusing way.

# COMMAND ----------

from urllib.parse import urlparse

dsn = dbutils.widgets.get("dsn").strip().strip('"').strip("'")

if not dsn:
    raise ValueError(
        "The 'dsn' box is empty. Paste the connection string into the box at "
        "the top of the notebook, then re-run this cell."
    )
if not dsn.startswith(("postgresql://", "postgres://")):
    raise ValueError(
        "That does not look like a Postgres connection string — it should "
        "start with postgresql:// . (Value not shown.)"
    )

parsed = urlparse(dsn)
if not parsed.hostname:
    raise ValueError("The connection string has no host. (Value not shown.)")
if not parsed.username:
    raise ValueError("The connection string has no username. (Value not shown.)")

w.secrets.put_secret(scope=SCOPE, key=KEY, string_value=dsn)
dbutils.widgets.remove("dsn")

# Confirm without revealing: compare lengths of what went in and what came back.
stored_ok = len(dbutils.secrets.get(scope=SCOPE, key=KEY)) == len(dsn)

print(f"Stored secret      : {SCOPE}/{KEY}")
print(f"Verified round-trip: {stored_ok}")
print("\nConnection string points at (password not shown):")
print(f"  host    : {parsed.hostname}")
print(f"  port    : {parsed.port or 5432}")
print(f"  user    : {parsed.username}")
print(f"  database: {parsed.path.lstrip('/') or 'databricks_postgres'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — optional: grant the app read access
# MAGIC
# MAGIC A Databricks App runs as its own service principal, not as you. Adding
# MAGIC the Secret resource in the Apps UI normally handles this. Only run this
# MAGIC cell if the deployed app reports it cannot read the secret.
# MAGIC
# MAGIC Find the service principal's **client id** on the app's **Authorization**
# MAGIC tab, set it below, then run the cell.

# COMMAND ----------

APP_SERVICE_PRINCIPAL = ""  # e.g. "1234abcd-..."; leave blank to skip

if APP_SERVICE_PRINCIPAL:
    from databricks.sdk.service.workspace import AclPermission

    w.secrets.put_acl(
        scope=SCOPE, principal=APP_SERVICE_PRINCIPAL, permission=AclPermission.READ
    )
    print(f"Granted READ on '{SCOPE}' to {APP_SERVICE_PRINCIPAL}.")
    for acl in w.secrets.list_acls(scope=SCOPE):
        print(f"  {acl.principal}: {acl.permission}")
else:
    print("Skipped — set APP_SERVICE_PRINCIPAL above if the app cannot read the secret.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC The secret exists. Now point the app at it.
# MAGIC
# MAGIC On your app: **Edit → Resources → Add resource → Secret**
# MAGIC
# MAGIC | Field | Value |
# MAGIC | --- | --- |
# MAGIC | Secret scope | `database` |
# MAGIC | Secret key | `lakebase-url` |
# MAGIC | **Resource key** | **`lakebase-url`** — only needed if you use the `valueFrom` block in `app.yaml` |
# MAGIC | Permission | Can read |
# MAGIC
# MAGIC The default setup (`LAKEBASE_SECRET_SCOPE` / `LAKEBASE_SECRET_KEY` env vars
# MAGIC already in `app.yaml`) reads the secret via the SDK and needs no resource
# MAGIC binding — this step is only for the alternative `valueFrom` path.
# MAGIC
# MAGIC Then **Deploy**. The app creates its tables on first boot; there is
# MAGIC nothing else to run before `POST /weather/sync`.
# MAGIC
# MAGIC Confirm at `https://<your-app-url>/healthz`.
# MAGIC
# MAGIC ### Finally
# MAGIC
# MAGIC **Delete this notebook.** Widget values persist in notebook state and
# MAGIC revision history can retain what was typed into them. The secret is
# MAGIC safely in the scope now; this notebook is no longer needed.
