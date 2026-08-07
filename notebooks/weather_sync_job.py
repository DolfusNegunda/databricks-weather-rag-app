# Databricks notebook source
# MAGIC %md
# MAGIC ## Weather alert resync
# MAGIC
# MAGIC Run on a schedule by `resources/weather_sync_job.yml` (every 15 minutes).
# MAGIC Calls the already-deployed app's `POST /weather/sync` with
# MAGIC `source_types: ["alert"]` for the tracked locations -- alerts are the one
# MAGIC source type that can meaningfully change minute to minute; forecasts and
# MAGIC area forecast discussions update far less often and are not resynced here.
# MAGIC
# MAGIC Reads `WEATHER_APP_URL` and `WEATHER_LOCATIONS` from a job parameter first
# MAGIC (set in `resources/weather_sync_job.yml`, overridable per run), falling
# MAGIC back to an environment variable of the same name for a manual/local run.

# COMMAND ----------

import json
import os

import requests


def _param(name: str, default: str = "") -> str:
    try:
        value = dbutils.widgets.get(name)  # noqa: F821 -- injected by the Databricks runtime
    except Exception:
        value = ""
    return (value or os.environ.get(name) or default).strip()


app_url = _param("WEATHER_APP_URL")
if not app_url:
    raise ValueError(
        "WEATHER_APP_URL is required -- set it as a job parameter (see "
        "resources/weather_sync_job.yml) or as the WEATHER_APP_URL "
        "environment variable, pointing at the deployed app's base URL."
    )

# "|", not "," -- these location labels are themselves "City, ST" and
# already contain commas, so a comma-split would shred them.
locations_raw = _param("WEATHER_LOCATIONS", "Chicago, IL|Austin, TX|Miami, FL")
locations = [loc.strip() for loc in locations_raw.split("|") if loc.strip()]

# COMMAND ----------

sync_url = app_url.rstrip("/") + "/weather/sync"

response = requests.post(
    sync_url,
    json={"locations": locations, "source_types": ["alert"]},
    # The app fans out to 2 NWS calls per location (resolve_point,
    # get_alerts) at up to 15s each -- 3 locations worst-case is ~90s before
    # even touching Lakebase, so 60s is too tight.
    timeout=180,
)
response.raise_for_status()

result = response.json()
print(json.dumps(result, indent=2))
