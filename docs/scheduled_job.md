# Scheduled alert resync (stretch goal)

A Databricks Job, defined as a Databricks Asset Bundle (DAB), that keeps
`weather_documents` fresh for active NWS alerts without waiting on someone to
call `POST /weather/sync` by hand.

## What it does

Every 15 minutes (`0 */15 * * * ?`, `America/Chicago`), the job
`weather-alert-resync` runs one notebook task
(`notebooks/weather_sync_job.py`) that:

1. Reads the deployed app's URL from the `WEATHER_APP_URL` job parameter (or
   the environment variable of the same name).
2. Reads the pipe-separated `WEATHER_LOCATIONS` job parameter (default
   `Chicago, IL|Austin, TX|Miami, FL`, matching `WEATHER_DEFAULT_LOCATIONS`
   elsewhere in this repo).
3. POSTs to that app's `/weather/sync` with `source_types: ["alert"]` only.
4. Prints the JSON response.

Only `alert` is resynced on this cadence. NWS forecasts and area forecast
discussions are updated by the National Weather Service far less often than
every 15 minutes, so resyncing them at this frequency would just be wasted
requests; alerts are the one source type that can meaningfully change
minute to minute.

## Files

- `databricks.yml` -- the bundle root: name, included resources, and `dev` /
  `prod` targets.
- `resources/weather_sync_job.yml` -- the job definition: schedule, job
  parameters, and the single notebook task.
- `notebooks/weather_sync_job.py` -- the notebook the job runs.

## Deploying

```bash
databricks bundle deploy -t dev
```

This validates and uploads the bundle. `mode: development` on the `dev`
target namespaces the deployed job per-user and, importantly, **forces every
schedule to deploy paused** regardless of the `pause_status: UNPAUSED` set in
`resources/weather_sync_job.yml` -- that's a DAB behavior, not a bug here. To
actually get an unpaused, firing-every-15-minutes job, deploy the `prod`
target instead:

```bash
databricks bundle deploy -t prod
```

Before either command, edit the placeholder `workspace.host` in
`databricks.yml` for the target you're using -- it is intentionally left as
`REPLACE_WITH_YOUR_WORKSPACE_HOST` and must never be committed with a real
workspace URL.

## Running it

```bash
databricks bundle run weather-alert-resync -t dev
```

(swap `-t dev` for `-t prod` to run the production deployment.)

`WEATHER_APP_URL` defaults to an empty string in
`resources/weather_sync_job.yml` on purpose: this bundle is built and
deployed before the app it targets necessarily has a stable URL, so that URL
cannot be filled in automatically here. Set a real value either by editing
the `default:` in `resources/weather_sync_job.yml` to the app's actual URL
before deploying, or by overriding the job parameter for a given run (see
`databricks bundle run --help` for the parameter-override flag your CLI
version supports). Until it's set, the notebook fails fast with a clear
error naming the missing parameter rather than silently no-op'ing.

## A caveat this repo can't verify from here

Databricks Apps are authenticated by default -- a plain, unauthenticated
`requests.post` from this job may get a 401 or a login redirect instead of
reaching `/weather/sync`, depending on how the deployed app's access is
configured and whether the job's identity has been granted permission to
call it. If that happens, the fix is on the notebook side: mint a token for
the job's identity (for example via the Databricks SDK's `WorkspaceClient`)
and send it as an `Authorization: Bearer <token>` header on the POST. This
isn't wired up here because it depends on the specific auth model of the
already-deployed app, which isn't knowable from this repo alone.
