-- Setup script for weather_documents table
-- Run this manually in your Lakebase Postgres database before running the app
-- or the embedding notebook. Idempotent -- safe to run more than once.

-- Raw, normalized weather text harvested from the National Weather Service
-- API: active alerts, gridpoint forecasts, and area forecast discussions.
-- This is the RAW document store; notebooks/ingest_weather_embeddings.py
-- reads from it and writes vectors into a separate weather_embeddings table.
CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,   -- NWS alert id, or a stable hash for
                                        -- forecast/discussion documents
    location       TEXT NOT NULL,      -- "Chicago, IL" as given by the caller
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    office         TEXT,               -- NWS forecast office, e.g. "LOT"
    grid_x         INTEGER,
    grid_y         INTEGER,
    source_type    TEXT NOT NULL,      -- 'alert' | 'forecast' | 'discussion'
    event          TEXT,               -- e.g. "Flash Flood Warning"
    headline       TEXT,
    severity       TEXT,
    narrative_text TEXT NOT NULL,      -- the free text that gets embedded
    issued_at      TIMESTAMPTZ,
    effective_at   TIMESTAMPTZ,
    expires_at     TIMESTAMPTZ,
    payload        JSONB NOT NULL,     -- raw API response, for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_hash   TEXT,               -- sha1(narrative_text); lets a re-sync
                                        -- detect real changes vs a re-fetch
                                        -- of unchanged text
    embedded_at    TIMESTAMPTZ,        -- NULL => needs (re-)embedding

    CONSTRAINT weather_documents_source_type_valid
        CHECK (source_type IN ('alert', 'forecast', 'discussion')),
    CONSTRAINT weather_documents_narrative_not_blank
        CHECK (length(btrim(narrative_text)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type, issued_at DESC);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

-- The ingestion notebook's very first query is "which rows still need
-- embedding" -- a partial index keeps that cheap even as the table grows,
-- since most rows will have embedded_at set most of the time.
CREATE INDEX IF NOT EXISTS idx_weather_documents_unembedded
    ON weather_documents (synced_at)
    WHERE embedded_at IS NULL;

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
