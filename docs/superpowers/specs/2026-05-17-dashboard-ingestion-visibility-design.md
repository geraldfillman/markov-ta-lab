# Dashboard Ingestion Visibility Design

## Goal

Improve the static dashboard charts and add GitHub Pages-compatible visibility into the latest data ingestion run.

## Scope

This first slice adds "live-ish" ingestion visibility, not a backend stream. The ingestion process writes a small run artifact as it works, and the static dashboard reads that artifact at build time. Users can refresh the published dashboard after a run to see ingestion status, provider, symbols, row counts, missing-data counts, errors, and timestamps.

## Architecture

### Ingestion Run Log

Add a reusable ingestion logging module under `src/ingestion_status.py`.

The module owns:

- `IngestionRun`: run-level metadata such as `run_id`, `provider`, `start`, `end`, `status`, `started_at`, `finished_at`.
- `IngestionSymbolStatus`: per-symbol metadata such as `symbol`, `status`, `rows`, `first_date`, `last_date`, `total_missing`, and `error`.
- JSON persistence to `reports/tables/ingestion_status.json`.

The JSON file is the contract between ingestion and dashboard generation.

### Data Download Integration

Update `notebooks/01_data_download.py` to write the ingestion status file during the batch process.

Expected behavior:

- At run start, write `status: running`.
- For each symbol, record `success` or `error`.
- On successful completion, write `status: success`.
- If a symbol fails but others continue, keep the run as `partial`.
- If the whole run fails before useful output, write `status: failed`.

The script should still print the existing missing-data report.

### Dashboard Improvements

Update `src/dashboard.py` to load optional `ingestion_status.json`.

Add:

- An "Ingestion" tab.
- A compact run summary card.
- A per-symbol ingestion table.
- Clear empty state when the file does not exist.

Improve charts without adding dependencies:

- Replace the current basic bar blocks with SVG charts for walk-forward return and Sharpe.
- Add tooltip/title text and axis labels.
- Keep all rendering self-contained in the generated HTML so GitHub Pages remains simple.

## Error Handling

Missing `ingestion_status.json` should not fail dashboard generation. It should render an empty-state message.

Malformed JSON should not break the whole dashboard. The dashboard should surface a concise ingestion warning in the Ingestion tab and continue rendering the existing research tables.

## Testing

Add focused tests for:

- Ingestion status JSON round-trip.
- `01_data_download.py` emits a status file for a small fake provider path or injected fake data path.
- Dashboard generation includes the Ingestion tab when status exists.
- Dashboard generation still succeeds when status is missing.
- Dashboard HTML contains SVG chart scaffolding for improved visualizations.

## Out Of Scope

- Browser-pushed real-time streaming.
- A FastAPI/Flask service.
- WebSockets or server-sent events.
- New charting dependencies.
- New data providers.

## Follow-Up

Once the JSON contract is stable, a local live monitor can tail the same status file and update a browser view during ingestion.
