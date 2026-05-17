# Dashboard Ingestion Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub Pages-compatible ingestion visibility and improve dashboard chart quality without adding a backend or charting dependency.

**Architecture:** Create a small `src.ingestion_status` module that persists ingestion run metadata to `reports/tables/ingestion_status.json`. Update the existing batch data download script to write that status artifact, and update `src.dashboard` to embed ingestion status plus SVG chart scaffolding into the generated static HTML.

**Tech Stack:** Python 3.11+, pandas, stdlib dataclasses/json/pathlib/datetime, existing static HTML/JS dashboard generator, pytest.

---

## File Structure

- Create `src/ingestion_status.py`: owns the JSON schema, dataclasses, status helpers, and atomic persistence.
- Create `tests/test_ingestion_status.py`: unit tests for round-trip persistence, symbol success/error summaries, and malformed/missing file handling.
- Modify `notebooks/01_data_download.py`: integrate status writing into the existing ingestion flow while keeping the printed missing-data report.
- Create `tests/test_data_download_ingestion_status.py`: script-level test using monkeypatched fake data and scratch output paths.
- Modify `src/dashboard.py`: load optional ingestion status, add an Ingestion tab, render a summary/table, and add SVG chart scaffolding for performance.
- Modify `tests/test_dashboard.py`: assert missing-status behavior, status-tab behavior, and SVG chart presence.
- Optionally update `README.md`: add one sentence pointing dashboard users to the Ingestion tab after data refreshes.

## Task 1: Ingestion Status Module

**Files:**
- Create: `src/ingestion_status.py`
- Create: `tests/test_ingestion_status.py`

- [ ] **Step 1: Write failing tests for JSON round-trip and missing status**

Add this file:

```python
"""Tests for ingestion status persistence."""

from __future__ import annotations

from pathlib import Path

from src.ingestion_status import (
    IngestionRun,
    IngestionSymbolStatus,
    load_ingestion_status,
    save_ingestion_status,
)


def test_ingestion_status_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ingestion_status.json"
    run = IngestionRun(
        run_id="run-1",
        provider="yfinance",
        start="2024-01-01",
        end="2024-01-05",
        status="partial",
        started_at="2024-01-05T10:00:00Z",
        finished_at="2024-01-05T10:00:03Z",
        symbols=[
            IngestionSymbolStatus(
                symbol="SPY",
                status="success",
                rows=3,
                first_date="2024-01-02",
                last_date="2024-01-04",
                total_missing=0,
            ),
            IngestionSymbolStatus(
                symbol="BAD",
                status="error",
                rows=0,
                error="No OHLCV rows returned",
            ),
        ],
    )

    save_ingestion_status(run, path)

    loaded = load_ingestion_status(path)
    assert loaded is not None
    assert loaded["status"] == "partial"
    assert loaded["provider"] == "yfinance"
    assert loaded["symbols"][0]["symbol"] == "SPY"
    assert loaded["symbols"][1]["error"] == "No OHLCV rows returned"


def test_load_ingestion_status_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_ingestion_status(tmp_path / "missing.json") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_ingestion_status.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion_status'`.

- [ ] **Step 3: Implement minimal module**

Create `src/ingestion_status.py`:

```python
"""Data-ingestion status artifacts for dashboard visibility."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


RunStatus = Literal["running", "success", "partial", "failed"]
SymbolStatus = Literal["pending", "success", "error"]


@dataclass(frozen=True)
class IngestionSymbolStatus:
    symbol: str
    status: SymbolStatus
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    total_missing: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IngestionRun:
    run_id: str
    provider: str
    start: str
    end: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    symbols: list[IngestionSymbolStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = [symbol.to_dict() for symbol in self.symbols]
        return payload


def save_ingestion_status(run: IngestionRun, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.to_dict()
    fd, tmp_name = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, out_path)
    except Exception:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()
        raise
    return out_path


def load_ingestion_status(path: str | Path) -> dict[str, object] | None:
    in_path = Path(path)
    if not in_path.exists():
        return None
    return json.loads(in_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_ingestion_status.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\ingestion_status.py tests\test_ingestion_status.py
git commit -m "Add ingestion status artifact model"
```

## Task 2: Data Download Status Emission

**Files:**
- Modify: `notebooks/01_data_download.py`
- Create: `tests/test_data_download_ingestion_status.py`

- [ ] **Step 1: Write failing script-level test**

Add `tests/test_data_download_ingestion_status.py`:

```python
"""Tests for data-download ingestion status output."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "notebooks" / "01_data_download.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("data_download_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    ).rename_axis("Date")


def test_data_download_writes_ingestion_status(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    status_path = tmp_path / "tables" / "ingestion_status.json"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            provider="fake",
            start="2024-01-01",
            end="2024-01-05",
            symbols="SPY,QQQ",
            status_path=str(status_path),
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        ),
    )
    monkeypatch.setattr(
        module,
        "download_ohlcv",
        lambda symbols, start, end, provider: {symbol: _sample_frame() for symbol in symbols},
    )
    monkeypatch.setattr(module, "enrich_market_data", lambda frame: frame.assign(state=0))

    module.main()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["provider"] == "fake"
    assert [row["symbol"] for row in payload["symbols"]] == ["SPY", "QQQ"]
    assert payload["symbols"][0]["rows"] == 3
    assert payload["symbols"][0]["first_date"] == "2024-01-02"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests\test_data_download_ingestion_status.py -q
```

Expected: FAIL with an unexpected `status_path`/`raw_dir`/`processed_dir` attribute or missing status file.

- [ ] **Step 3: Update script args and status writing**

Modify `notebooks/01_data_download.py`:

```python
from datetime import UTC, datetime
from uuid import uuid4

from src.config import DEFAULT_END, DEFAULT_START, FIRST_EXPERIMENT_SYMBOLS, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from src.ingestion_status import IngestionRun, IngestionSymbolStatus, save_ingestion_status
```

Add arguments in `parse_args()`:

```python
parser.add_argument("--raw-dir", default=RAW_DATA_DIR)
parser.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
parser.add_argument("--status-path", default=str(Path(TABLES_DIR) / "ingestion_status.json"))
```

Add helpers:

```python
def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _symbol_status(symbol: str, frame) -> IngestionSymbolStatus:
    missing = int(frame.isna().sum().sum())
    return IngestionSymbolStatus(
        symbol=symbol,
        status="success",
        rows=int(len(frame)),
        first_date=frame.index.min().strftime("%Y-%m-%d") if len(frame) else None,
        last_date=frame.index.max().strftime("%Y-%m-%d") if len(frame) else None,
        total_missing=missing,
    )
```

Update `main()` to create an `IngestionRun(status="running")`, save it before download, then save `success` after processed output is written. On exceptions, save `failed` with one `IngestionSymbolStatus(status="error", error=str(exc))` per symbol if no per-symbol data exists, then re-raise.

Use `save_raw(data, args.raw_dir)` and `save_processed(enriched, args.processed_dir)`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests\test_data_download_ingestion_status.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add notebooks\01_data_download.py tests\test_data_download_ingestion_status.py
git commit -m "Write ingestion status during data downloads"
```

## Task 3: Dashboard Ingestion Tab and Missing-State Handling

**Files:**
- Modify: `src/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing dashboard assertions**

In `tests/test_dashboard.py`, extend `test_generate_dashboard_writes_self_contained_html` after the other CSV fixtures:

```python
    (tables_dir / "ingestion_status.json").write_text(
        """
{
  "run_id": "run-1",
  "provider": "yfinance",
  "start": "2024-01-01",
  "end": "2024-01-05",
  "status": "success",
  "started_at": "2024-01-05T10:00:00Z",
  "finished_at": "2024-01-05T10:00:03Z",
  "symbols": [
    {"symbol": "SPY", "status": "success", "rows": 3, "first_date": "2024-01-02", "last_date": "2024-01-04", "total_missing": 0}
  ]
}
""",
        encoding="utf-8",
    )
```

Add assertions:

```python
    assert "Ingestion" in html
    assert "Latest Ingestion Run" in html
    assert "ingestionStatus" in html
    assert "yfinance" in html
```

Add a second test:

```python
def test_generate_dashboard_handles_missing_ingestion_status():
    root = _scratch_dir()
    tables_dir = root / "tables"
    dashboard_dir = root / "dashboard"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_dashboard_tables(tables_dir)

    output_path = generate_dashboard(tables_dir=tables_dir, output_dir=dashboard_dir)

    html = output_path.read_text(encoding="utf-8")
    assert "No ingestion status has been published yet" in html
```

If `_write_minimal_dashboard_tables` does not exist, extract the existing fixture setup in `test_generate_dashboard_writes_self_contained_html` into that helper.

- [ ] **Step 2: Run dashboard tests to verify failure**

Run:

```powershell
python -m pytest tests\test_dashboard.py -q
```

Expected: FAIL because dashboard data does not include `ingestionStatus` and HTML has no Ingestion tab.

- [ ] **Step 3: Implement ingestion loading and tab rendering**

In `src/dashboard.py`:

Add imports:

```python
from json import JSONDecodeError
```

In `_load_dashboard_data`, add:

```python
    ingestion_status, ingestion_error = _read_ingestion_status(tables_dir / "ingestion_status.json")
```

Add keys:

```python
        "ingestionStatus": ingestion_status,
        "ingestionError": ingestion_error,
```

Add helper:

```python
def _read_ingestion_status(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        return None, f"Could not read ingestion_status.json: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "ingestion_status.json did not contain an object"
    return payload, None
```

In `_render_dashboard_html`, add a new tab button:

```html
<button class="tab" data-panel="ingestion">Ingestion</button>
```

Add a panel:

```html
<section id="ingestion" class="panel grid two" hidden>
  <div class="card">
    <h2>Latest Ingestion Run</h2>
    <div id="ingestion-summary"></div>
  </div>
  <div class="card">
    <h2>Symbol Status</h2>
    <div class="table-wrap"><table id="ingestion-table"></table></div>
  </div>
</section>
```

Add JS renderer:

```javascript
function renderIngestion() {
  const summary = document.getElementById("ingestion-summary");
  const status = dashboardData.ingestionStatus;
  if (dashboardData.ingestionError) {
    summary.innerHTML = `<div class="note">${{escapeHtml(dashboardData.ingestionError)}}</div>`;
    renderTable("ingestion-table", ["symbol", "status", "rows", "total_missing", "error"], []);
    return;
  }
  if (!status) {
    summary.innerHTML = `<div class="note">No ingestion status has been published yet.</div>`;
    renderTable("ingestion-table", ["symbol", "status", "rows", "total_missing", "error"], []);
    return;
  }
  summary.innerHTML = `
    <div class="summary-block">
      <h3>${{escapeHtml(status.status || "unknown")}}</h3>
      <ul>
        <li>Provider: ${{escapeHtml(status.provider || "unknown")}}</li>
        <li>Window: ${{escapeHtml(status.start || "")}} to ${{escapeHtml(status.end || "")}}</li>
        <li>Started: ${{escapeHtml(status.started_at || "")}}</li>
        <li>Finished: ${{escapeHtml(status.finished_at || "running")}}</li>
      </ul>
    </div>`;
  renderTable("ingestion-table", ["symbol", "status", "rows", "first_date", "last_date", "total_missing", "error"], status.symbols || []);
}
```

Call `renderIngestion()` before setting `sources`.

- [ ] **Step 4: Run dashboard tests**

Run:

```powershell
python -m pytest tests\test_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\dashboard.py tests\test_dashboard.py
git commit -m "Show ingestion status in dashboard"
```

## Task 4: SVG Chart Improvements

**Files:**
- Modify: `src/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing SVG chart assertions**

In `tests/test_dashboard.py`, add assertions to the main dashboard test:

```python
    assert "<svg" in html
    assert "walkforward-return-chart" in html
    assert "walkforward-sharpe-chart" in html
```

- [ ] **Step 2: Run dashboard test to verify failure**

Run:

```powershell
python -m pytest tests\test_dashboard.py -q
```

Expected: FAIL because the current dashboard only renders div-based bars.

- [ ] **Step 3: Add self-contained SVG chart rendering**

In `src/dashboard.py`, add two containers in the Performance panel:

```html
<div id="walkforward-return-chart"></div>
<div id="walkforward-sharpe-chart"></div>
```

Add JS function:

```javascript
function renderSvgBarChart(containerId, rows, valueKey, label) {
  const container = document.getElementById(containerId);
  const width = 760;
  const rowHeight = 28;
  const height = Math.max(90, rows.length * rowHeight + 42);
  const values = rows.map(row => Number(row[valueKey]) || 0);
  const maxAbs = Math.max(...values.map(value => Math.abs(value)), 0.01);
  const zeroX = 160 + (width - 230) / 2;
  const scale = (width - 230) / 2 / maxAbs;
  const bars = rows.map((row, index) => {
    const value = Number(row[valueKey]) || 0;
    const y = 28 + index * rowHeight;
    const barX = value >= 0 ? zeroX : zeroX + value * scale;
    const barWidth = Math.max(2, Math.abs(value * scale));
    const color = value >= 0 ? "#147d64" : "#b42318";
    return `
      <text x="8" y="${y + 14}" font-size="12" fill="#17202a">${escapeHtml(row.symbol || "")}</text>
      <rect x="${barX}" y="${y}" width="${barWidth}" height="16" rx="3" fill="${color}">
        <title>${escapeHtml(row.symbol || "")}: ${formatPct(value)}</title>
      </rect>
      <text x="${width - 56}" y="${y + 14}" font-size="12" text-anchor="end" fill="#617080">${formatPct(value)}</text>`;
  }).join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)}">
      <text x="8" y="16" font-size="13" font-weight="700" fill="#17202a">${escapeHtml(label)}</text>
      <line x1="${zeroX}" x2="${zeroX}" y1="22" y2="${height - 12}" stroke="#dbe2ea" />
      ${bars}
    </svg>`;
}
```

Call it from `renderPerformance()`:

```javascript
renderSvgBarChart("walkforward-return-chart", rows, "total_return", "Total return by symbol");
renderSvgBarChart("walkforward-sharpe-chart", rows, "sharpe", "Sharpe by symbol");
```

Keep the old bar rows only if they still serve a separate purpose; otherwise remove them from `renderPerformance`.

- [ ] **Step 4: Run dashboard tests**

Run:

```powershell
python -m pytest tests\test_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\dashboard.py tests\test_dashboard.py
git commit -m "Improve dashboard performance charts"
```

## Task 5: End-to-End Verification and Publish Prep

**Files:**
- Optional modify: `README.md`
- Generated only, do not commit unless explicitly desired: `reports/dashboard/index.html`

- [ ] **Step 1: Optionally add README note**

If README should mention the new tab, add one sentence under dashboard or workflow docs:

```markdown
The dashboard includes an Ingestion tab when `reports/tables/ingestion_status.json` has been generated by `notebooks/01_data_download.py`.
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m pytest tests\test_ingestion_status.py tests\test_data_download_ingestion_status.py tests\test_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 3: Build dashboard locally**

Run:

```powershell
python scripts\build_dashboard.py
```

Expected output includes:

```text
Saved dashboard to reports\dashboard\index.html
```

- [ ] **Step 4: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS with only optional-library skips for `ruptures`, `hmmlearn`, and `hypothesis` if those extras are not installed.

- [ ] **Step 5: Commit README if changed**

```powershell
git add README.md
git commit -m "Document dashboard ingestion status"
```

Skip this commit if README was not changed.

## Self-Review

Spec coverage:

- Ingestion JSON contract: Task 1.
- Data download status writes: Task 2.
- Dashboard Ingestion tab and empty/malformed handling: Task 3.
- Better self-contained charts: Task 4.
- Verification and publish readiness: Task 5.

Placeholder scan:

- No placeholder markers or unspecified implementation gaps remain.

Type consistency:

- `IngestionRun`, `IngestionSymbolStatus`, `save_ingestion_status`, and `load_ingestion_status` names are consistent across tasks.
- Dashboard keys use `ingestionStatus` and `ingestionError` consistently.
