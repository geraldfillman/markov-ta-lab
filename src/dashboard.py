"""Static dashboard generation for Markov TA Lab reports."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR
from src.ingestion_status import load_ingestion_status


def generate_dashboard(
    tables_dir: str | Path = TABLES_DIR,
    output_dir: str | Path = "reports/dashboard",
    ingestion_status_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML dashboard from report CSV tables."""
    tables_path = Path(tables_dir)
    output_path = Path(output_dir) / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_dashboard_data(tables_path, ingestion_status_path=ingestion_status_path)
    output_path.write_text(_render_dashboard_html(data), encoding="utf-8")
    return output_path


def _load_dashboard_data(tables_dir: Path, ingestion_status_path: str | Path | None = None) -> dict[str, object]:
    walkforward = _read_csv(tables_dir / "walkforward_backtest_summary.csv")
    baselines = _read_csv(tables_dir / "walkforward_baseline_comparison.csv")
    vol_conditioned = _read_csv(tables_dir / "vol_conditioned_state_expectancy.csv")
    clusters = _read_latest_clusters(tables_dir)
    cluster_pooled = _read_optional_csv(tables_dir / "cluster_pooled_state_expectancy.csv")
    markov_weighted = _read_optional_csv(tables_dir / "markov_weighted_ev.csv")
    sensitivity = _read_optional_csv(tables_dir / "walkforward_sensitivity.csv")
    stability = _read_optional_csv(tables_dir / "sensitivity_stability_summary.csv")
    sharpe_ci = _read_optional_csv(tables_dir / "walkforward_sharpe_ci.csv")
    ingestion_status, ingestion_error, ingestion_message = _load_ingestion_snapshot(
        Path(ingestion_status_path) if ingestion_status_path is not None else tables_dir / "ingestion_status.json"
    )

    return {
        "walkforward": _records(walkforward),
        "baselines": _records(baselines),
        "volConditioned": _records(vol_conditioned),
        "clusters": _records(clusters),
        "clusterPooled": _records(cluster_pooled),
        "markovWeighted": _records(markov_weighted),
        "sensitivity": _records(sensitivity),
        "stability": _records(stability),
        "sharpeCi": _records(sharpe_ci),
        "ingestionStatus": ingestion_status,
        "ingestionError": ingestion_error,
        "ingestionMessage": ingestion_message,
        "summary": _summary(walkforward, baselines, clusters),
        "sources": {
            "walkforward": "walkforward_backtest_summary.csv",
            "baselines": "walkforward_baseline_comparison.csv",
            "volConditioned": "vol_conditioned_state_expectancy.csv",
            "clusters": _latest_cluster_name(tables_dir),
            "clusterPooled": "cluster_pooled_state_expectancy.csv",
            "markovWeighted": "markov_weighted_ev.csv",
            "sensitivity": "walkforward_sensitivity.csv",
            "stability": "sensitivity_stability_summary.csv",
            "sharpeCi": "walkforward_sharpe_ci.csv",
            "ingestionStatus": "ingestion_status.json",
        },
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dashboard input missing: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_latest_clusters(tables_dir: Path) -> pd.DataFrame:
    candidates = sorted(
        tables_dir.glob("asset_behavior_clusters*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Dashboard input missing: asset_behavior_clusters*.csv")
    return pd.read_csv(candidates[0])


def _latest_cluster_name(tables_dir: Path) -> str:
    candidates = sorted(
        tables_dir.glob("asset_behavior_clusters*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else "asset_behavior_clusters.csv"


def _load_ingestion_snapshot(path: Path) -> tuple[dict[str, object] | None, str | None, str | None]:
    try:
        status = load_ingestion_status(path)
        if status is None:
            return None, None, "No ingestion status captured yet."
        _validate_ingestion_status(status)
        return status, None, None
    except Exception as exc:
        return None, f"Could not load ingestion status: {exc}", None


def _validate_ingestion_status(status: object) -> None:
    if not isinstance(status, dict):
        raise ValueError("expected a JSON object")

    required = ("provider", "start", "end", "status", "started_at", "symbols")
    missing = [key for key in required if key not in status]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")

    if not isinstance(status["symbols"], list):
        raise ValueError("symbols must be a list")

    for index, symbol in enumerate(status["symbols"]):
        if not isinstance(symbol, dict):
            raise ValueError(f"symbols[{index}] must be an object")
        for key in ("symbol", "status"):
            if key not in symbol:
                raise ValueError(f"symbols[{index}] missing field: {key}")


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_json_safe_record(record) for record in frame.to_dict(orient="records")]


def _json_safe_record(record: dict[str, object]) -> dict[str, object]:
    return {key: _json_safe_value(value) for key, value in record.items()}


def _json_safe_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict, set)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _summary(walkforward: pd.DataFrame, baselines: pd.DataFrame, clusters: pd.DataFrame) -> dict[str, object]:
    best = walkforward.sort_values("total_return", ascending=False).iloc[0].to_dict() if len(walkforward) else {}
    worst = walkforward.sort_values("total_return", ascending=True).iloc[0].to_dict() if len(walkforward) else {}
    strategy_rows = baselines[baselines["model"] == "state_ev_strategy"] if "model" in baselines.columns else pd.DataFrame()
    beat_count = int((strategy_rows.get("excess_vs_buy_hold", pd.Series(dtype=float)) > 0).sum())
    cluster_counts = clusters["cluster_label"].value_counts().sort_index().to_dict() if "cluster_label" in clusters.columns else {}

    return {
        "bestSymbol": best.get("symbol"),
        "bestTotalReturn": best.get("total_return"),
        "worstSymbol": worst.get("symbol"),
        "worstTotalReturn": worst.get("total_return"),
        "symbols": int(len(walkforward)),
        "beatBuyHoldCount": beat_count,
        "clusterCounts": cluster_counts,
    }


def _render_dashboard_html(data: dict[str, object]) -> str:
    payload = _safe_script_json(data)
    generated_note = escape("Data is embedded from reports/tables at generation time.")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Markov TA Lab Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --surface: #ffffff;
      --surface-2: #eef2f6;
      --text: #17202a;
      --muted: #617080;
      --line: #dbe2ea;
      --accent: #176b87;
      --accent-2: #34a0a4;
      --good: #147d64;
      --bad: #b42318;
      --warn: #9a6700;
      --shadow: 0 14px 34px rgba(23, 32, 42, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    header {{ padding: 28px 32px 18px; border-bottom: 1px solid var(--line); background: var(--surface); position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.15; font-weight: 760; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    p {{ color: var(--muted); margin: 8px 0 0; line-height: 1.5; }}
    main {{ padding: 24px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    .grid {{ display: grid; gap: 16px; }}
    .kpis {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 18px; }}
    .two {{ grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.8fr); }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 18px; min-width: 0; }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi .value {{ font-size: 24px; font-weight: 760; margin-top: 8px; }}
    .kpi .sub {{ color: var(--muted); font-size: 13px; margin-top: 5px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    .tab {{ border: 1px solid var(--line); background: var(--surface); border-radius: 7px; padding: 9px 12px; color: var(--muted); cursor: pointer; font-weight: 650; }}
    .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .chart-wrap {{ width: 100%; overflow-x: auto; }}
    .chart-svg {{ display: block; width: 100%; min-width: 620px; height: auto; }}
    .chart-grid {{ stroke: var(--line); stroke-width: 1; }}
    .zero-axis {{ stroke: #738394; stroke-width: 1.5; }}
    .chart-bar {{ rx: 3; ry: 3; }}
    .chart-bar.pos {{ fill: var(--accent); }}
    .chart-bar.neg {{ fill: #c24135; }}
    .chart-label {{ fill: var(--text); font-size: 12px; font-weight: 700; }}
    .chart-value {{ fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
    .chart-axis-label {{ fill: var(--muted); font-size: 11px; }}
    .chart-empty {{ color: var(--muted); background: #fbfcfd; border: 1px dashed var(--line); border-radius: 8px; padding: 22px; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #fbfcfd; position: sticky; top: 0; z-index: 1; }}
    .table-wrap {{ overflow: auto; max-height: 430px; border: 1px solid var(--line); border-radius: 8px; }}
    .pill {{ display: inline-flex; border-radius: 999px; padding: 4px 9px; background: var(--surface-2); color: var(--muted); font-size: 12px; font-weight: 700; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .note {{ border-left: 3px solid var(--warn); padding: 10px 12px; background: #fff8e6; color: #5d4200; border-radius: 5px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .summary-block {{ background: #fbfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .summary-block h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .summary-block ul {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.55; }}
    .cluster-list {{ display: grid; gap: 10px; }}
    .cluster-item {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--line); }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 20px; }}
    @media (max-width: 980px) {{
      header {{ position: static; padding: 22px 18px 14px; }}
      main {{ padding: 18px; }}
      .kpis, .two, .summary-grid {{ grid-template-columns: 1fr; }}
      th {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Institutional Flow Markov Lab</h1>
    <p>Walk-forward state EV, baseline comparison, volatility-conditioned payoff maps, and behavior clusters. {generated_note}</p>
  </header>
  <main>
    <section class="grid kpis" id="kpis"></section>
    <section class="card" style="margin-bottom:16px;">
      <h2>Plain-English Summary</h2>
      <div class="summary-grid">
        <div class="summary-block">
          <h3>What the results show</h3>
          <ul id="plain-results"></ul>
        </div>
        <div class="summary-block">
          <h3>How to read this</h3>
          <ul>
            <li>Walk-forward results are the cleaner test because they use only prior outcomes.</li>
            <li>Return bars show strategy return, not a promise about future prices.</li>
            <li>Baseline rows show whether the model added value beyond simple alternatives.</li>
          </ul>
        </div>
        <div class="summary-block">
          <h3>What needs more research</h3>
          <ul>
            <li>Pool sparse states by behavior cluster and compare against single-symbol estimates.</li>
            <li>Add Markov probability weighted EV instead of current-state EV only.</li>
            <li>Run sensitivity tests across lookback, horizon, cost, and minimum sample settings.</li>
            <li>Add confidence intervals so noisy states are not over-interpreted.</li>
          </ul>
        </div>
      </div>
    </section>
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab active" data-panel="performance">Performance</button>
      <button class="tab" data-panel="baselines">Baselines</button>
      <button class="tab" data-panel="volatility">Volatility EV</button>
      <button class="tab" data-panel="clusters">Clusters</button>
      <button class="tab" data-panel="ingestion">Ingestion</button>
      <button class="tab" data-panel="research">Research QA</button>
    </nav>
    <section id="performance" class="panel grid two">
      <div class="card">
        <h2>Walk-Forward Performance</h2>
        <div id="performance-bars"></div>
      </div>
      <div class="card">
        <h2>Risk / Exposure</h2>
        <div class="table-wrap"><table id="risk-table"></table></div>
      </div>
    </section>
    <section id="baselines" class="panel card" hidden>
      <h2>Baseline Comparison</h2>
      <div class="table-wrap"><table id="baseline-table"></table></div>
    </section>
    <section id="volatility" class="panel grid two" hidden>
      <div class="card">
        <h2>Volatility-Conditioned EV</h2>
        <div id="vol-bars"></div>
      </div>
      <div class="card">
        <h2>Top Conditional Rows</h2>
        <div class="table-wrap"><table id="vol-table"></table></div>
      </div>
    </section>
    <section id="clusters" class="panel grid two" hidden>
      <div class="card">
        <h2>Asset Behavior Clusters</h2>
        <div id="cluster-list" class="cluster-list"></div>
      </div>
      <div class="card">
        <h2>Cluster Feature Table</h2>
        <div class="table-wrap"><table id="cluster-table"></table></div>
      </div>
    </section>
    <section id="ingestion" class="panel grid two" hidden>
      <div class="card">
        <h2>Ingestion Status</h2>
        <div id="ingestion-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h2>Symbol Rows</h2>
        <div class="table-wrap"><table id="ingestion-table"></table></div>
      </div>
    </section>
    <section id="research" class="panel grid two" hidden>
      <div class="card">
        <h2>Cluster-Pooled EV</h2>
        <div class="table-wrap"><table id="cluster-pooled-table"></table></div>
      </div>
      <div class="card">
        <h2>Markov-Weighted EV</h2>
        <div class="table-wrap"><table id="markov-table"></table></div>
      </div>
      <div class="card" style="grid-column:1 / -1;">
        <h2>Sensitivity Tests</h2>
        <div class="table-wrap"><table id="sensitivity-table"></table></div>
      </div>
      <div class="card">
        <h2>Parameter Stability (per symbol)</h2>
        <p class="note">Higher std / wider IQR ⇒ Sharpe is sensitive to the parameter grid (less robust).</p>
        <div class="table-wrap"><table id="stability-table"></table></div>
      </div>
      <div class="card">
        <h2>Block-Bootstrap Sharpe CI</h2>
        <p class="note">Block size = trade horizon; non-overlapping trades preserved.</p>
        <div class="table-wrap"><table id="sharpe-ci-table"></table></div>
      </div>
    </section>
    <section class="card" style="margin-top:16px;">
      <h2>Interpretation Notes</h2>
      <p class="note">Prefer walk-forward outputs over full-sample prototype outputs. The walk-forward EV filter uses only prior realized outcomes, but it is still current-state based and not yet cluster-pooled or Markov-probability weighted.</p>
      <footer id="sources"></footer>
    </section>
  </main>
  <script>
    const dashboardData = {payload};
    const fmtPct = value => value == null || Number.isNaN(Number(value)) ? "n/a" : `${{(Number(value) * 100).toFixed(1)}}%`;
    const fmtNum = value => value == null || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(3);
    const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));

    function kpis() {{
      const s = dashboardData.summary;
      const items = [
        ["Best symbol", s.bestSymbol ?? "n/a", fmtPct(s.bestTotalReturn)],
        ["Weakest symbol", s.worstSymbol ?? "n/a", fmtPct(s.worstTotalReturn)],
        ["Symbols", s.symbols, "walk-forward universe"],
        ["Beat buy-hold", s.beatBuyHoldCount, "state EV rows"]
      ];
      document.getElementById("kpis").innerHTML = items.map(([label, value, sub]) => `
        <article class="card kpi"><div class="label">${{esc(label)}}</div><div class="value">${{esc(value)}}</div><div class="sub">${{esc(sub)}}</div></article>
      `).join("");
    }}

    function plainSummary() {{
      const s = dashboardData.summary;
      const clusterText = Object.entries(s.clusterCounts || {{}}).map(([label, count]) => `${{count}} ${{label.replaceAll("_", " ")}}`).join(", ");
      const items = [
        `${{s.bestSymbol ?? "The top symbol"}} had the strongest walk-forward strategy return at ${{fmtPct(s.bestTotalReturn)}}.`,
        `${{s.worstSymbol ?? "The weakest symbol"}} had the weakest walk-forward strategy return at ${{fmtPct(s.worstTotalReturn)}}.`,
        `${{s.beatBuyHoldCount}} strategy row(s) beat buy-and-hold in this run, so the model is not yet consistently outperforming passive exposure.`,
        `The current universe splits into behavior families: ${{clusterText || "no cluster data available"}}.`
      ];
      document.getElementById("plain-results").innerHTML = items.map(item => `<li>${{esc(item)}}</li>`).join("");
    }}

    function svgDivergingBars(target, rows, options) {{
      const {{
        valueKey,
        labelKey,
        title,
        valueFormatter = fmtPct,
        maxRows = 12,
        emptyText = "No chart rows available."
      }} = options;
      const container = document.getElementById(target);
      const chartRows = rows
        .map(row => ({{...row, chartValue: Number(row[valueKey])}}))
        .filter(row => Number.isFinite(row.chartValue))
        .slice(0, maxRows);

      if (!chartRows.length) {{
        container.innerHTML = `<div class="chart-empty">${{esc(emptyText)}}</div>`;
        return;
      }}

      const width = 760;
      const left = 132;
      const right = 88;
      const top = 32;
      const rowHeight = 34;
      const bottom = 26;
      const barHeight = 16;
      const plotWidth = width - left - right;
      const zeroX = left + plotWidth / 2;
      const maxAbs = Math.max(...chartRows.map(row => Math.abs(row.chartValue)), 0.000001);
      const height = top + bottom + chartRows.length * rowHeight;
      const titleId = `${{target}}-title`;
      const axisY = height - bottom + 6;
      const truncate = value => {{
        const text = String(value ?? "n/a");
        return text.length > 18 ? `${{text.slice(0, 15)}}...` : text;
      }};

      const bars = chartRows.map((row, index) => {{
        const value = row.chartValue;
        const y = top + index * rowHeight + 5;
        const scaled = Math.max(2, Math.abs(value) / maxAbs * (plotWidth / 2));
        const x = value < 0 ? zeroX - scaled : zeroX;
        const label = row[labelKey] ?? "n/a";
        const formatted = valueFormatter(value);
        const valueX = value < 0 ? Math.max(left + 8, zeroX - scaled - 7) : Math.min(width - right - 8, zeroX + scaled + 7);
        const valueAnchor = value < 0 ? "end" : "start";
        return `
          <text class="chart-label" x="${{left - 10}}" y="${{y + 12}}" text-anchor="end"><title>${{esc(label)}}</title>${{esc(truncate(label))}}</text>
          <rect class="chart-bar ${{value < 0 ? "neg" : "pos"}}" x="${{x.toFixed(2)}}" y="${{y}}" width="${{scaled.toFixed(2)}}" height="${{barHeight}}">
            <title>${{esc(label)}}: ${{esc(formatted)}}</title>
          </rect>
          <text class="chart-value" x="${{valueX.toFixed(2)}}" y="${{y + 12}}" text-anchor="${{valueAnchor}}">${{esc(formatted)}}</text>`;
      }}).join("");

      container.innerHTML = `
        <div class="chart-wrap">
          <svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-labelledby="${{titleId}}">
            <title id="${{titleId}}">${{esc(title)}}</title>
            <line class="chart-grid" x1="${{left}}" x2="${{width - right}}" y1="${{top - 10}}" y2="${{top - 10}}"></line>
            <line class="zero-axis" x1="${{zeroX}}" x2="${{zeroX}}" y1="${{top - 14}}" y2="${{height - bottom}}"></line>
            <line class="chart-grid" x1="${{left}}" x2="${{width - right}}" y1="${{height - bottom}}" y2="${{height - bottom}}"></line>
            ${{bars}}
            <text class="chart-axis-label" x="${{left}}" y="${{axisY}}" text-anchor="start">-${{esc(valueFormatter(maxAbs))}}</text>
            <text class="chart-axis-label" x="${{zeroX}}" y="${{axisY}}" text-anchor="middle">0</text>
            <text class="chart-axis-label" x="${{width - right}}" y="${{axisY}}" text-anchor="end">${{esc(valueFormatter(maxAbs))}}</text>
          </svg>
        </div>`;
    }}

    function table(target, columns, rows) {{
      const head = `<thead><tr>${{columns.map(col => `<th>${{esc(col.label)}}</th>`).join("")}}</tr></thead>`;
      const body = `<tbody>${{rows.map(row => `<tr>${{columns.map(col => `<td>${{esc(col.format ? col.format(row[col.key], row) : row[col.key])}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
      document.getElementById(target).innerHTML = head + body;
    }}

    function renderPerformance() {{
      const rows = [...dashboardData.walkforward].sort((a, b) => Number(b.total_return) - Number(a.total_return));
      svgDivergingBars("performance-bars", rows, {{
        valueKey: "total_return",
        labelKey: "symbol",
        title: "Walk-forward performance total return by symbol"
      }});
      table("risk-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"sharpe", label:"Sharpe", format: fmtNum}},
        {{key:"max_drawdown", label:"Max DD", format: fmtPct}},
        {{key:"win_rate", label:"Win", format: fmtPct}},
        {{key:"exposure_time", label:"Exposure", format: fmtPct}},
        {{key:"trade_count", label:"Trades"}}
      ], rows);
    }}

    function renderBaselines() {{
      table("baseline-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"model", label:"Model"}},
        {{key:"total_return", label:"Return", format: fmtPct}},
        {{key:"excess_vs_buy_hold", label:"Excess vs B/H", format: fmtPct}},
        {{key:"win_rate", label:"Win", format: fmtPct}},
        {{key:"trade_count", label:"Trades"}}
      ], dashboardData.baselines);
    }}

    function renderVolatility() {{
      const rows = [...dashboardData.volConditioned]
        .filter(row => row.count_5 != null && Number(row.count_5) >= 10 && row.ev_after_cost_5 != null)
        .sort((a, b) => Math.abs(Number(b.ev_after_cost_5)) - Math.abs(Number(a.ev_after_cost_5)))
        .slice(0, 16)
        .map(row => ({{...row, key: `${{row.symbol}} V${{row.vol_state}} S${{row.state}}`}}));
      svgDivergingBars("vol-bars", rows, {{
        valueKey: "ev_after_cost_5",
        labelKey: "key",
        title: "Volatility-conditioned expected value after cost",
        maxRows: 10
      }});
      table("vol-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"vol_state", label:"Vol"}},
        {{key:"state", label:"State"}},
        {{key:"label", label:"Label"}},
        {{key:"count_5", label:"N"}},
        {{key:"ev_after_cost_5", label:"EV 5", format: fmtPct}}
      ], rows);
    }}

    function renderClusters() {{
      const grouped = Object.groupBy ? Object.groupBy(dashboardData.clusters, row => row.cluster_label) : dashboardData.clusters.reduce((acc, row) => ((acc[row.cluster_label] ||= []).push(row), acc), {{}});
      document.getElementById("cluster-list").innerHTML = Object.entries(grouped).map(([label, rows]) => `
        <div class="cluster-item"><span><span class="pill">${{esc(label)}}</span></span><strong>${{rows.map(row => esc(row.symbol)).join(", ")}}</strong></div>
      `).join("");
      table("cluster-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"cluster_label", label:"Cluster"}},
        {{key:"realized_volatility", label:"Realized Vol", format: fmtPct}},
        {{key:"trend_persistence", label:"Trend Persist", format: fmtNum}},
        {{key:"reversal_frequency", label:"Reversal", format: fmtPct}},
        {{key:"volume_stability", label:"Volume Stable", format: fmtPct}}
      ], dashboardData.clusters);
    }}

    function renderIngestion() {{
      const summary = document.getElementById("ingestion-summary");
      if (dashboardData.ingestionError) {{
        summary.innerHTML = `<div class="summary-block" style="grid-column:1 / -1;"><h3>Status artifact</h3><p class="note">${{esc(dashboardData.ingestionError)}}</p></div>`;
        document.getElementById("ingestion-table").innerHTML = "";
        return;
      }}
      if (!dashboardData.ingestionStatus) {{
        summary.innerHTML = `<div class="summary-block" style="grid-column:1 / -1;"><h3>Status artifact</h3><p class="note">${{esc(dashboardData.ingestionMessage || "No ingestion status captured yet.")}}</p></div>`;
        document.getElementById("ingestion-table").innerHTML = "";
        return;
      }}

      const status = dashboardData.ingestionStatus;
      summary.innerHTML = [
        ["Status", status.status ?? "n/a", status.run_id ?? ""],
        ["Provider", status.provider ?? "n/a", `${{status.start ?? "n/a"}} to ${{status.end ?? "n/a"}}`],
        ["Started", status.started_at ?? "n/a", `Finished: ${{status.finished_at ?? "n/a"}}`]
      ].map(([label, value, sub]) => `
        <div class="summary-block"><h3>${{esc(label)}}</h3><div class="value">${{esc(value)}}</div><p>${{esc(sub)}}</p></div>
      `).join("");

      const rows = [...(status.symbols || [])].map(row => ({{
        ...row,
        missing_count: row.missing_count ?? row.total_missing ?? 0,
        first_date: row.first_date ?? "n/a",
        last_date: row.last_date ?? "n/a",
        error: row.error ?? ""
      }}));
      table("ingestion-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"status", label:"Status"}},
        {{key:"rows", label:"Rows"}},
        {{key:"first_date", label:"First Date"}},
        {{key:"last_date", label:"Last Date"}},
        {{key:"missing_count", label:"missing_count"}},
        {{key:"error", label:"Error"}}
      ], rows);
    }}

    function renderResearchQA() {{
      const pooled = [...dashboardData.clusterPooled]
        .filter(row => row.ev_after_cost_5 != null)
        .sort((a, b) => Number(b.ev_after_cost_5) - Number(a.ev_after_cost_5))
        .slice(0, 16);
      table("cluster-pooled-table", [
        {{key:"cluster_label", label:"Cluster"}},
        {{key:"state", label:"State"}},
        {{key:"label", label:"Label"}},
        {{key:"count_5", label:"N"}},
        {{key:"ev_after_cost_5", label:"EV 5", format: fmtPct}},
        {{key:"ci_low_5", label:"CI Low", format: fmtPct}},
        {{key:"ci_high_5", label:"CI High", format: fmtPct}}
      ], pooled);

      const markov = [...dashboardData.markovWeighted]
        .filter(row => row.markov_weighted_ev != null)
        .sort((a, b) => Math.abs(Number(b.markov_weighted_ev)) - Math.abs(Number(a.markov_weighted_ev)))
        .slice(0, 20);
      table("markov-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"date", label:"Date"}},
        {{key:"current_state", label:"State"}},
        {{key:"markov_weighted_ev", label:"Weighted EV", format: fmtPct}},
        {{key:"coverage", label:"Coverage", format: fmtPct}},
        {{key:"weighted_samples", label:"Weighted N", format: fmtNum}}
      ], markov);

      const sensitivity = [...dashboardData.sensitivity]
        .sort((a, b) => Number(b.sharpe ?? -999) - Number(a.sharpe ?? -999))
        .slice(0, 24);
      table("sensitivity-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"horizon", label:"H"}},
        {{key:"lookback", label:"Lookback"}},
        {{key:"cost_bps", label:"Cost"}},
        {{key:"min_samples", label:"Min N"}},
        {{key:"total_return", label:"Return", format: fmtPct}},
        {{key:"sharpe", label:"Sharpe", format: fmtNum}},
        {{key:"trade_count", label:"Trades"}}
      ], sensitivity);

      const stability = [...(dashboardData.stability || [])]
        .sort((a, b) => Number(b.sharpe_median ?? -999) - Number(a.sharpe_median ?? -999));
      table("stability-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"n_configs", label:"# Configs"}},
        {{key:"sharpe_median", label:"Median", format: fmtNum}},
        {{key:"sharpe_std", label:"Std", format: fmtNum}},
        {{key:"sharpe_iqr", label:"IQR", format: fmtNum}},
        {{key:"sharpe_min", label:"Min", format: fmtNum}},
        {{key:"sharpe_max", label:"Max", format: fmtNum}},
        {{key:"sharpe_share_negative", label:"Share <0", format: fmtPct}}
      ], stability);

      const sharpeCi = [...(dashboardData.sharpeCi || [])]
        .sort((a, b) => Number(b.sharpe_point ?? -999) - Number(a.sharpe_point ?? -999));
      table("sharpe-ci-table", [
        {{key:"symbol", label:"Symbol"}},
        {{key:"n_trades", label:"# Trades"}},
        {{key:"sharpe_point", label:"Sharpe", format: fmtNum}},
        {{key:"sharpe_ci_low", label:"CI Low", format: fmtNum}},
        {{key:"sharpe_ci_high", label:"CI High", format: fmtNum}},
        {{key:"confidence", label:"Conf", format: fmtPct}}
      ], sharpeCi);
    }}

    document.querySelectorAll(".tab").forEach(button => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(panel => panel.hidden = true);
        button.classList.add("active");
        document.getElementById(button.dataset.panel).hidden = false;
      }});
    }});

    document.getElementById("sources").textContent = `Sources: ${{Object.values(dashboardData.sources).join(", ")}}`;
    kpis();
    plainSummary();
    renderPerformance();
    renderBaselines();
    renderVolatility();
    renderClusters();
    renderIngestion();
    renderResearchQA();
  </script>
</body>
</html>
"""


def _safe_script_json(data: dict[str, object]) -> str:
    """Serialize JSON for embedding in an inline script tag."""
    return (
        json.dumps(data, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
