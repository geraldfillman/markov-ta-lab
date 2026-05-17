"""Static dashboard generation for Markov TA Lab reports."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR


def generate_dashboard(
    tables_dir: str | Path = TABLES_DIR,
    output_dir: str | Path = "reports/dashboard",
) -> Path:
    """Generate a self-contained HTML dashboard from report CSV tables."""
    tables_path = Path(tables_dir)
    output_path = Path(output_dir) / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_dashboard_data(tables_path)
    output_path.write_text(_render_dashboard_html(data), encoding="utf-8")
    return output_path


def _load_dashboard_data(tables_dir: Path) -> dict[str, object]:
    walkforward = _read_csv(tables_dir / "walkforward_backtest_summary.csv")
    baselines = _read_csv(tables_dir / "walkforward_baseline_comparison.csv")
    vol_conditioned = _read_csv(tables_dir / "vol_conditioned_state_expectancy.csv")
    clusters = _read_latest_clusters(tables_dir)
    cluster_pooled = _read_optional_csv(tables_dir / "cluster_pooled_state_expectancy.csv")
    markov_weighted = _read_optional_csv(tables_dir / "markov_weighted_ev.csv")
    sensitivity = _read_optional_csv(tables_dir / "walkforward_sensitivity.csv")
    stability = _read_optional_csv(tables_dir / "sensitivity_stability_summary.csv")
    sharpe_ci = _read_optional_csv(tables_dir / "walkforward_sharpe_ci.csv")

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
    payload = json.dumps(data, allow_nan=False)
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
    .bar-row {{ display: grid; grid-template-columns: 54px minmax(0, 1fr) 86px; gap: 12px; align-items: center; margin: 10px 0; }}
    .bar-track {{ height: 16px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); min-width: 2px; }}
    .neg {{ background: linear-gradient(90deg, #e26d5c, #b42318); }}
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

    function bars(target, rows, valueKey, labelKey, valueFormatter = fmtPct) {{
      const values = rows.map(row => Number(row[valueKey] ?? 0));
      const max = Math.max(...values.map(Math.abs), 0.000001);
      document.getElementById(target).innerHTML = rows.map(row => {{
        const value = Number(row[valueKey] ?? 0);
        const width = Math.max(2, Math.abs(value) / max * 100);
        return `<div class="bar-row"><strong>${{esc(row[labelKey])}}</strong><div class="bar-track"><div class="bar ${{value < 0 ? "neg" : ""}}" style="width:${{width}}%"></div></div><span class="${{value < 0 ? "bad" : "good"}}">${{valueFormatter(value)}}</span></div>`;
      }}).join("");
    }}

    function table(target, columns, rows) {{
      const head = `<thead><tr>${{columns.map(col => `<th>${{esc(col.label)}}</th>`).join("")}}</tr></thead>`;
      const body = `<tbody>${{rows.map(row => `<tr>${{columns.map(col => `<td>${{esc(col.format ? col.format(row[col.key], row) : row[col.key])}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
      document.getElementById(target).innerHTML = head + body;
    }}

    function renderPerformance() {{
      const rows = [...dashboardData.walkforward].sort((a, b) => Number(b.total_return) - Number(a.total_return));
      bars("performance-bars", rows, "total_return", "symbol");
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
      bars("vol-bars", rows.slice(0, 10), "ev_after_cost_5", "key");
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
    renderResearchQA();
  </script>
</body>
</html>
"""
