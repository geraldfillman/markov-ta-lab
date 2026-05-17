"""Tests for static dashboard generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.dashboard import generate_dashboard


def _write_dashboard_inputs(tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True)
    (tables_dir / "walkforward_backtest_summary.csv").write_text(
        "\n".join(
            [
                "symbol,total_return,sharpe,max_drawdown,win_rate,exposure_time,trade_count",
                "SPY,0.12,1.2,-0.05,0.55,0.40,8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tables_dir / "walkforward_baseline_comparison.csv").write_text(
        "\n".join(
            [
                "symbol,model,total_return,excess_vs_buy_hold,win_rate,trade_count",
                "SPY,state_ev_strategy,0.12,0.02,0.55,8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tables_dir / "vol_conditioned_state_expectancy.csv").write_text(
        "\n".join(
            [
                "symbol,vol_state,state,label,count_5,ev_after_cost_5",
                "SPY,low,1,up,11,0.01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tables_dir / "asset_behavior_clusters.csv").write_text(
        "\n".join(
            [
                "symbol,cluster_label,realized_volatility,trend_persistence,reversal_frequency,volume_stability",
                "SPY,trend,0.18,0.60,0.20,0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _embedded_payload(html: str) -> dict[str, object]:
    match = re.search(r"const dashboardData = (\{.*?\});", html, flags=re.S)
    assert match is not None
    return json.loads(match.group(1))


def test_missing_ingestion_status_does_not_fail_and_embeds_empty_status(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)

    output_path = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir)

    html = output_path.read_text(encoding="utf-8")
    payload = _embedded_payload(html)
    assert output_path.exists()
    assert payload["ingestionStatus"] is None
    assert payload["ingestionMessage"] == "No ingestion status captured yet."
    assert "Ingestion" in html
    assert "No ingestion status captured yet." in html


def test_valid_ingestion_status_is_embedded_and_rendered(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)
    (tables_dir / "ingestion_status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "provider": "yfinance",
                "start": "2024-01-01",
                "end": "2024-01-05",
                "status": "partial",
                "started_at": "2024-01-05T10:00:00Z",
                "finished_at": "2024-01-05T10:00:03Z",
                "symbols": [
                    {
                        "symbol": "SPY",
                        "status": "success",
                        "rows": 3,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-04",
                        "total_missing": 1,
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir).read_text(encoding="utf-8")
    payload = _embedded_payload(html)

    assert payload["ingestionStatus"]["provider"] == "yfinance"  # type: ignore[index]
    assert payload["ingestionStatus"]["symbols"][0]["symbol"] == "SPY"  # type: ignore[index]
    assert "Ingestion Status" in html
    assert "ingestion-table" in html
    assert "missing_count" in html


def test_malformed_ingestion_status_embeds_error_and_still_writes_dashboard(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)
    (tables_dir / "ingestion_status.json").write_text("{not json", encoding="utf-8")

    output_path = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir)
    html = output_path.read_text(encoding="utf-8")
    payload = _embedded_payload(html)

    assert output_path.exists()
    assert payload["ingestionStatus"] is None
    assert "ingestionError" in payload
    assert "Could not load ingestion status" in payload["ingestionError"]  # type: ignore[operator]
    assert "Could not load ingestion status" in html


def test_invalid_ingestion_status_schema_embeds_error_and_still_writes_dashboard(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)
    (tables_dir / "ingestion_status.json").write_text(
        json.dumps({"provider": "yfinance", "symbols": "SPY"}),
        encoding="utf-8",
    )

    output_path = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir)
    payload = _embedded_payload(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert payload["ingestionStatus"] is None
    assert "missing field" in payload["ingestionError"]  # type: ignore[operator]


def test_embedded_payload_cannot_close_inline_script(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)
    hostile_error = "</script><script>alert('xss')</script>"
    (tables_dir / "ingestion_status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "provider": "yfinance",
                "start": "2024-01-01",
                "end": "2024-01-05",
                "status": "failed",
                "started_at": "2024-01-05T10:00:00Z",
                "finished_at": "2024-01-05T10:00:03Z",
                "symbols": [
                    {
                        "symbol": "SPY",
                        "status": "error",
                        "rows": 0,
                        "first_date": None,
                        "last_date": None,
                        "total_missing": 0,
                        "error": hostile_error,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir).read_text(encoding="utf-8")
    payload = _embedded_payload(html)

    assert hostile_error not in html
    assert "\\u003c/script\\u003e" in html
    assert payload["ingestionStatus"]["symbols"][0]["error"] == hostile_error  # type: ignore[index]


def test_dashboard_uses_accessible_svg_charts_with_zero_axis_and_empty_state(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)

    html = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir).read_text(encoding="utf-8")

    assert "function svgDivergingBars" in html
    assert '<svg class="chart-svg"' in html
    assert 'role="img"' in html
    assert 'aria-labelledby="${titleId}"' in html
    assert '<title id="${titleId}">${esc(title)}</title>' in html
    assert 'class="zero-axis"' in html
    assert "No chart rows available." in html
    assert "performance-bars" in html
    assert "vol-bars" in html


def test_svg_chart_reserves_value_label_gutters_for_full_scale_bars(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    output_dir = tmp_path / "dashboard"
    _write_dashboard_inputs(tables_dir)
    (tables_dir / "walkforward_backtest_summary.csv").write_text(
        "\n".join(
            [
                "symbol,total_return,sharpe,max_drawdown,win_rate,exposure_time,trade_count",
                "MAXPOS_LONG_LABEL,12.345,1.2,-0.05,0.55,0.40,8",
                "MAXNEG_LONG_LABEL,-12.345,-1.2,-0.50,0.45,0.35,9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    html = generate_dashboard(tables_dir=tables_dir, output_dir=output_dir).read_text(encoding="utf-8")
    payload = _embedded_payload(html)

    assert "const valueGutter = 78;" in html
    assert "const plotLeft = left + valueGutter;" in html
    assert "const plotRight = width - right - valueGutter;" in html
    assert 'class="chart-value chart-value-${value < 0 ? "neg" : "pos"}"' in html
    assert 'data-label-side="${value < 0 ? "left-gutter" : "right-gutter"}"' in html
    assert "const valueX = value < 0 ? plotLeft - 8 : plotRight + 8;" in html
    assert 'text-anchor="${valueAnchor}"' in html
    assert 'lengthAdjust="spacingAndGlyphs"' in html
    assert [row["total_return"] for row in payload["walkforward"]] == [12.345, -12.345]  # type: ignore[index]
