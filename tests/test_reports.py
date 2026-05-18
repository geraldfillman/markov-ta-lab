"""Tests for experiment report generation."""

from pathlib import Path
from uuid import uuid4

from src.reports import generate_report


def _repo_tmp_path(name: str) -> Path:
    path = Path(".test_output") / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_generate_report_writes_markdown_with_defaults():
    output_dir = _repo_tmp_path("report_generation")

    path = generate_report(
        "first_state_expectancy",
        {
            "question": "Do deterministic states have different forward returns?",
            "hypothesis": "Some states should show better cost-adjusted expectancy.",
            "results": "State expectancy table generated.",
        },
        output_dir=output_dir,
    )

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")

    assert report_path.name == "first_state_expectancy.md"
    assert "# Experiment Report: first_state_expectancy" in text
    assert "Do deterministic states have different forward returns?" in text
    assert "State expectancy table generated." in text
    assert "Not evaluated in this run." in text
